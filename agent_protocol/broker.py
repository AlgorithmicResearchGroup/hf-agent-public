import zmq
import threading
import time
import logging
from collections import deque
from typing import Optional, Dict, Any, List
from .messages import Message, MessageType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageBroker:
    def __init__(
        self,
        router_port: int = 5555,
        pub_port: int = 5556,
        bind_address: str = "*",
        enable_logging: bool = True,
        pull_port: int = None,  # backward compat alias
    ):
        self.router_port = pull_port or router_port
        self.pub_port = pub_port
        self.bind_address = bind_address
        self.enable_logging = enable_logging

        self.context = zmq.Context()
        self.router_socket = None
        self.pub_socket = None
        self.running = False
        self.thread = None

        # Agent registry: agent_id -> zmq identity bytes
        self.agent_registry: Dict[str, bytes] = {}

        # Last Value Cache: topic -> Message
        self.lvc: Dict[str, Message] = {}

        # Work queue
        self.task_queue: deque = deque()
        self.task_log: List[Dict[str, Any]] = []

        self.stats = {
            "messages_received": 0,
            "messages_broadcast": 0,
            "messages_routed": 0,
            "start_time": None,
            "connected_agents": set(),
            "tasks_submitted": 0,
            "tasks_assigned": 0,
        }

    def start(self) -> None:
        logger.info(f"Starting message broker on ROUTER:{self.router_port}, PUB:{self.pub_port}")

        self.router_socket = self.context.socket(zmq.ROUTER)
        self.router_socket.bind(f"tcp://{self.bind_address}:{self.router_port}")

        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://{self.bind_address}:{self.pub_port}")

        self.running = True
        self.stats["start_time"] = time.time()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

        logger.info("Message broker started successfully")

    def _run(self) -> None:
        poller = zmq.Poller()
        poller.register(self.router_socket, zmq.POLLIN)

        while self.running:
            socks = dict(poller.poll(timeout=100))
            if self.router_socket in socks:
                frames = self.router_socket.recv_multipart()
                # ROUTER: [identity, empty_delimiter, message_bytes]
                identity = frames[0]
                message_bytes = frames[-1]
                self._handle_router_message(identity, message_bytes)

    def _handle_router_message(self, identity: bytes, message_bytes: bytes) -> None:
        message = Message.from_bytes(message_bytes)

        self.stats["messages_received"] += 1
        self.stats["connected_agents"].add(message.agent_id)

        if self.enable_logging:
            logger.info(f"Received from {message.agent_id}: {message}")

        msg_type = message.message_type

        if msg_type == MessageType.REGISTER:
            self._handle_register(identity, message)
        elif msg_type == MessageType.TASK_SUBMIT:
            self._handle_task_submit(identity, message)
        elif msg_type == MessageType.TASK_REQUEST:
            self._handle_task_request(identity, message)
        elif msg_type == MessageType.TASK_COMPLETE:
            self._handle_task_complete(identity, message)
        elif msg_type == MessageType.CONTROL:
            self._handle_control_message(message)
            self._broadcast(message)
        elif message.target:
            self._route_to_agent(message)
        else:
            self._broadcast(message)

    def _handle_register(self, identity: bytes, message: Message) -> None:
        agent_id = message.agent_id
        topics = message.payload.get("subscribed_topics", [])

        self.agent_registry[agent_id] = identity
        logger.info(f"Registered agent: {agent_id} (topics: {topics})")

        # ACK via ROUTER
        ack = Message(
            agent_id="broker",
            message_type=MessageType.ACK,
            payload={"status": "registered", "agent_id": agent_id},
            topic="control",
        )
        self.router_socket.send_multipart([identity, b"", ack.to_bytes()])

        # LVC replay for subscribed topics
        for topic in topics:
            if topic in self.lvc:
                cached = self.lvc[topic]
                replay = Message(
                    agent_id=cached.agent_id,
                    message_type=cached.message_type,
                    payload=cached.payload,
                    topic=cached.topic,
                    metadata={**cached.metadata, "lvc_replay": True},
                    timestamp=cached.timestamp,
                )
                self.router_socket.send_multipart([identity, b"", replay.to_bytes()])

        # Also broadcast as DISCOVERY for PUB/SUB subscribers
        self._broadcast(message)

    def _broadcast(self, message: Message) -> None:
        """Broadcast via PUB socket. Update LVC for DATA messages."""
        if message.topic:
            topic_prefix = f"{message.topic}:".encode('utf-8')
            self.pub_socket.send_multipart([topic_prefix, message.to_bytes()])
        else:
            self.pub_socket.send(message.to_bytes())

        if message.message_type == MessageType.DATA and message.topic:
            self.lvc[message.topic] = message

        self.stats["messages_broadcast"] += 1

    def _route_to_agent(self, message: Message) -> None:
        """Route a directed message to a specific agent via ROUTER."""
        target_id = message.target
        identity = self.agent_registry[target_id]  # KeyError = fail hard
        self.router_socket.send_multipart([identity, b"", message.to_bytes()])
        self.stats["messages_routed"] += 1

    def _handle_task_submit(self, identity: bytes, message: Message) -> None:
        task_entry = {
            "task_id": message.message_id,
            "submitted_by": message.agent_id,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "status": "pending",
            "assigned_to": None,
        }
        self.task_queue.append(task_entry)
        self.task_log.append(task_entry)
        self.stats["tasks_submitted"] += 1

        ack = Message(
            agent_id="broker",
            message_type=MessageType.ACK,
            payload={"status": "task_queued", "task_id": message.message_id},
            topic="control",
        )
        self.router_socket.send_multipart([identity, b"", ack.to_bytes()])

    def _handle_task_request(self, identity: bytes, message: Message) -> None:
        if self.task_queue:
            task_entry = self.task_queue.popleft()
            task_entry["status"] = "assigned"
            task_entry["assigned_to"] = message.agent_id
            self.stats["tasks_assigned"] += 1

            assign = Message(
                agent_id="broker",
                message_type=MessageType.TASK_ASSIGN,
                payload=task_entry,
                topic="control",
            )
            self.router_socket.send_multipart([identity, b"", assign.to_bytes()])
        else:
            empty = Message(
                agent_id="broker",
                message_type=MessageType.TASK_ASSIGN,
                payload={"status": "no_tasks"},
                topic="control",
            )
            self.router_socket.send_multipart([identity, b"", empty.to_bytes()])

    def _handle_task_complete(self, identity: bytes, message: Message) -> None:
        task_id = message.payload.get("task_id")
        for entry in self.task_log:
            if entry["task_id"] == task_id:
                entry["status"] = "completed"
                break

        ack = Message(
            agent_id="broker",
            message_type=MessageType.ACK,
            payload={"status": "task_completed", "task_id": task_id},
            topic="control",
        )
        self.router_socket.send_multipart([identity, b"", ack.to_bytes()])

    def _handle_control_message(self, message: Message) -> None:
        if message.payload.get("command") == "stats":
            stats_message = Message(
                agent_id="broker",
                message_type=MessageType.CONTROL,
                payload={"stats": self.get_stats()},
                topic="broker",
            )
            self.pub_socket.send(stats_message.to_bytes())

    def stop(self) -> None:
        logger.info("Stopping message broker...")
        self.running = False

        if self.thread:
            self.thread.join(timeout=2)

        if self.router_socket:
            self.router_socket.close()
        if self.pub_socket:
            self.pub_socket.close()

        logger.info("Message broker stopped")

    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.stats["start_time"] if self.stats["start_time"] else 0
        return {
            "messages_received": self.stats["messages_received"],
            "messages_broadcast": self.stats["messages_broadcast"],
            "messages_routed": self.stats["messages_routed"],
            "connected_agents": len(self.stats["connected_agents"]),
            "unique_agents": list(self.stats["connected_agents"]),
            "registered_agents": list(self.agent_registry.keys()),
            "tasks_submitted": self.stats["tasks_submitted"],
            "tasks_assigned": self.stats["tasks_assigned"],
            "tasks_pending": len(self.task_queue),
            "uptime_seconds": uptime,
        }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.context.term()
