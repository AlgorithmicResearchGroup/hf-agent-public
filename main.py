#!/usr/bin/env python3
"""Hugging Face model advisor entrypoint."""
import os
import sys
import json
import time
import threading
from collections import deque
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

from agent.worker import Worker


def main():
    """Main entry point for agent"""
    task_instructions = ""

    if len(sys.argv) > 1:
        task_instructions = " ".join(sys.argv[1:])
    else:
        task_instructions = os.environ.get("TASK_DESCRIPTION", "")

    if not task_instructions:
        print("Usage: python main.py '<task description>' or set TASK_DESCRIPTION env var")
        return

    print(f"Hugging Face Model Advisor | Task: {task_instructions[:100]}...", flush=True)

    # Load configuration from manifest.json
    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]

    model_name = os.environ.get("AGENT_MODEL", settings["model"])
    max_iterations = int(os.environ.get("MAX_ITERATIONS", settings["max_iterations"]))
    provider = os.environ.get("PROVIDER", settings["provider"])
    max_runtime_seconds = int(os.environ.get("MAX_RUNTIME_SECONDS", settings["max_runtime_seconds"]))
    max_tokens = int(os.environ.get("MAX_TOKENS", settings["max_tokens"]))

    print(f"\nConfig: {settings_pack_name} | Model: {model_name} | Max: {max_runtime_seconds}s / {max_iterations} steps", flush=True)

    agents_md_path = os.path.join(os.getcwd(), "Agents.md")
    agents_md = ""
    if os.path.exists(agents_md_path):
        with open(agents_md_path, "r") as f:
            agents_md = f.read()
        print(f"Loaded Agents.md ({len(agents_md)} chars)", flush=True)

    run_id = int(time.time() * 1000)

    shared_workspace = os.environ.get("SHARED_WORKSPACE", "")
    if shared_workspace:
        work_dir = shared_workspace
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = os.path.join(os.path.dirname(__file__), "runs", str(run_id))
        os.makedirs(work_dir)
    print(f"Working directory: {work_dir}", flush=True)

    # --- Protocol setup ---
    protocol_enabled = os.environ.get("PROTOCOL_ENABLED", "false").lower() == "true"
    protocol_agent = None
    broker = None
    protocol_config = None
    message_buffer = None
    message_lock = None

    if protocol_enabled:
        from agent_protocol.agent import Agent as ProtocolAgent
        from agent_protocol.broker import MessageBroker
        from agent.tool_registry import _protocol_state

        agent_id = os.environ.get("AGENT_ID", f"agent-{run_id}")
        broker_router = os.environ.get("BROKER_ROUTER", os.environ.get("BROKER_PUSH", "tcp://localhost:5555"))
        broker_sub = os.environ.get("BROKER_SUB", "tcp://localhost:5556")
        broker_mode = os.environ.get("BROKER_MODE", "connect")
        work_queue_enabled = os.environ.get("WORK_QUEUE_ENABLED", "false").lower() == "true"
        topics = [t.strip() for t in os.environ.get("AGENT_TOPICS", "general").split(",")]

        message_buffer = deque()
        message_lock = threading.Lock()

        def message_handler(message):
            with message_lock:
                message_buffer.append(message)

        if broker_mode == "host":
            broker = MessageBroker(enable_logging=False)
            broker.start()
            time.sleep(0.5)
            print(f"Protocol broker started (hosting)", flush=True)

        protocol_agent = ProtocolAgent(
            agent_id=agent_id,
            broker_router=broker_router,
            broker_sub=broker_sub,
            topics=topics,
            message_handler=message_handler,
            enable_logging=False,
        )
        protocol_agent.start()
        time.sleep(0.3)

        _protocol_state["agent"] = protocol_agent
        _protocol_state["message_buffer"] = message_buffer
        _protocol_state["message_lock"] = message_lock

        protocol_config = {"agent_id": agent_id, "topics": topics, "work_queue_enabled": work_queue_enabled}

        print(f"Protocol enabled: agent_id={agent_id}, topics={topics}, broker_mode={broker_mode}, work_queue={work_queue_enabled}", flush=True)

    work_queue_enabled = protocol_config.get("work_queue_enabled", False) if protocol_config else False

    worker = Worker(
        user_id=1,
        run_id=run_id,
        user_query=task_instructions,
        worker_number=1,
        provider=provider,
        model_name=model_name,
        max_tokens=max_tokens,
        work_dir=work_dir,
        agents_md=agents_md,
        protocol_enabled=protocol_enabled,
        protocol_config=protocol_config,
        message_buffer=message_buffer,
        message_lock=message_lock,
        shared_workspace=shared_workspace,
        work_queue_enabled=work_queue_enabled,
    )

    print(f"Running worker with {provider} provider and {model_name} model", flush=True)

    start_time = time.time()

    for step in range(max_iterations):
        elapsed_seconds = time.time() - start_time
        remaining_seconds = max_runtime_seconds - elapsed_seconds

        if remaining_seconds <= 0:
            print(f"\nTIMEOUT REACHED after {elapsed_seconds:.1f} seconds!", flush=True)
            break

        print(f"\n[Step {step + 1}/{max_iterations}] {remaining_seconds / 60:.0f}m remaining")

        elapsed_time = timedelta(seconds=elapsed_seconds)
        result = worker.run_step(elapsed_time)

        subtask_result = result.get("subtask_result", {})
        if isinstance(subtask_result, dict) and subtask_result.get("tool") == "done":
            print(f"\nAgent finished: {subtask_result.get('stdout', '')}", flush=True)
            break

    # --- Protocol cleanup ---
    if protocol_agent:
        protocol_agent.stop()
    if broker:
        broker.stop()

    total_tokens = worker.total_prompt_tokens + worker.total_response_tokens
    print(f"\nTokens: {total_tokens:,} ({worker.total_prompt_tokens:,} in / {worker.total_response_tokens:,} out) | Cost: ${worker.total_cost:.4f} | Steps: {step + 1}")
    print("Agent execution completed.", flush=True)


if __name__ == "__main__":
    main()
