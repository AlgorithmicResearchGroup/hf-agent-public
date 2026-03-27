#!/usr/bin/env python3
"""
Multi-agent Hugging Face model advisor orchestrator.

Usage:
  python run_collab_long.py "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
  python run_collab_long.py --prompts challenge_prompts.json --prompt 2

Env vars:
  MAX_WAVES           QA/fix retry waves (default 2, 0 = skip QA)
  QA_ITERATIONS       QA agent iteration budget (default 30)
  FIX_ITERATIONS      Fix agent iteration budget (default 15)
  FIX_RUNTIME_SECONDS Fix agent runtime budget (default 120)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv

from artifact_publisher import build_artifacts_info, upload_workspace_to_bucket, write_artifacts_manifest
from agent.utils import anthropic_to_openai
from agent_protocol.broker import MessageBroker

load_dotenv()


DECOMPOSE_TOOL = {
    "name": "decompose_task",
    "description": "Decompose a Hugging Face model-selection or fine-tuning request into 2-4 concrete agent assignments for parallel execution.",
    "input_schema": {
        "type": "object",
        "properties": {
            "agents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Short lowercase agent identifier such as 'alice' or 'bob'.",
                        },
                        "role": {
                            "type": "string",
                            "description": "One-line role summary.",
                        },
                        "task": {
                            "type": "string",
                            "description": "Detailed task description with concrete output files and search goals.",
                        },
                        "depends_on": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of agent IDs this agent depends on.",
                        },
                    },
                    "required": ["id", "role", "task", "depends_on"],
                },
            },
        },
        "required": ["agents"],
    },
}

ASSIGN_FIXES_TOOL = {
    "name": "assign_fixes",
    "description": "Assign QA issues to the original agents most likely to fix them.",
    "input_schema": {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"},
                        "error_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "fix_task": {"type": "string"},
                    },
                    "required": ["agent_id", "error_ids", "fix_task"],
                },
            },
        },
        "required": ["assignments"],
    },
}

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are a technical project manager decomposing Hugging Face model-selection, fine-tuning, and capability-evaluation requests into concrete parallel agent assignments.

The user asks for outputs like:
- "I have 1x A100 80GB and want to fine-tune a coding model for repository Q&A. What model on Hugging Face should I use?"
- "I have 2x L4s and want a multilingual embedding model for retrieval. Which Hugging Face model should I start from?"
- "For a document VQA job with limited budget, which open model on Hugging Face should I fine-tune and how?"

Each agent is an autonomous coding/research model with access to:
- shell commands
- local file read/write/edit
- web search and page fetching
- Hugging Face Hub search and repo inspection
- inter-agent messaging

Agents share one working directory.

Default role patterns for these tasks:
- Hugging Face landscape agent: find the most relevant model repos, benchmarks, licenses, and supporting docs. Output `huggingface_ecosystem.md`.
- Hardware / training strategy agent: map the user's hardware, budget, and task constraints to feasible training approaches and tradeoffs. Output `training_plan.md`.
- Implementation agent: depends on the prior artifacts, extracts practical takeaways, and produces small runnable code snippets or examples prepared for direct embedding into the final report. Output `code_examples.md` and optional `snippets/`.
- Synthesis agent: depends on the others and writes `report.md`, embedding the final annotated snippets directly in the report.

Rules:
1. Create 2-4 agents only.
2. Use the minimum number needed. Do not create trivial roles.
3. Keep dependency chains shallow, but do not let implementation run ahead of the landscape and training-plan work.
4. Every agent must write one or more concrete output files in the shared workspace.
5. Be specific about what each agent should research, what files they should read, and what files they should produce.
6. For "latest" or "recent" topics, require live retrieval and date-aware synthesis.
7. Research agents should gather only enough sources to produce a first useful artifact, then write it. Do not spend the full budget searching.
8. For arXiv links, prefer readable HTML sources such as ar5iv or abstract pages instead of raw PDFs.
9. The implementation agent should usually read `huggingface_ecosystem.md` and `training_plan.md` before writing broad snippets or recommendations.
10. The final report must be Markdown and should include:
   - Executive Summary
   - Recommended Model
   - Alternatives and Tradeoffs
   - Hardware / Fine-Tuning Fit
   - Hugging Face Context
   - Hugging Face Models / Techniques
   - Code Snippets
   - Caveats / Open Questions
   - References
11. The final report must embed at least one code snippet directly in `report.md` and each embedded snippet must include a line-by-line explanation.
12. If the user gives hardware or budget constraints, the final report must convert them into a concrete recommendation about model size, fine-tuning method, and likely operational limits.
13. If the user asks about a model or technique, explicitly connect it to named Hugging Face assets such as model repos, libraries, kernels, docs, datasets, or Spaces when relevant.
14. Do not invent a separate QA agent in the decomposition. QA is handled by the orchestrator.

Call the decompose_task tool with the final agent assignments."""

ASSIGN_FIXES_SYSTEM_PROMPT = """\
You are assigning QA failures back to the original agents in a Hugging Face model-advisor build.

Given the original agent roles/tasks and a QA error report, assign each error to the agent best positioned to fix it.
Write fix tasks that reference the affected files and the concrete issue to address."""

QA_AGENT_TASK_TEMPLATE = """\
You are the QA agent for a multi-agent Hugging Face model-advisor run.

ORIGINAL TASK:
{original_task}

AGENTS:
{agent_summaries}

WORKSPACE FILES:
{file_listing}

Your job is to verify the final recommendation deliverable and any generated snippets.

Testing protocol:
1. Read the key files, especially `report.md` if it exists.
2. Verify that the final report exists and is substantive.
3. Verify that the report contains:
   - an executive summary
   - a concrete recommended model or clearly ranked recommendation set
   - hardware / fine-tuning fit analysis when the task provides resource constraints
   - current references or source URLs
   - a Hugging Face section or equivalent repo/model context
   - an explicit Hugging Face models / techniques discussion when the task is about a model or technique
   - at least one code snippet embedded in `report.md` or an explicit implementation note when appropriate
   - a line-by-line explanation for each embedded code snippet
4. If there are standalone `.py` files in `snippets/` or elsewhere, run lightweight smoke checks such as `python -m py_compile`.
5. Only fail for genuinely broken deliverables: missing report, empty report, no actionable recommendation, no references, generic-only Hugging Face coverage, missing embedded snippet explanations, obviously broken snippet files, or major omissions.

Write a file named `qa_report.json` with this schema:
{{
  "status": "pass" or "fail",
  "summary": "one paragraph summary",
  "errors": [
    {{
      "id": "err_1",
      "severity": "critical" or "warning",
      "category": "missing_deliverable|missing_citations|runtime_crash|test_failure|integration_bug",
      "file": "affected file",
      "description": "what is wrong",
      "evidence": "what you observed",
      "suggested_fix": "how to fix it"
    }}
  ],
  "files_tested": ["..."],
  "commands_run": [{{"cmd": "command", "exit_code": 0}}]
}}

Critical rules:
- You must actually inspect the outputs and run checks when code files exist.
- Write `qa_report.json` before calling done.
- If everything looks good, set status to `pass` and errors to an empty list."""

FIX_AGENT_TASK_TEMPLATE = """\
You are {agent_id} on a fix iteration.

YOUR ORIGINAL ROLE: {original_role}

ERRORS TO FIX:
{error_details}

FIX INSTRUCTIONS:
{fix_task}

Rules:
- Read the relevant files first.
- Make focused fixes only.
- Re-run a lightweight sanity check for the files you changed.
- Call done when the assigned issues are fixed."""


def call_orchestrator_openai(model_name, task):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this Hugging Face model-selection task into agent assignments:\n\n{task}"},
        ],
        tools=[anthropic_to_openai(DECOMPOSE_TOOL)],
        parallel_tool_calls=False,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


def call_orchestrator_anthropic(model_name, task):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
    response = client.messages.create(
        model=model_name,
        system=ORCHESTRATOR_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Decompose this Hugging Face model-selection task into agent assignments:\n\n{task}"},
        ],
        max_tokens=4096,
        tools=[DECOMPOSE_TOOL],
        tool_choice={"type": "tool", "name": "decompose_task"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Anthropic response contained no tool_use block")


def _build_assign_fixes_prompt(report, agents):
    agent_lines = []
    for agent in agents:
        agent_lines.append(f"- {agent['id']} ({agent['role']}): {agent['task'][:240]}...")
    error_lines = []
    for err in report["errors"]:
        error_lines.append(
            f"- {err['id']} [{err['severity']}] in {err['file']}: {err['description']}\n"
            f"  Evidence: {err['evidence'][:240]}\n"
            f"  Suggested fix: {err.get('suggested_fix', 'N/A')}"
        )
    return (
        "ORIGINAL AGENTS:\n"
        + "\n".join(agent_lines)
        + "\n\nQA ERRORS:\n"
        + "\n".join(error_lines)
        + "\n\nAssign each error to the best agent and use the assign_fixes tool."
    )


def call_assign_fixes_openai(model_name, report, agents):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI"))
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": ASSIGN_FIXES_SYSTEM_PROMPT},
            {"role": "user", "content": _build_assign_fixes_prompt(report, agents)},
        ],
        tools=[anthropic_to_openai(ASSIGN_FIXES_TOOL)],
        parallel_tool_calls=False,
    )
    tool_call = response.choices[0].message.tool_calls[0]
    return json.loads(tool_call.function.arguments)


def call_assign_fixes_anthropic(model_name, report, agents):
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC"))
    response = client.messages.create(
        model=model_name,
        system=ASSIGN_FIXES_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _build_assign_fixes_prompt(report, agents)},
        ],
        max_tokens=4096,
        tools=[ASSIGN_FIXES_TOOL],
        tool_choice={"type": "tool", "name": "assign_fixes"},
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise RuntimeError("Anthropic response contained no tool_use block")


def topological_waves(agents):
    remaining = {agent["id"]: set(agent["depends_on"]) for agent in agents}
    agent_by_id = {agent["id"]: agent for agent in agents}
    waves = []
    while remaining:
        ready = [agent_id for agent_id, deps in remaining.items() if not deps]
        if not ready:
            raise RuntimeError(f"Cyclic or invalid dependencies in agent graph: {remaining}")
        waves.append([agent_by_id[agent_id] for agent_id in ready])
        for agent_id in ready:
            del remaining[agent_id]
        for deps in remaining.values():
            deps -= set(ready)
    return waves


def stabilize_agent_graph(agents):
    def role_text(agent):
        return agent["role"].lower()

    def task_text(agent):
        return agent["task"].lower()

    def is_impl(agent):
        role = role_text(agent)
        if (
            re.search(r"\bimplementation\b", role)
            or re.search(r"\bsnippet\b", role)
            or re.search(r"\bcode\b", role)
            or re.search(r"\bpractical\b", role)
        ):
            return True
        text = task_text(agent)
        return any(word in text for word in ("runnable snippet", "code example", "write code_examples.md"))

    def is_synthesis(agent):
        role = role_text(agent)
        if (
            "synthesis" in role
            or "report writer" in role
            or "report synthesizer" in role
            or "final recommender" in role
        ):
            return True
        text = task_text(agent)
        return "write the final" in text or "write report.md" in text

    def is_research(agent):
        if is_synthesis(agent):
            return False
        if is_impl(agent):
            return False
        role = role_text(agent)
        if any(
            word in role
            for word in ("literature", "research", "paper", "hugging face", "ecosystem", "survey", "training", "strategy", "hardware")
        ):
            return True
        text = task_text(agent)
        return any(word in text for word in ("write papers.md", "write huggingface_ecosystem.md", "write training_plan.md"))

    def phase(agent):
        if is_synthesis(agent):
            return 2
        if is_impl(agent):
            return 1
        return 0

    all_ids = [agent["id"] for agent in agents]
    research_ids = [agent["id"] for agent in agents if phase(agent) == 0]
    phase_by_id = {agent["id"]: phase(agent) for agent in agents}

    stabilized = []
    for agent in agents:
        current_phase = phase(agent)
        if current_phase == 2:
            deps = {other_id for other_id in all_ids if other_id != agent["id"] and phase_by_id[other_id] < current_phase}
        elif current_phase == 1:
            deps = {other_id for other_id in research_ids if other_id != agent["id"]}
        elif is_research(agent):
            deps = set()
        else:
            deps = {
                dep
                for dep in agent["depends_on"]
                if dep in all_ids and dep != agent["id"] and phase_by_id.get(dep, 0) <= current_phase
            }
        stabilized.append({**agent, "depends_on": sorted(deps)})
    return stabilized


def build_agent_task(agent_def, all_agents):
    agent_id = agent_def["id"]
    role = agent_def["role"]
    task = agent_def["task"]
    depends_on = agent_def["depends_on"]
    role_lower = role.lower()
    task_lower = task.lower()

    teammate_lines = []
    for agent in all_agents:
        if agent["id"] == agent_id:
            continue
        marker = " (dependency)" if agent["id"] in depends_on else ""
        teammate_lines.append(f"  - {agent['id']}: {agent['role']}{marker}")

    if depends_on:
        wait_block = (
            f"Your dependencies are already completed or will complete before your work: {', '.join(depends_on)}.\n"
            "Read their files first, then continue immediately."
        )
    else:
        wait_block = "You have no dependencies. Start immediately and produce useful artifacts quickly."

    role_specific_rules = []
    if "hugging face" in role_lower or "ecosystem" in role_lower:
        role_specific_rules.append(
            "- Name specific Hugging Face assets when relevant: model repos, libraries, kernels, datasets, docs, or Spaces. Avoid generic ecosystem-only language."
        )
    if "training" in role_lower or "strategy" in role_lower or "hardware" in role_lower:
        role_specific_rules.append(
            "- Convert the user's resource limits into concrete fit guidance: recommended parameter scale, fine-tuning method, memory-saving approach, and operational caveats."
        )
    if (
        re.search(r"\bimplementation\b", role_lower)
        or "code" in role_lower
        or "snippet" in role_lower
        or "practical" in role_lower
    ):
        role_specific_rules.append(
            "- In code_examples.md, include 1-3 small snippets that are ready to embed into report.md."
        )
        role_specific_rules.append(
            "- For each snippet, include a numbered line-by-line explanation that covers every code line in order."
        )
    if "synthesis" in role_lower or "report" in role_lower or "writer" in role_lower or "report.md" in task_lower:
        role_specific_rules.append(
            "- report.md must embed the final code snippets directly in fenced code blocks rather than only linking to external files."
        )
        role_specific_rules.append(
            "- After each embedded snippet, add a numbered line-by-line explanation that explains every code line in order."
        )
        role_specific_rules.append(
            "- Add a section that explicitly names the relevant Hugging Face models, repos, libraries, kernels, datasets, docs, or Spaces for the topic."
        )
    role_specific_block = ""
    if role_specific_rules:
        role_specific_block = "ROLE-SPECIFIC RULES:\n" + "\n".join(role_specific_rules) + "\n\n"

    return "".join(
        [
            f"You are {agent_id}. Your role: {role}\n\n",
            f"TEAMMATES:\n{chr(10).join(teammate_lines) if teammate_lines else '  - none'}\n\n",
            f"DEPENDENCIES:\n{wait_block}\n\n",
            f"YOUR TASK:\n{task}\n\n",
            "SHARED WORKSPACE:\n",
            "- All artifacts go in the working directory using relative paths.\n",
            "- Prefer files such as huggingface_ecosystem.md, training_plan.md, code_examples.md, report.md, snippets/example.py.\n",
            "- Write the first useful version of your artifact early, then refine it.\n",
            "- Do not create a shared/ subdirectory.\n\n",
            role_specific_block,
            "COMMUNICATION:\n",
            "- When you complete your deliverable, send a message to the team with the files you created and the key findings.\n",
            "- Check messages when appropriate, but do not spin-wait.\n",
            "- After announcing completion, call done.",
        ]
    )


def _list_workspace_files(shared_dir):
    lines = []
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            if filename.endswith(".pyc"):
                continue
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            size = os.path.getsize(path)
            lines.append(f"  {rel} ({size} bytes)")
    return "\n".join(lines) if lines else "  (empty)"


def build_qa_task(original_task, agents, shared_dir):
    agent_summaries = "\n".join(f"  - {agent['id']}: {agent['role']}" for agent in agents)
    return QA_AGENT_TASK_TEMPLATE.format(
        original_task=original_task,
        agent_summaries=agent_summaries,
        file_listing=_list_workspace_files(shared_dir),
    )


def read_qa_report(shared_dir):
    report_path = os.path.join(shared_dir, "qa_report.json")
    if not os.path.exists(report_path):
        print("[QA] No qa_report.json found.")
        return {
            "status": "fail",
            "summary": "QA agent did not produce qa_report.json.",
            "errors": [
                {
                    "id": "err_timeout",
                    "severity": "critical",
                    "category": "missing_deliverable",
                    "file": "qa_report.json",
                    "description": "QA report is missing.",
                    "evidence": "qa_report.json was not found after the QA wave.",
                    "suggested_fix": "Ensure the QA agent writes qa_report.json before calling done.",
                }
            ],
            "files_tested": [],
            "commands_run": [],
        }
    with open(report_path) as handle:
        return json.load(handle)


def build_fix_defs(assignments, agents, report):
    agent_by_id = {agent["id"]: agent for agent in agents}
    error_by_id = {err["id"]: err for err in report["errors"]}
    fix_defs = []
    for assignment in assignments["assignments"]:
        agent = agent_by_id[assignment["agent_id"]]
        error_details = []
        for error_id in assignment["error_ids"]:
            err = error_by_id[error_id]
            error_details.append(
                f"[{error_id}] ({err['severity']}) {err['file']}: {err['description']}\n"
                f"  Evidence: {err['evidence']}\n"
                f"  Suggested fix: {err.get('suggested_fix', 'N/A')}"
            )
        fix_defs.append(
            {
                "id": assignment["agent_id"],
                "task": FIX_AGENT_TASK_TEMPLATE.format(
                    agent_id=assignment["agent_id"],
                    original_role=agent["role"],
                    error_details="\n\n".join(error_details),
                    fix_task=assignment["fix_task"],
                ),
            }
        )
    return fix_defs


def run_agent(agent_def, log_prefix, agent_model, max_iterations, max_runtime, shared_dir):
    env = {
        **os.environ,
        "PROTOCOL_ENABLED": "true",
        "BROKER_MODE": "connect",
        "AGENT_ID": agent_def["id"],
        "AGENT_MODEL": agent_model,
        "MAX_ITERATIONS": str(max_iterations),
        "MAX_RUNTIME_SECONDS": str(max_runtime),
        "SHARED_WORKSPACE": shared_dir,
    }
    proc = subprocess.Popen(
        [sys.executable, "main.py", agent_def["task"]],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        print(f"[{log_prefix}] {text}", flush=True)
    proc.wait()
    return proc.returncode


def run_wave(agent_defs, wave_name, agent_model, max_iterations, max_runtime, shared_dir):
    print(f"\n{'=' * 70}")
    print(f"WAVE: {wave_name} ({len(agent_defs)} agent(s))")
    print(f"{'=' * 70}\n")

    threads = []
    results = {}

    for agent_def in agent_defs:
        name = f"{wave_name}:{agent_def['id'].upper()}"

        def target(a=agent_def, n=name):
            results[n] = run_agent(a, n, agent_model, max_iterations, max_runtime, shared_dir)

        thread = threading.Thread(target=target)
        thread.start()
        threads.append(thread)
        time.sleep(1)

    for thread in threads:
        thread.join()

    for name, code in results.items():
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  [{name}] {status}")

    return results


def _print_workspace(shared_dir):
    print("\nShared workspace contents:")
    for root, dirs, files in os.walk(shared_dir):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
        for filename in sorted(files):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, shared_dir)
            print(f"  {rel} ({os.path.getsize(path)} bytes)")


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent Hugging Face model advisor orchestrator")
    parser.add_argument("task", nargs="?", help="Task description string")
    parser.add_argument("--prompts", help="Path to a JSON file containing prompt definitions")
    parser.add_argument("--prompt", type=int, help="1-indexed prompt number from the prompts file")
    parser.add_argument("--shared-dir", help="Use an existing shared workspace instead of creating a new one")
    args = parser.parse_args()

    if args.prompts:
        with open(args.prompts) as handle:
            prompts = json.load(handle)
        idx = (args.prompt or 1) - 1
        entry = prompts[idx]
        print(f"Loaded prompt #{idx + 1}: {entry['name']}")
        task = entry["prompt"]
    else:
        task = args.task

    if not task:
        parser.error("Provide a task string or use --prompts/--prompt.")

    return task, args.shared_dir


def main():
    task, shared_dir_arg = parse_args()

    manifest_path = os.path.join(os.path.dirname(__file__), "manifest.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    settings_pack_name = os.environ.get("SETTINGS_PACK", manifest["defaultSettingsPack"])
    settings = manifest["settingsPacks"][settings_pack_name]
    provider = os.environ.get("PROVIDER", settings["provider"])
    orchestrator_model = os.environ.get("ORCHESTRATOR_MODEL", settings["model"])
    agent_model = os.environ.get("AGENT_MODEL", settings["model"])
    max_iterations = int(os.environ.get("MAX_ITERATIONS", "40"))
    max_runtime = int(os.environ.get("MAX_RUNTIME_SECONDS", "300"))
    max_waves = int(os.environ.get("MAX_WAVES", "2"))
    qa_iterations = int(os.environ.get("QA_ITERATIONS", "30"))
    fix_iterations = int(os.environ.get("FIX_ITERATIONS", "15"))
    fix_runtime = int(os.environ.get("FIX_RUNTIME_SECONDS", "120"))
    report_bucket = os.environ.get("REPORT_BUCKET")
    report_prefix = os.environ.get("REPORT_PREFIX")

    print("=" * 70)
    print("HUGGING FACE MODEL ADVISOR ORCHESTRATOR")
    print("=" * 70)
    print(f"Task: {task}")
    print(f"Orchestrator: {provider}/{orchestrator_model}")
    print(f"Agent model: {agent_model}")
    print(f"Max iterations: {max_iterations} | Max runtime: {max_runtime}s")
    print(f"QA waves: {max_waves} | QA iters: {qa_iterations} | Fix iters: {fix_iterations}")
    print()

    print("[ORCHESTRATOR] Decomposing task...\n")
    if provider == "openai":
        result = call_orchestrator_openai(orchestrator_model, task)
    else:
        result = call_orchestrator_anthropic(orchestrator_model, task)

    agents = stabilize_agent_graph(result["agents"])
    print(f"[ORCHESTRATOR] Created {len(agents)} agents:")
    for agent in agents:
        deps = f" (depends on: {', '.join(agent['depends_on'])})" if agent["depends_on"] else ""
        print(f"  - {agent['id']}: {agent['role']}{deps}")
    print()

    agent_defs = [{"id": agent["id"], "task": build_agent_task(agent, agents)} for agent in agents]

    if shared_dir_arg:
        shared_dir = shared_dir_arg
        os.makedirs(shared_dir, exist_ok=True)
    else:
        shared_dir = os.path.join(os.path.dirname(__file__), "runs", f"shared-{int(time.time() * 1000)}")
        os.makedirs(shared_dir)
    print(f"[SHARED] Workspace: {shared_dir}")

    broker = MessageBroker(enable_logging=False)
    broker.start()
    time.sleep(0.5)
    print("[BROKER] Started on :5555/:5556")

    try:
        for wave_idx, wave_agents in enumerate(topological_waves(agents), start=1):
            wave_ids = {agent["id"] for agent in wave_agents}
            wave_defs = [agent_def for agent_def in agent_defs if agent_def["id"] in wave_ids]
            run_wave(wave_defs, f"BUILD-{wave_idx}", agent_model, max_iterations, max_runtime, shared_dir)

        qa_passed = max_waves == 0
        if max_waves > 0:
            for wave_num in range(1, max_waves + 1):
                report_path = os.path.join(shared_dir, "qa_report.json")
                if os.path.exists(report_path):
                    os.remove(report_path)

                qa_task = build_qa_task(task, agents, shared_dir)
                qa_defs = [{"id": "qa", "task": qa_task}]
                run_wave(qa_defs, f"QA-{wave_num}", agent_model, qa_iterations, fix_runtime, shared_dir)

                report = read_qa_report(shared_dir)
                print(f"\n[QA-{wave_num}] Status: {report['status']}")
                print(f"[QA-{wave_num}] Summary: {report['summary']}")

                if report["status"] == "pass":
                    qa_passed = True
                    break

                for err in report.get("errors", []):
                    print(f"  [{err['id']}] {err['severity']} - {err['file']}: {err['description']}")

                if wave_num == max_waves:
                    print(f"\n[ORCHESTRATOR] QA failed after {max_waves} wave(s).")
                    break

                print(f"\n[ORCHESTRATOR] Assigning {len(report['errors'])} error(s) to agents...")
                if provider == "openai":
                    assignments = call_assign_fixes_openai(orchestrator_model, report, agents)
                else:
                    assignments = call_assign_fixes_anthropic(orchestrator_model, report, agents)

                for assignment in assignments["assignments"]:
                    print(f"  - {assignment['agent_id']}: {', '.join(assignment['error_ids'])}")

                fix_defs = build_fix_defs(assignments, agents, report)
                run_wave(fix_defs, f"FIX-{wave_num}", agent_model, fix_iterations, fix_runtime, shared_dir)

        summary = {
            "status": "pass" if qa_passed else "fail",
            "task": task,
            "workspace": shared_dir,
            "agents": [{"id": agent["id"], "role": agent["role"]} for agent in agents],
            "qa_waves_run": max_waves,
            "qa_passed": qa_passed,
            "files": [],
        }
        for root, dirs, files in os.walk(shared_dir):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules", ".venv", "venv")]
            for filename in sorted(files):
                if filename.endswith(".pyc"):
                    continue
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, shared_dir)
                summary["files"].append({"path": rel, "size_bytes": os.path.getsize(path)})

        report_path = os.path.join(shared_dir, "qa_report.json")
        if os.path.exists(report_path):
            with open(report_path) as handle:
                summary["qa_report"] = json.load(handle)
        else:
            summary["qa_report"] = None

        artifacts_info = None
        if report_bucket:
            artifacts_info = build_artifacts_info(shared_dir, report_bucket, report_prefix)
            artifacts_info["upload_status"] = "pending"
            summary["artifacts"] = artifacts_info

        summary_path = os.path.join(shared_dir, "run_summary.json")
        with open(summary_path, "w") as handle:
            json.dump(summary, handle, indent=2)

        upload_error = None
        if report_bucket:
            write_artifacts_manifest(shared_dir, artifacts_info, status="pending")
            try:
                artifacts_info = upload_workspace_to_bucket(shared_dir, report_bucket, report_prefix)
                artifacts_info["upload_status"] = "uploaded"
                summary["artifacts"] = artifacts_info
                with open(summary_path, "w") as handle:
                    json.dump(summary, handle, indent=2)
                write_artifacts_manifest(shared_dir, artifacts_info, status="uploaded")
                upload_workspace_to_bucket(shared_dir, report_bucket, report_prefix)
                print("\n[ARTIFACTS] Uploaded run outputs")
                print(f"[ARTIFACTS] Primary: {artifacts_info['primary_artifact']['remote_uri']}")
                print(f"[ARTIFACTS] Bucket: {artifacts_info['bucket_url']}")
            except Exception as exc:
                upload_error = str(exc)
                artifacts_info["upload_status"] = "failed"
                artifacts_info["upload_error"] = upload_error
                summary["artifacts"] = artifacts_info
                with open(summary_path, "w") as handle:
                    json.dump(summary, handle, indent=2)
                write_artifacts_manifest(shared_dir, artifacts_info, status="failed", upload_error=upload_error)
                print(f"\n[ARTIFACTS] Upload failed: {upload_error}")

        print("\n" + "=" * 70)
        print("ALL WAVES COMPLETE")
        print("=" * 70)
        print(f"QA result: {'PASSED' if qa_passed else 'FAILED'}")
        print(f"Run summary written to: {summary_path}")
        _print_workspace(shared_dir)
        sys.exit(0 if qa_passed and not upload_error else 1)
    finally:
        broker.stop()


if __name__ == "__main__":
    main()
