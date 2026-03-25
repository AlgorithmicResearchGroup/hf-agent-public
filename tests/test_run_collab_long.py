from run_collab_long import QA_AGENT_TASK_TEMPLATE, build_agent_task, stabilize_agent_graph


def test_stabilize_agent_graph_makes_impl_wait_for_research_and_synth_wait_for_all():
    agents = [
        {"id": "lit", "role": "Literature scout", "task": "Write papers.md", "depends_on": []},
        {"id": "hf", "role": "Hugging Face ecosystem scan", "task": "Write huggingface_ecosystem.md", "depends_on": []},
        {"id": "impl", "role": "Implementation examples", "task": "Write code_examples.md", "depends_on": []},
        {"id": "synth", "role": "Synthesis report writer", "task": "Write report.md", "depends_on": []},
    ]

    stabilized = {agent["id"]: agent for agent in stabilize_agent_graph(agents)}

    assert stabilized["impl"]["depends_on"] == ["hf", "lit"]
    assert stabilized["synth"]["depends_on"] == ["hf", "impl", "lit"]


def test_build_agent_task_requires_embedded_annotated_report_snippets_for_synthesis():
    agent = {
        "id": "synth",
        "role": "Synthesis report writer",
        "task": "Read team artifacts and write report.md",
        "depends_on": ["hf", "impl", "lit"],
    }
    task_text = build_agent_task(agent, [agent])

    assert "report.md must embed the final code snippets directly" in task_text
    assert "numbered line-by-line explanation" in task_text
    assert "relevant Hugging Face models, repos, libraries, kernels, datasets, docs, or Spaces" in task_text


def test_qa_template_checks_embedded_snippets_and_explicit_hf_coverage():
    assert "concrete recommended model or clearly ranked recommendation set" in QA_AGENT_TASK_TEMPLATE
    assert "hardware / fine-tuning fit analysis" in QA_AGENT_TASK_TEMPLATE
    assert "explicit Hugging Face models / techniques discussion" in QA_AGENT_TASK_TEMPLATE
    assert "code snippet embedded in `report.md`" in QA_AGENT_TASK_TEMPLATE
    assert "line-by-line explanation for each embedded code snippet" in QA_AGENT_TASK_TEMPLATE
    assert "generic-only Hugging Face coverage" in QA_AGENT_TASK_TEMPLATE
