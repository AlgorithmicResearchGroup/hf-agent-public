from agent.prompts import get_worker_system_prompt


def test_worker_prompt_is_hf_model_advisor_focused():
    prompt = get_worker_system_prompt("/tmp/work")

    assert "Hugging Face model-selection and fine-tuning advisor" in prompt
    assert "Recommending which model to use for a specific hardware budget" in prompt
    assert "search_web" in prompt
    assert "search_huggingface" in prompt
    assert "inspect_huggingface_repo" in prompt
    assert "read_huggingface_file" in prompt
    assert "census_api" not in prompt
    assert "internal_db" not in prompt


def test_worker_prompt_requires_embedded_annotated_snippets_and_named_hf_assets():
    prompt = get_worker_system_prompt("/tmp/work")

    assert "include the snippet directly in Markdown" in prompt
    assert "line-by-line explanation" in prompt
    assert "include the code snippets inside `report.md` itself" in prompt
    assert "explicitly name the relevant Hugging Face models, repos, libraries, docs, datasets, or Spaces" in prompt
