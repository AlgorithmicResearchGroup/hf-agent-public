from agent.tool_registry import get_all_tools


def test_base_toolset_matches_ml_research_cutover():
    names = {tool["name"] for tool in get_all_tools()}

    assert "search_web" in names
    assert "fetch_page" in names
    assert "search_huggingface" in names
    assert "inspect_huggingface_repo" in names
    assert "read_huggingface_file" in names

    assert "census_api" not in names
    assert "internal_db" not in names

