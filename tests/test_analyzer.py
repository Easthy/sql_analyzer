import json
import pytest
from pathlib import Path
from sql_analyzer import SQLAnalyzer

TEST_DATA_DIR = Path(__file__).parent / "test_data"

def clean_and_sort_graph_data(graph_data: dict) -> dict:
    """Simplify and sort graph data for stable comparison by keeping only key fields."""
    nodes = sorted([{"id": n["id"]} for n in graph_data.get("nodes", [])], key=lambda x: x["id"])
    links = sorted([
        {"source": l["source"], "target": l["target"], "type": l.get("type")} 
        for l in graph_data.get("links", [])
    ], key=lambda x: (x["source"], x["target"]))
    return {"nodes": nodes, "links": links}

def run_analyzer_test(test_env, sql_filename, golden_filename):
    """Common test logic for running analyzer and comparing with golden file."""
    # Setup test files
    sql_dir = test_env["sql_dir"]
    state_file = test_env["state_file"]
    
    sql_content = (TEST_DATA_DIR / "sql_models" / sql_filename).read_text()
    (sql_dir / sql_filename).write_text(sql_content)

    # Load and prepare expected result
    with open(TEST_DATA_DIR / "golden_files" / golden_filename, 'r') as f:
        expected_graph = clean_and_sort_graph_data(json.load(f))
    
    # Execute analyzer
    analyzer = SQLAnalyzer(config=test_env["config_manager"])
    analyzer.run()

    # Verify results
    assert state_file.exists(), "State file was not created"
    
    with open(state_file, 'r') as f:
        actual_graph = clean_and_sort_graph_data(json.load(f))
    
    assert actual_graph["nodes"] == expected_graph["nodes"], "Node lists do not match"
    assert actual_graph["links"] == expected_graph["links"], "Link lists do not match"
    assert actual_graph == expected_graph, "Final graph does not match the golden file"

# Test cases configuration
TEST_CASES = [
    ("simple.create_table_as_select_from_table.sql", 
     "simple.create_table_as_select_from_table.json"),

    ("simple.create_table.sql", 
     "simple.create_table.json"),

    ("simple.create_table_as_select_without_from.sql",
     "simple.create_table_as_select_without_from.json"),

    ("simple.create_table_insert_with_columns_diff_name_with_alias.sql",
     "simple.create_table_insert_with_columns_diff_name_with_alias.json"),

    ("simple.create_table_insert_with_columns_diff_name_without_alias.sql",
     "simple.create_table_insert_with_columns_diff_name_without_alias.json"),

    ("simple.create_table_insert_with_columns_same_name.sql",
    "simple.create_table_insert_with_columns_same_name.json"),

    ("simple.create_table_insert_without_columns_diff_name_with_alias.sql",
     "simple.create_table_insert_without_columns_diff_name_with_alias.json"),

    ("simple.create_table_insert_without_columns_diff_name_without_alias.sql",
     "simple.create_table_insert_without_columns_diff_name_without_alias.json")
]

# Generate individual test functions
@pytest.mark.parametrize("sql_file,expected_file", TEST_CASES)
def test_sql_analyzer(test_env, sql_file, expected_file):
    """Parametrized test for SQL analyzer."""
    run_analyzer_test(test_env, sql_file, expected_file)