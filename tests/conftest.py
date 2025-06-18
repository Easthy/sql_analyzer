import pytest
import sys
from pathlib import Path
from types import ModuleType

# Ensure pytest can see the main code
# This is needed if you run pytest from the root project folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from sql_analyzer import ConfigManager

@pytest.fixture
def test_env(tmp_path, monkeypatch):
    """
    Creates a temporary test environment for each test.
    - tmp_path: built-in pytest fixture that provides a temporary directory.
    - monkeypatch: built-in fixture for safely modifying variables/attributes.
    """
    # 1. Create temporary folders for SQL models and the state file
    sql_models_dir = tmp_path / "sql_models"
    sql_models_dir.mkdir()

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    state_file = tmp_path / "output" / "dependency_state.json"

    # 3. Return a configured ConfigManager for use in the test
    # ConfigManager will validate and pick up our patched settings
    class ConfigManager:
        sql_models_dir = tmp_path / "sql_models"
        state_file = tmp_path / "output" / "dependency_state.json"
        sql_dialect = "redshift"
        sql_file_extension = ".sql"
        normalize_names = True
        source_models_file = tmp_path / "sources.yml"
    
    monkeypatch.setattr('sql_analyzer.ConfigManager', ConfigManager)
    config_manager = ConfigManager()
    
    # Provide the test with access to these paths
    yield {
        "tmp_path": tmp_path,
        "config_manager": config_manager,
        "sql_dir": sql_models_dir,
        "state_file": state_file
    }
