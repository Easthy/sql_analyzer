# config.py
import os
from pathlib import Path

# Project root (assuming config.py is located at or near the root)
# BASE_DIR = Path(__file__).resolve().parent
# If the script will be run from a different location, it's better to set an absolute path
# or a path relative to the execution location. Let's make it configurable.
# Specify the ABSOLUTE path to the folder containing your SQL models
# Example: SQL_MODELS_DIR = Path("/path/to/your/sql_models")
# Or a relative path from where the script is executed:
SQL_MODELS_DIR = Path("./sql_models")  # SQL script of you data models

# List of source tables (that are used ti build models' tables)
SQL_SOURCE_MODELS = Path("sources.yml")  # YAML file, describing tables and their columns

# File to save the dependency graph state
STATE_FILE = Path("./dependency_state.json")

# SQL dialect (important for correct parsing)
# Examples: "postgres", "mysql", "snowflake", "bigquery", "clickhouse"
SQL_DIALECT = "redshift"

# File extension for SQL model files
SQL_FILE_EXTENSION = ".sql"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(levelname)s - %(message)s'
