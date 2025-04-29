# -*- coding: utf-8 -*-
import os
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any, Union, Type, Mapping

import networkx as nx
import sqlglot
from sqlglot import Expression
from sqlglot.errors import ParseError
from sqlglot.lineage import lineage as sqlglot_lineage
from sqlglot.lineage import Node as LineageNode
from sqlglot.dialects import Dialect
from sqlglot.optimizer.scope import Scope
from sqlglot.schema import Schema
from networkx.readwrite import json_graph
from rich.logging import RichHandler
from rich.text import Text
import warnings
import config
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress pandas future warning from networkx/json

# --- Checking for the presence of required attributes in config ---
required_configs = ['SQL_MODELS_DIR', 'STATE_FILE', 'SQL_DIALECT', 'SQL_FILE_EXTENSION']
missing_configs = [cfg for cfg in required_configs if not hasattr(config, cfg)]
if missing_configs:
    print(f"Ошибка: В config.py отсутствуют следующие обязательные параметры: {', '.join(missing_configs)}")
    exit(1)

# --- Logging and Constants ---
# Ensure directory for state file exists
config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(config, 'LOG_LEVEL', 'INFO'),
    format=getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    encoding='utf-8',
    handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)] # Simplified handler
)
logger = logging.getLogger('sql_analyzer') # Use a specific logger name

TBL_PREFIX = "tbl"
COL_PREFIX = "col"
# Default to True if NORMALIZE_NAMES not in config
NORMALIZE_NAMES = getattr(config, 'NORMALIZE_NAMES', True)

# --- Utilities for Names and IDs ---
def normalize_name(name: Optional[str]) -> Optional[str]:
    """Converts the name to lowercase if normalization is enabled"""
    if NORMALIZE_NAMES and name:
        return name.lower()
    return name

def format_node_id(node_type: str, schema: Optional[str], name: Optional[str], column: Optional[str] = None) -> str:
    """Generates a unique ID for a graph node with optional normalization"""
    schema_norm = normalize_name(schema if schema else 'unknown_schema')
    name_norm = normalize_name(name if name else 'unknown_table')

    if node_type == TBL_PREFIX:
        if not schema_norm or not name_norm:
             raise ValueError(f"Schema and name are required for table node: schema='{schema}', name='{name}'")
        return f"{TBL_PREFIX}:{schema_norm}.{name_norm}"
    elif node_type == COL_PREFIX:
        if not schema_norm or not name_norm or not column:
            raise ValueError(f"Schema, table name, and column name are required for column node: schema='{schema}', name='{name}', column='{column}'")
        col_norm = normalize_name(column)
        return f"{COL_PREFIX}:{schema_norm}.{name_norm}.{col_norm}"
    else:
        raise ValueError(f"Unknown node type: {node_type}")

def get_name_from_node_id(node_id: str) -> Dict:
    """
    Extracts the table or column name from a node ID.

    If the node represents a table (TBL_PREFIX), returns the table name.
    If the node represents a column (COL_PREFIX), returns the column name.
    """
    if node_id.startswith(f"{TBL_PREFIX}:"):
        # Format: tbl:schema.table
        try:
            _, schema_table = node_id.split(":", 1)
            schema, table = schema_table.split(".", 1)
            return {"schema": schema, "table": table}
        except ValueError:
            raise ValueError(f"Invalid table node_id format: {node_id}")
    elif node_id.startswith(f"{COL_PREFIX}:"):
        # Format: col:schema.table.column
        try:
            _, schema_table_col = node_id.split(":", 1)
            schema, table, column = schema_table_col.split(".", 2)
            return {"schema": schema, "table": table, "column": column}
        except ValueError:
            raise ValueError(f"Invalid column node_id format: {node_id}")
    else:
        raise ValueError(f"Unknown node_id prefix: {node_id}")

def parse_node_id(node_id: str) -> Dict[str, Optional[str]]:
    """Parses the node ID into components"""
    if not isinstance(node_id, str):
        raise ValueError(f"Cannot parse node ID: Expected string, got {type(node_id)} ({node_id})")

    parts = node_id.split(":", 1)
    if len(parts) != 2:
         raise ValueError(f"Cannot parse node ID: {node_id}. Invalid format (missing ':').")
    node_type = parts[0]
    full_name = parts[1]

    name_parts = full_name.split('.')
    if node_type == TBL_PREFIX:
        if len(name_parts) < 2:
             raise ValueError(f"Cannot parse table node ID: {node_id}. Expected format 'tbl:schema.table'")
        schema = name_parts[0]
        table = '.'.join(name_parts[1:])
        return {"type": node_type, "schema": schema, "table": table, "column": None}
    elif node_type == COL_PREFIX:
        if len(name_parts) < 3:
             raise ValueError(f"Cannot parse column node ID: {node_id}. Expected format 'col:schema.table.column'")
        schema = name_parts[0]
        table = name_parts[1] # Assume second part is always the table for columns
        column = '.'.join(name_parts[2:])
        if not schema or not table or not column:
             raise ValueError(f"Cannot parse column node ID: {node_id}. Invalid schema, table, or column part.")
        return {"type": node_type, "schema": schema, "table": table, "column": column}
    else:
         raise ValueError(f"Cannot parse node ID: Unknown type '{node_type}' in {node_id}")

# --- File and SQL handling functions ---
def find_sql_files(directory: Path) -> List[Path]:
    """Рекурсивно находит все файлы .sql в указанной директории."""
    logger.info(f"Searching for SQL files in: {directory}")
    sql_files = list(directory.rglob(f"*{getattr(config, 'SQL_FILE_EXTENSION', '.sql')}"))
    logger.info(f"Found {len(sql_files)} SQL files.")
    if not sql_files:
        logger.warning(f"В директории {directory} не найдено файлов с расширением {getattr(config, 'SQL_FILE_EXTENSION', '.sql')}")
    return sql_files

def extract_model_name_from_path(file_path: Path, root_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the model name (schema.table) from the file path.
    Uses the file name in the format 'schema.table_name.sql
    """
    filename_stem = file_path.stem
    name_parts = filename_stem.split('.')

    if len(name_parts) < 2:
        logger.warning(
            f"Не удалось определить схему и таблицу из имени файла: '{file_path.name}'. "
            f"Ожидаемый формат: 'schema.table.sql'. Файл будет пропущен."
        )
        return None, None

    schema = name_parts[0]
    table_name = '.'.join(name_parts[1:])

    logger.debug(f"Файл: {file_path.name}, Схема: {schema}, Таблица: {table_name}")
    return schema, table_name

def parse_source_models(source_file_list: Path) -> List:
    """
    Parses source models' definition from yaml file and return dict,
    containing columns as well
    """
    def load_source_tables(source_file_list: Path) -> None:
        """Load source table schemas from YAML file."""
        try:
            with open(source_file_list, 'r') as file:
                source_tables = yaml.safe_load(file)
            logger.info(f"Loaded source tables: {list(source_tables.keys())}")
            return source_tables
        except Exception as e:
            logger.error(f"Failed to load sources file: {e}")
            raise

    logger.info("--- Phase 1: Source models parsing ---")
    source_models = []
    parsed_definitions = load_source_tables(source_file_list)

    for table, columns in parsed_definitions.items():
        schema, table_name = table.split('.')
        model_id = format_node_id(TBL_PREFIX, schema, table_name)
        source_models.append({
            "schema": schema,
            "name": table_name,
            "source_type": 'source_table',
            "columns": columns,
            "model_id": format_node_id(TBL_PREFIX, schema, table_name)
        })
    return source_models


def find_main_statement(sql_content: str, target_schema: str, target_table: str) -> Optional[Expression]:
    """
    Parses the SQL and finds the 'main' statement (INSERT or CREATE) that defines the target table
    """
    expressions: List[Expression] = []
    try:
        expressions = sqlglot.parse(sql_content, read=getattr(config, 'SQL_DIALECT', None))
    except ParseError as e:
        logger.error(f"SQL parsing error for {target_schema}.{target_table}: {e}")
        # Log detailed parsing errors
        if hasattr(e, 'errors') and e.errors:
            for error_info in e.errors:
                start = error_info.get('start', 0)
                end = error_info.get('end', len(sql_content))
                line = error_info.get('line', '?')
                col = error_info.get('col', '?')
                context_start = max(0, start - 50)
                context_end = min(len(sql_content), end + 50)
                problem_sql = sql_content[context_start:context_end]
                logger.error(f"  Error: {error_info.get('description')} (row ~{line}, column ~{col})")
                logger.error(f"  Context: ...{problem_sql}...")
        else:
             logger.error("  Parsing error details are unavailable")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while parsing SQL for {target_schema}.{target_table}: {e}")
        logger.debug(f"SQL context (the beginning):\n{sql_content[:500]}...")
        return None

    if not expressions:
        logger.warning(f"No SQL statements found in file for {target_schema}.{target_table}")
        return None

    main_statement = None
    target_schema_norm = normalize_name(target_schema)
    target_table_norm = normalize_name(target_table)

    # Iterate backwards to find the last relevant statement
    for expr in reversed(expressions):
        table_expr = None
        schema_name = None
        table_name = None

        if isinstance(expr, sqlglot.exp.Insert):
            # Extracts the target table/schema from various INSERT forms
            insert_target = expr.this
            if isinstance(insert_target, sqlglot.exp.Table):
                table_expr = insert_target
                table_name = table_expr.name
                schema_name = table_expr.db
            elif isinstance(insert_target, sqlglot.exp.Dot) and isinstance(insert_target.this, sqlglot.exp.Identifier) and isinstance(insert_target.expression, sqlglot.exp.Identifier):
                 schema_name = insert_target.expression.name
                 table_name = insert_target.this.name
            elif isinstance(insert_target, sqlglot.exp.Schema): # Handles INSERT INTO schema.table (cols...) or INSERT INTO db.schema.table (cols...)
                 innermost_table = insert_target.find(sqlglot.exp.Table)
                 if innermost_table:
                     table_name = innermost_table.name
                     schema_name = innermost_table.db # Schema specified directly on table?
                 if not schema_name and isinstance(insert_target.this, sqlglot.exp.Identifier): # Schema name itself?
                     schema_name = insert_target.this.name


            if table_name:
                table_name_norm = normalize_name(table_name)
                schema_name_norm = normalize_name(schema_name if schema_name else target_schema)

                if table_name_norm == target_table_norm and schema_name_norm == target_schema_norm:
                    main_statement = expr
                    logger.debug(f"Found INSERT statement for {target_schema}.{target_table}")
                    break
                else:
                    logger.debug(f"Found INSERT in the table '{schema_name}.{table_name}', but does not match the target '{target_schema}.{target_table}'.")

        elif isinstance(expr, (sqlglot.exp.Create)):
            target_create_expr = expr.this
            kind = expr.kind or "OBJECT"

            if isinstance(target_create_expr, sqlglot.exp.Table):
                table_name = target_create_expr.name
                schema_name = target_create_expr.db
            elif isinstance(target_create_expr, sqlglot.exp.Dot) and isinstance(target_create_expr.this, sqlglot.exp.Identifier) and isinstance(target_create_expr.expression, sqlglot.exp.Identifier):
                 schema_name = target_create_expr.expression.name
                 table_name = target_create_expr.this.name
            elif isinstance(target_create_expr, sqlglot.exp.Identifier):
                 table_name = target_create_expr.this
                 schema_name = None

            if table_name:
                table_name_norm = normalize_name(table_name)
                schema_name_norm = normalize_name(schema_name if schema_name else target_schema)

                if table_name_norm == target_table_norm and schema_name_norm == target_schema_norm:
                    main_statement = expr
                    logger.debug(f"Found CREATE {kind} statement for {target_schema}.{target_table}")
                    break
                else:
                    logger.debug(f"Found CREATE {kind} for '{schema_name}.{table_name}', but does not match the target '{target_schema}.{target_table}'.")

    if not main_statement:
         logger.warning(f"No main INSERT/CREATE TABLE/VIEW statement found for... {target_schema}.{target_table} в файле.")
         # Optionally, try to find the *last* SELECT statement as a fallback? Risky.
         # last_select = next((e for e in reversed(expressions) if isinstance(e, exp.Select)), None)
         # if last_select:
         #    logger.warning("Using the last SELECT as the main statement")
         #    main_statement = last_select

    return main_statement

def find_source_columns_from_lineage(node: LineageNode, dialect: str, target_col_id: str) -> List[sqlglot.exp.Column]:
    """
    Recursively traverses the lineage Node tree and collects all
    sqlglot.exp.Column expressions that represent terminal sources
    (i.e., leaf Column nodes in the lineage tree with table information).
    """
    sources: List[sqlglot.exp.Column] = []
    processed_node_ids = set()

    target_col_name = get_name_from_node_id(target_col_id).get('column')

    _cols = []
    def traverse(current_node: LineageNode):
        node_id = id(current_node)
        if node_id in processed_node_ids:
            return
        processed_node_ids.add(node_id)

        # Compare normalized names to avoid accidentally adding the target itself as a source
        # (even though lineage usually doesn't return it as a leaf)
        is_target_node = normalize_name(current_node.name) == normalize_name(target_col_name)

        # Check if the current node is a leaf in the lineage tree
        is_leaf_node = not current_node.downstream

        if is_leaf_node:
            table_schema = str(current_node.source).split('.')[0]
            table_name = current_node.name.split('.')[0]
            column = current_node.name.split('.')[1]

            # If the leaf node is itself a column (and not the target)
            source_col_id = format_node_id(
                node_type=COL_PREFIX,
                schema=table_schema,
                name=table_name,
                column=column
            )
            logger.debug(f"Found {source_col_id} column as source for the {target_col_id}")
            _cols.append(source_col_id)

        # Recursively traverse child nodes (downstream dependencies)
        # Sources can only appear at the leaf nodes of the lineage tree
        for child_node in current_node.downstream:
            traverse(child_node)

    # Start traversal from the root node returned by lineage()
    if node:
        traverse(node)

    # Log the discovered sources
    if _cols:
        logger.debug(f"Found source columns for the target '{target_col_id}': {_cols}")

    return _cols

def parse_sql(model_id, root_dir: Path, file_path, target_schema, target_table) -> Dict:
    model = None
    relative_path = str(file_path.relative_to(root_dir)) if root_dir in file_path.parents else str(file_path)
    try:
        sql_content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return model

    # Find the main statement (INSERT/CREATE)
    main_statement = find_main_statement(sql_content, target_schema, target_table)
    if not main_statement:
        logger.warning(f"Skipping {model_id}: main INSERT/CREATE statement not found.")
        return model

    target_columns: List[str] = []
    select_expressions: List[Expression] = []

    try:
        # The code for extracting target_columns and select_expressions remains unchanged
        select_part = None
        if isinstance(main_statement, sqlglot.exp.Insert):
            if isinstance(main_statement.expression, (sqlglot.exp.Select, sqlglot.exp.Union)):
                select_part = main_statement.expression.find(sqlglot.exp.Select)
            target_col_names_explicit = []
            target_spec = main_statement.this
            # Handle cases like INSERT INTO schema.table (col1, col2)
            columns_in_target = None
            if isinstance(target_spec, sqlglot.exp.Schema) and target_spec.expressions:
                columns_in_target = target_spec.expressions
            # Handle cases like INSERT INTO table (col1, col2)
            elif isinstance(target_spec, sqlglot.exp.Tuple): # Often used for column lists
                columns_in_target = target_spec.expressions

            if columns_in_target:
                target_col_names_explicit = [col.name for col in columns_in_target if isinstance(col, sqlglot.exp.Identifier)]

            if target_col_names_explicit:
                target_columns = target_col_names_explicit
                if select_part:
                    select_expressions = select_part.expressions
            elif select_part:
                target_columns = [col.alias_or_name for col in select_part.expressions]
                select_expressions = select_part.expressions

        elif isinstance(main_statement, sqlglot.exp.Create):
            query_expression = main_statement.expression
            if isinstance(query_expression, (sqlglot.exp.Select, sqlglot.exp.Union)):
                 select_part = query_expression.find(sqlglot.exp.Select)

            if select_part:
                target_columns = [col.alias_or_name for col in select_part.expressions]
                select_expressions = select_part.expressions

        original_len = len(target_columns)
        target_columns = [col for col in target_columns if isinstance(col, str)]
        if len(target_columns) != original_len:
            logger.warning(f"Some target columns are neither strings nor None and have been skipped in {model_id}")

    except Exception as e:
        logger.error(f"Error extracting target columns/expressions for {model_id} from {file_path.name}: {e}", exc_info=True)
        target_columns = []
        select_expressions = []

    return {
        "schema": target_schema,
        "name": target_table,
        "source_type": 'model',
        "file_path": relative_path,
        "columns": target_columns,
        "main_statement": main_statement,
        "model_id": model_id
    }

def parse_sql_models(sql_files: List[Path], root_dir: Path) -> List:
    processed_files = 0
    total_files = len(sql_files)

    known_models: List[Dict] = [] # List of models defined by files
    model_definitions: Dict[str, Dict[str, Any]] = {} # model_id -> {file_path, schema, table_name}
    # 1. Model Identification by Files
    logger.info("--- Phase 2: SQL models parsing ---")
    for file_path in sql_files:
        processed_files += 1
        schema, table_name = extract_model_name_from_path(file_path, root_dir)
        logger.info(f"[{processed_files}/{total_files}] Analyzing: {file_path} (Model: {schema}.{table_name})")
        if schema and table_name:
            try:
                model_id = format_node_id(TBL_PREFIX, schema, table_name)
                if model_id in model_definitions:
                    logger.warning(
                        f"Duplicate definition for model {model_id} found in {file_path.name} and {model_definitions[model_id]['file_path'].name}. "
                        f"Using the latest one: {file_path.name}"
                    )
                # TODO: Model definitions is used only to verify duplicates. It is better to rewrite
                model_definitions[model_id] = {"file_path": file_path, "orig_schema": schema, "orig_table_name": table_name}
                sql_model = parse_sql(model_id, root_dir, file_path, schema, table_name)
                known_models.append(sql_model)

            except ValueError as e:
                logger.error(f"ID formatting error for file {file_path.name} ({schema}.{table_name}): {e}")
        else:
            logger.warning(f"Skipping file (failed to determine model): {file_path.name}")

    logger.info(f"Detected {len(known_models)} models with SQL definitions.")
    return known_models

def find_table_to_table_depencies(models: List) -> List:
    logger.info("--- Phase 3: Searching for table to table dependencies ---")
    source_model_ids = set()
    source_models = []

    for model in models:
        if not model.get("source_type") == "model":
            logger.debug(f"Skipping searching table to table dependency for: {model}")
            continue
        # Table-level dependencies
        source_tables_in_query = []
        try:
            main_statement = model.get("main_statement")
            # Find all tables used, excluding CTEs defined within the main statement
            all_tables = list(main_statement.find_all(sqlglot.exp.Table))
            # Searching for CTE
            ctes_in_scope = set()
            with_scope = main_statement.find(sqlglot.exp.With)
            if with_scope:
                # Normalize CTE names for comparison
                ctes_in_scope = {normalize_name(cte.alias_or_name) for cte in with_scope.expressions if cte.alias_or_name}

            # Filter tables, keeping only those that are not CTEs
            source_tables_in_query = [
                tbl for tbl in all_tables
                if normalize_name(tbl.name) not in ctes_in_scope
            ]
            if ctes_in_scope:
                logger.debug(f"Detected CTEs: {ctes_in_scope} in {model.get('model_id')}")

        except Exception as e:
            logger.error(f"Error while searching for source tables in {file_path.name} for {model_id}: {e}")

        processed_upstream_models = set()
        target_schema_norm = normalize_name(model.get('schema'))
        target_table_norm = normalize_name(model.get('name'))

        for table_expr in source_tables_in_query:
            source_table_name = table_expr.name
            source_table_name_norm = normalize_name(source_table_name)

            # Schema from expression or target
            source_schema = table_expr.db if table_expr.db else target_schema
            source_schema_norm = normalize_name(source_schema)

            # Ignore self-reference
            if source_schema_norm == target_schema_norm and source_table_name_norm == target_table_norm:
                continue

            try:
                upstream_model_id = format_node_id(TBL_PREFIX, source_schema, source_table_name)
            except ValueError as e:
                logger.error(f"ID formatting error for source table {source_schema}.{source_table_name} in {model_id}: {e}")
                continue

            if upstream_model_id in processed_upstream_models:
                continue
            processed_upstream_models.add(upstream_model_id)

            # Detecting source models
            if upstream_model_id not in [model.get('model_id') for model in models]:
                source_model_ids.add(upstream_model_id)

            # Adding dependency edge (table depends on other table)
            if not 'table_dependency' in model:
                model['table_dependency'] = []
            model['table_dependency'].append(upstream_model_id)

    if len(source_model_ids) > 0:
        for model_id in source_model_ids:
            model = get_name_from_node_id(model_id)
            source_models.append({
                "schema": model.get("schema"),
                "name": model.get("table"),
                "source_type": 'source_table',
                "model_id": model_id
            })
        logger.info(f"Found {len(source_model_ids)} source model(s) that have not been included into sources description file {config.SQL_SOURCE_MODELS}")
    return models, source_models

def get_column_dependency(model_id: str, target_col_id: str, main_statement: sqlglot.exp.Insert) -> List:
    """
    Calculates the dependencies of columns on each other
    """
    column_dependency = []
    col_name = get_name_from_node_id(target_col_id).get('column')
    # --- Lineage Analysis ---
    lineage_target_column = col_name  # Use the column name as the lineage target
    lineage_sql_statement = main_statement  # Full INSERT or CREATE AS SELECT statement

    # Get the root node of the lineage tree for the target column
    lineage_result_node: Optional[LineageNode] = None
    try:
        lineage_result_node = sqlglot_lineage(
            column=lineage_target_column,
            sql=lineage_sql_statement,
            dialect=getattr(config, "SQL_DIALECT", None),
            # schema=... # Schema can be provided for better resolution
        )
    except NotImplementedError as nie:
        logger.warning(
            f"Lineage is not implemented for the part of the expression related to column '{col_name}' in {model_id}. Error: {nie}"
        )
    except KeyError as ke:
        # Often occurs when sqlglot fails to resolve a column or table name
        logger.warning(
            f"KeyError during lineage for column '{col_name}' in {model_id}: {ke}. The name might not have been resolved."
        )
    except Exception as e:
        logger.error(
            f"Error while executing sqlglot.lineage for column '{col_name}' in {model_id}: {e}",
            exc_info=False
        )

    if lineage_result_node:
        # Searching for source columns in the lineage tree
        source_col_ids: List[
            sqlglot.exp.Column
        ] = find_source_columns_from_lineage(
            lineage_result_node,
            getattr(config, "SQL_DIALECT", None),
            target_col_id  # ID of the target column for comparison
        )

        # Processing the found source columns
        for source_col_id in source_col_ids:
            column_dependency.append({
                'target_col_id': target_col_id,
                'source_col_id': source_col_id
            })
            logger.debug(
                f"Found column dependency: {target_col_id} -> {source_col_id}"
            )

    return column_dependency

def find_column_to_column_depencies(models: List) -> List:
    logger.info("--- Phase 4: Searching for column to column dependencies ---")

    for model in models:
        if not model.get("source_type") == "model":
            logger.debug(f"Skipping searching column to column dependency for: {model}")
            continue

        model['column_dependency'] = {}
        for column in model.get('columns'):
            target_col_id = format_node_id(COL_PREFIX, model.get('schema'), model.get('name'), column)
            model['column_dependency'][column] = get_column_dependency(model.get('model_id'), target_col_id, model.get('main_statement'))

    return models

def draw_nodes(graph: nx.classes.digraph.DiGraph, models: List) -> nx.classes.digraph.DiGraph:
    logger.info("--- Phase 5: Draw nodes ---")
    for model in models:
        # Add a node for the current model (table)
        try:
            node_attrs = {
                "type": TBL_PREFIX,
                "schema": model.get('schema'),
                "name": model.get('name'),
                "source_type": model.get('source_type'),
                "file_path": model.get('file_path')
            }
            graph.add_node(model.get('model_id'), **node_attrs)
            logger.debug(f"Model node added: {model.get('model_id')}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing or adding model node {model.get('model_id')}: {e}")
            continue

        columns = model.get('columns', [])
        if len(columns) > 0:
            for column in columns:
                target_col_id = format_node_id(COL_PREFIX, model.get('schema'), model.get('name'), column)
                col_name_norm = normalize_name(column)

                # Adding a target column node
                col_attrs = {
                    "type": COL_PREFIX,
                    "schema": model.get('schema'),
                    "table": model.get('name'),
                    "column": col_name_norm
                }
                graph.add_node(target_col_id, **col_attrs)
                logger.debug(f"Added colum node {col_name_norm} for {model.get('model_id')}.")
    return graph

def draw_edges(graph: nx.classes.digraph.DiGraph, models: List) -> nx.classes.digraph.DiGraph:
    logger.info("--- Phase 6: Draw edges ---")
    for model in models:
        if 'table_dependency' in model:
            # Adding dependency edge (table depends on other table)
            for upstream_model_id in model.get('table_dependency'):
                if graph.has_node(upstream_model_id):
                    graph.add_edge(model.get('model_id'), upstream_model_id, type='table_dependency')
                    logger.debug(f"Table <-> Table dependency added: {model.get('model_id')} -> {upstream_model_id}")

        columns = model.get('columns', [])
        if len(columns) > 0:
            for column in columns:
                target_col_id = format_node_id(COL_PREFIX, model.get('schema'), model.get('name'), column)
                graph.add_edge(model.get('model_id'), target_col_id, type='contains_column')
                logger.debug(f"Column <-> Table dependency added: {model.get('model_id')} -> {model.get('upstream_model_id')}")
        else:
            logger.debug(f"There are no columns for the model: {model.get('model_id')}")

        if 'column_dependency' in model:
            for column, dependencies in model.get('column_dependency').items():
                for dependency in dependencies:
                    try:
                        # Добавляем ребро зависимости колонок: target_col -> source_col
                        if graph.has_node(dependency.get('source_col_id')):
                            graph.add_edge(
                                dependency.get('target_col_id'), dependency.get('source_col_id'), type="column_dependency"
                            )
                            logger.debug(
                                f"Column <-> Column dependency added: {dependency.get('target_col_id')} -> {dependency.get('source_col_id')}"
                            )
                        else:
                            logger.warning(
                                f"Failed to add edge {dependency.get('target_col_id')} -> {dependency.get('source_col_id')}: source node not found."
                            )

                    except (ValueError, KeyError) as e:
                        logger.error(
                            f"Error while adding edge for source column {model.get('schema')}.{model.get('name')}.{column}: {e}"
                        )
    return graph

# --- Graph saving/loading and comparison functions ---
def save_graph_state(graph: nx.DiGraph, state_file: Path):
    """Saves the graph to a JSON file."""
    logger.info(f"Saving graph state to {state_file}...")
    try:
        # Prepare graph data for JSON serialization
        export_graph = graph.copy() # Work on a copy
        # Convert non-serializable types (like Path) in node/edge attributes
        for node, data in export_graph.nodes(data=True):
            if 'file_path' in data and isinstance(data['file_path'], Path):
                 data['file_path'] = str(data['file_path'])
            # Sanitize other non-serializable types if necessary
            for key, value in list(data.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    logger.debug(f"Converting node attribute '{node}' [{key}]: {type(value)} -> str")
                    data[key] = str(value)
        for u, v, data in export_graph.edges(data=True):
            for key, value in list(data.items()):
                if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    logger.debug(f"Converting edge attribute '({u}, {v})' [{key}]: {type(value)} -> str")
                    data[key] = str(value)

        graph_data = json_graph.node_link_data(export_graph)
        with state_file.open('w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        logger.info("Graph state saved successfully.")
    except TypeError as te:
        logger.error(f"Error serializing graph to JSON: {te}.", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to save graph state: {e}", exc_info=True)

def load_graph_state(state_file: Path) -> Optional[nx.DiGraph]:
    """Loads a graph from a JSON file."""
    state_file_path = Path(state_file) # Verifying Path object
    if not state_file_path.exists():
        logger.info(f"State file {state_file_path} not found. No previous state available.")
        return None
    logger.info(f"Loading graph state from {state_file_path}...")
    try:
        with state_file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        # Use multigraph=False since we don't expect parallel edges of the same type
        graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.info(f"Graph state loaded successfully. Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")
        return graph
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to decode JSON from file {state_file_path}: {jde}")
        return None
    except Exception as e:
        logger.error(f"Failed to load graph state from {state_file_path}: {e}", exc_info=True)
        return None

def find_affected_downstream(graph: nx.DiGraph, changed_node_id: str) -> Dict[str, Set[str]]:
    """
    Finds all downstream nodes (models and columns) that depend on the modified node.
    Uses reverse graph traversal (from source to consumers).
    """
    affected: Dict[str, Set[str]] = {"tables": set(), "columns": set()}

    # Normalize the ID for graph lookup if normalization is enabled
    search_node_id = changed_node_id
    if NORMALIZE_NAMES:
        try:
            parsed_input_id = parse_node_id(changed_node_id)
            search_node_id = format_node_id(
                parsed_input_id['type'],
                parsed_input_id.get('schema'),
                parsed_input_id.get('table'), # parse_node_id возвращает 'table'
                parsed_input_id.get('column')
            )
            if changed_node_id != search_node_id:
                logger.debug(f"Normalized ID for lookup: {search_node_id} (from {changed_node_id})")
        except ValueError as e:
            logger.warning(f"Failed to normalize ID '{changed_node_id}' for lookup: {e}. Lookup will proceed using the original ID.")
            search_node_id = changed_node_id # Searching for "as is"

    if not graph.has_node(search_node_id):
        logger.error(f"Узел '{search_node_id}' не найден в графе.")
        # Try to find without normalization, if it was applied
        if search_node_id != changed_node_id and graph.has_node(changed_node_id):
            logger.warning(f"Node '{search_node_id}' not found, but '{changed_node_id}' was found. Using it instead.")
            search_node_id = changed_node_id
        else:
            return affected

    # nx.ancestors(graph, node) finds all nodes X such that there is a path X -> ... -> node.
    # Since our edges go from target -> source, ancestors(graph, source_node) will return all target_nodes.
    downstream_dependents = nx.ancestors(graph, search_node_id)

    logger.info(f"Searching for nodes dependent on {search_node_id}...")
    for node_id in downstream_dependents:
        try:
            node_data = graph.nodes[node_id]
            node_type = node_data.get("type")
            if node_type == TBL_PREFIX:
                affected["tables"].add(node_id)
            elif node_type == COL_PREFIX:
                affected["columns"].add(node_id)
        except KeyError:
            logger.warning(f"No data found for node {node_id} while searching for dependents.")

    # Add the modified node itself and its immediate components
    try:
        parsed_search_id = parse_node_id(search_node_id)
        if parsed_search_id['type'] == COL_PREFIX:
            affected["columns"].add(search_node_id)
            # Also add the table to which the modified column belongs
            table_id = format_node_id(TBL_PREFIX, parsed_search_id['schema'], parsed_search_id['table'])
            if graph.has_node(table_id):
                 affected["tables"].add(table_id)
        elif parsed_search_id['type'] == TBL_PREFIX:
            affected["tables"].add(search_node_id)
            # If the table has changed, all its columns (in the current graph) are also affected
            # We search via predecessors, since the edges go from table -> column
            for predecessor_id in graph.predecessors(search_node_id):
                edge_data = graph.get_edge_data(predecessor_id, search_node_id)
                # Edges go from table to column via contains_column(table_id, col_id)
                # So we should be using successors(table_id)
            for successor_id in graph.successors(search_node_id):
                edge_data = graph.get_edge_data(search_node_id, successor_id)
                if edge_data and edge_data.get("type") == 'contains_column':
                    if graph.nodes[successor_id].get("type") == COL_PREFIX:
                        affected["columns"].add(successor_id)

    except (ValueError, KeyError) as e:
        logger.warning(f"Error while auto-adding/adding components of node {search_node_id} to affected: {e}")


    logger.info(f"Found {len(affected['tables'])} dependent tables and {len(affected['columns'])} dependent columns for '{search_node_id}'.")
    return affected

def find_significat_changes(previous_graph: nx.DiGraph, current_graph: nx.DiGraph):
    # Comparing states and analyzing the impact of deletions/changes
    if previous_graph:
        logger.info("-" * 30)
        logger.info("Comparing with the previous state:")

        current_nodes = set(current_graph.nodes)
        previous_nodes = set(previous_graph.nodes)
        added_nodes = current_nodes - previous_nodes
        removed_nodes = previous_nodes - current_nodes

        current_edges = set(current_graph.edges)
        previous_edges = set(previous_graph.edges)
        added_edges = current_edges - previous_edges
        removed_edges = previous_edges - current_edges

        # --- Basic change report ---
        has_changes = added_nodes or removed_nodes or added_edges or removed_edges
        if has_changes:
            if added_nodes:
                logger.info(f"Added nodes ({len(added_nodes)}): {sorted(list(added_nodes))}")
            if removed_nodes:
                logger.info(f"Deleted Nodes ({len(removed_nodes)}): {sorted(list(removed_nodes))}")
            if added_edges:
                logger.info(f"Added dependencies ({len(added_edges)}): {sorted([(u,v) for u,v in added_edges])}")
            if removed_edges:
                logger.info(f"Removed dependencies ({len(removed_edges)}): {sorted([(u, v) for u, v in removed_edges])}")
        else:
            logger.info("No structural changes (nodes, edges) detected.")
            save_graph_state(current_graph, config.STATE_FILE)
            logger.info("Dependency analysis completed.")
            return

        # --- Analysis of the impact of removed elements ---
        impacted_by_removal_tables = set()
        impacted_by_removal_columns = set()
        directly_affected_targets = set() # Nodes in the CURRENT graph whose dependencies have DISAPPEARED

        # Looking for targets (predecessors) in the previous_graph that pointed to removed nodes
        for removed_node_id in removed_nodes:
            if removed_node_id in previous_graph:
                # Looking for nodes that *previously* depended on a removed node
                for predecessor_id in previous_graph.predecessors(removed_node_id):
                    # If this dependent node still exists
                    if predecessor_id in current_graph:
                        edge_data = previous_graph.get_edge_data(predecessor_id, removed_node_id)
                        edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
                        logger.debug(f"Node {predecessor_id} (exists) previously depended on the removed node {removed_node_id} via {edge_type}.")
                        directly_affected_targets.add(predecessor_id)

        # Looking for targets (u) whose edges (u, v) were removed, but both nodes u and v still exist
        for u, v in removed_edges:
            if u in current_graph and v in current_graph:
                if previous_graph.has_edge(u, v): # Убедимся, что ребро действительно было
                    edge_data = previous_graph.get_edge_data(u, v)
                    edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
                    logger.debug(f"The {edge_type} dependency of node {u} on {v} was removed (both nodes still exist).")
                    directly_affected_targets.add(u) # Узел 'u' затронут изменением

        # Now find the full downstream impact for directly_affected_targets in the CURRENT graph
        if directly_affected_targets:
            logger.info("-- Analyzing the impact of removed/modified dependencies ---")
            final_impacted_nodes = set() # Collect ALL affected nodes (directly and indirectly)

            for target_id in directly_affected_targets:
                # Add the node whose dependency has changed
                final_impacted_nodes.add(target_id)

                # Find all nodes that depend on this changed node in the CURRENT graph
                # Use nx.ancestors because edges go from target -> source
                try:
                    if current_graph.has_node(target_id):
                        downstream_dependents = nx.ancestors(current_graph, target_id)
                        logger.debug(f"Searching downstream from {target_id}: {downstream_dependents}")
                        final_impacted_nodes.update(downstream_dependents)
                    else:
                        # This shouldn't happen according to the logic above, but just in case
                        logger.warning(f"Node {target_id}, marked as directly affected, is missing from the current graph during downstream search.")

                except nx.NetworkXError as ne:
                    logger.error(f"Error while searching for dependents of {target_id} in the current graph: {ne}")


            # Categorizing affected nodes
            for node_id in final_impacted_nodes:
                # Retrieve data from the current graph if the node exists
                if node_id in current_graph:
                    try:
                        node_data = current_graph.nodes[node_id]
                        node_type = node_data.get("type")
                        if node_type == TBL_PREFIX:
                            impacted_by_removal_tables.add(node_id)
                        elif node_type == COL_PREFIX:
                            impacted_by_removal_columns.add(node_id)
                    except KeyError:
                        logger.warning(f"No data found for existing node {node_id} in the current graph.")
                else:
                    # If the node was affected but is now deleted (e.g., target_id was removed via cascade)
                    # Try to retrieve data from the previous graph
                    if node_id in previous_graph:
                         try:
                            node_data = previous_graph.nodes[node_id]
                            node_type = node_data.get("type")
                            if node_type == TBL_PREFIX:
                                impacted_by_removal_tables.add(f"{node_id} (was removed)")
                            elif node_type == COL_PREFIX:
                                impacted_by_removal_columns.add(f"{node_id} (was removed)")
                         except KeyError:
                            logger.warning(f"No data found for deleted node {node_id} in the previous graph.")
                    else:
                        # A very unusual case
                        logger.warning(f"Affected node {node_id} was not found in either the current or the previous graph.")


            if impacted_by_removal_tables or impacted_by_removal_columns:
                logger.info("Detected impact on the following objects due to removed/modified dependencies:")
                if impacted_by_removal_tables:
                    logger.info(f"  Impacted tables ({len(impacted_by_removal_tables)}): {sorted(list(impacted_by_removal_tables))}")
                if impacted_by_removal_columns:
                    logger.info(f"  Impacted columns ({len(impacted_by_removal_columns)}): {sorted(list(impacted_by_removal_columns))}")
            else:
                logger.info("Directly affected targets were identified, but no remaining objects were impacted (only removed nodes were affected).")
                if directly_affected_targets:
                    logger.info("Dependency changes affected only removed objects.")
                else: # This branch should not be reached: has_changes=True, but directly_affected_targets is empt
                    logger.info("Removed or modified nodes/edges did not break any known dependencies of existing objects.")

        logger.info("-" * 30)

    else:
        logger.info("Previous state not found — skipping comparison and impact analysis.")

def main():
    logger.info("=" * 50)
    logger.info("Starting SQL model dependency analyzer...")
    logger.info(f"Using models directory: {config.SQL_MODELS_DIR}")
    logger.info(f"State file: {config.STATE_FILE}")
    logger.info(f"Name normalization: {NORMALIZE_NAMES}")
    logger.info(f"SQL dialect: {getattr(config, 'SQL_DIALECT', 'Not specified')}")
    logger.info("=" * 50)

    if not config.SQL_MODELS_DIR.is_dir():
        logger.error(f"SQL models directory not found: {config.SQL_MODELS_DIR}")
        logger.error("Please check the path in config.py and make sure the directory exists.")
        return

    sql_files = find_sql_files(config.SQL_MODELS_DIR)
    if not sql_files:
        logger.warning("No SQL files found for analysis.")
        if not Path(config.STATE_FILE).exists():
            logger.info("Creating an empty state file.")
            save_graph_state(nx.DiGraph(), Path(config.STATE_FILE))
        logger.info("Shutting down.")
        return

    # 1. Loading previous state
    previous_graph = load_graph_state(config.STATE_FILE)

    # 2. Building current state
    models = []
    source_models = []
    sql_models = []

    if config.SQL_SOURCE_MODELS.is_file():
        source_models = parse_source_models(config.SQL_SOURCE_MODELS)
    else:
        logger.info("Source models' file was not provided. Source models will be detected from SQL scripts")

    sql_models = parse_sql_models(sql_files, config.SQL_MODELS_DIR)
    models = source_models + sql_models

    models, source_models = find_table_to_table_depencies(models)
    models = models + source_models
    models = find_column_to_column_depencies(models)

    current_graph = nx.DiGraph()
    current_graph = draw_nodes(current_graph, models)
    current_graph = draw_edges(current_graph, models)

    # 3. Changes
    find_significat_changes(previous_graph, current_graph)

    # 4. Save current graph state
    save_graph_state(current_graph, config.STATE_FILE)

    logger.info("Dependency analysis completed.")

if __name__ == "__main__":
    main()
