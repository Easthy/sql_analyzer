# -*- coding: utf-8 -*-
import json
import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any

import networkx as nx
import sqlglot
import yaml
from networkx.readwrite import json_graph
from rich.logging import RichHandler
from sqlglot import Expression
import sqlglot.expressions as exp
from sqlglot.errors import ParseError
from sqlglot.lineage import Node as LineageNode
from sqlglot.lineage import lineage as sqlglot_lineage

import config

# Disable warning that could happen during work with networkx and JSON
warnings.simplefilter(action='ignore', category=FutureWarning)

TBL_PREFIX = "tbl"
COL_PREFIX = "col"

class ConfigManager:
    """
    Manages application configuration and sets up logging.
    Ensures that all required configuration parameters are present.
    """
    def __init__(self):
        self._validate_config()
        self.sql_models_dir = config.SQL_MODELS_DIR
        self.state_file = config.STATE_FILE
        self.sql_dialect = getattr(config, 'SQL_DIALECT', None)
        self.sql_file_extension = getattr(config, 'SQL_FILE_EXTENSION', '.sql')
        self.normalize_names = getattr(config, 'NORMALIZE_NAMES', True)
        self.source_models_file = getattr(config, 'SQL_SOURCE_MODELS', None)
        
        self.setup_logging()

    def _validate_config(self):
        """Checks for the presence of required attributes in the config module."""
        required = ['SQL_MODELS_DIR', 'STATE_FILE', 'SQL_DIALECT', 'SQL_FILE_EXTENSION']
        missing = [cfg for cfg in required if not hasattr(config, cfg)]
        if missing:
            raise ValueError(f"Error: Missing required parameters in config.py: {', '.join(missing)}")

    def setup_logging(self):
        """Configures the application's logger."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        log_level = getattr(config, 'LOG_LEVEL', 'INFO')
        log_format = getattr(config, 'LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            encoding='utf-8',
            handlers=[RichHandler(rich_tracebacks=True, show_time=False, show_path=False)]
        )
        logging.getLogger('sql_analyzer')


class NameUtils:
    """A collection of static utility methods for handling node names and IDs."""
    
    _normalize_enabled = getattr(config, 'NORMALIZE_NAMES', True)
    
    @classmethod
    def normalize_name(cls, name: Optional[str]) -> Optional[str]:
        """Converts the name to lowercase if normalization is enabled."""
        if cls._normalize_enabled and name:
            return name.lower()
        return name

    @classmethod
    def format_node_id(cls, node_type: str, schema: Optional[str], name: Optional[str], column: Optional[str] = None) -> str:
        """Generates a unique ID for a graph node with optional normalization."""
        schema_norm = cls.normalize_name(schema if schema else 'unknown_schema')
        name_norm = cls.normalize_name(name if name else 'unknown_table')

        if node_type == TBL_PREFIX:
            if not schema_norm or not name_norm:
                raise ValueError(f"Schema and name are required for table node: schema='{schema}', name='{name}'")
            return f"{TBL_PREFIX}:{schema_norm}.{name_norm}"
        elif node_type == COL_PREFIX:
            if not schema_norm or not name_norm or not column:
                raise ValueError(f"Schema, table name, and column are required: schema='{schema}', name='{name}', column='{column}'")
            col_norm = cls.normalize_name(column)
            return f"{COL_PREFIX}:{schema_norm}.{name_norm}.{col_norm}"
        else:
            raise ValueError(f"Unknown node type: {node_type}")

    @classmethod
    def parse_node_id(cls, node_id: str) -> Dict[str, Optional[str]]:
        """Parses the node ID into its components."""
        if not isinstance(node_id, str):
            raise ValueError(f"Cannot parse node ID: Expected string, got {type(node_id)} ({node_id})")

        parts = node_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Cannot parse node ID: {node_id}. Invalid format.")
        
        node_type, full_name = parts
        name_parts = full_name.split('.')

        if node_type == TBL_PREFIX:
            if len(name_parts) < 2:
                raise ValueError(f"Cannot parse table node ID: {node_id}. Expected 'tbl:schema.table'")
            schema, table = name_parts[0], '.'.join(name_parts[1:])
            return {"type": node_type, "schema": schema, "table": table, "column": None}
        elif node_type == COL_PREFIX:
            if len(name_parts) < 3:
                raise ValueError(f"Cannot parse column node ID: {node_id}. Expected 'col:schema.table.column'")
            schema, table, column = name_parts[0], name_parts[1], '.'.join(name_parts[2:])
            return {"type": node_type, "schema": schema, "table": table, "column": column}
        else:
            raise ValueError(f"Cannot parse node ID: Unknown type '{node_type}' in {node_id}")


class ModelParser(ABC):
    """Abstract base class for model parsers."""
    def __init__(self, cfg: ConfigManager):
        self.config = cfg
        self.logger = logging.getLogger('sql_analyzer.parser')
    
    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """Parses models and returns a list of model definitions."""
        pass

class SourceModelParser(ModelParser):
    """Parses source models defined in a YAML file."""
    
    def parse(self) -> List[Dict[str, Any]]:
        """Parses the source YAML file and returns a list of source table models."""
        self.logger.info("--- Phase 1: Parsing source models ---")
        if not self.config.source_models_file or not self.config.source_models_file.is_file():
            self.logger.info("Source models file not provided or not found. Skipping.")
            return []

        try:
            with self.config.source_models_file.open('r') as file:
                source_definitions = yaml.safe_load(file)
            self.logger.info(f"Loaded source tables: {list(source_definitions.keys())}")
        except Exception as e:
            self.logger.error(f"Failed to load or parse sources file: {e}")
            return []

        source_models = []
        for table, columns in source_definitions.items():
            schema, table_name = table.split('.')
            source_models.append({
                "schema": schema,
                "name": table_name,
                "source_type": 'source_table',
                "columns": columns or [],
                "model_id": NameUtils.format_node_id(TBL_PREFIX, schema, table_name)
            })
        return source_models

class SqlModelParser(ModelParser):
    """Parses SQL files to extract model definitions and operations."""

    def parse(self) -> List[Dict[str, Any]]:
        """Finds and parses all SQL files in the configured directory."""
        self.logger.info("--- Phase 2: Parsing SQL models ---")
        sql_files = self._find_sql_files()
        if not sql_files:
            self.logger.warning("No SQL files found for analysis.")
            return []

        all_operations = []
        processed_files = 0
        total_files = len(sql_files)

        for file_path in sql_files:
            processed_files += 1
            schema, table_name = self._extract_model_name_from_path(file_path)
            self.logger.info(f"[{processed_files}/{total_files}] Analyzing: {file_path} (model: {schema}.{table_name})")

            if schema and table_name:
                model_id = NameUtils.format_node_id(TBL_PREFIX, schema, table_name)
                file_operations = self._parse_sql_file(model_id, file_path)
                if file_operations:
                    all_operations.extend(file_operations)
            else:
                self.logger.warning(f"Skipping file (could not determine model name): {file_path.name}")
        
        self.logger.info(f"Parsed {len(all_operations)} relevant CREATE/INSERT operations from {processed_files} files.")
        return all_operations

    def _find_sql_files(self) -> List[Path]:
        """Recursively finds all SQL files in the directory."""
        self.logger.info(f"Searching for SQL files in: {self.config.sql_models_dir}")
        sql_files = list(self.config.sql_models_dir.rglob(f"*{self.config.sql_file_extension}"))
        self.logger.info(f"Found {len(sql_files)} SQL files.")
        return sql_files
    
    def _extract_model_name_from_path(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Extracts schema.table from a filename like 'schema.table_name.sql'."""
        filename_stem = file_path.stem
        name_parts = filename_stem.split('.')
        if len(name_parts) < 2:
            self.logger.warning(f"Invalid filename format: '{file_path.name}'. Expected 'schema.table.sql'.")
            return None, None
        
        schema, table_name = name_parts[0], '.'.join(name_parts[1:])
        return schema, table_name
    
    def _parse_sql_file(self, model_id: str, file_path: Path) -> List[Dict]:
        """Parses a single SQL file for all CREATE/INSERT operations."""
        relative_path = str(file_path.relative_to(self.config.sql_models_dir))
        try:
            sql_content = file_path.read_text(encoding='utf-8')
            statements = sqlglot.parse(sql=sql_content, read=self.config.sql_dialect)
        except Exception as e:
            self.logger.error(f"Error reading or parsing file {file_path}: {e}")
            return []

        if not statements:
            return []

        parsed_operations = [
            op for stmt in statements if stmt and (op := self._extract_statement_info(stmt, model_id, relative_path))
        ]
        
        return self._filter_duplicate_operations(parsed_operations)

    def _filter_duplicate_operations(self, operations: List[Dict]) -> List[Dict]:
        """Filters out CREATE statements if an INSERT for the same table exists."""
        inserts = {op['model_id']: op for op in operations if isinstance(op['main_statement'], exp.Insert)}
        creates = {op['model_id']: op for op in operations if isinstance(op['main_statement'], exp.Create)}

        final_ops = []
        for op in operations:
            model_id = op['model_id']

            if isinstance(op['main_statement'], exp.Insert):
                if model_id in creates: # If a CREATE also exists, use its column definition
                    op['columns'] = creates[model_id]['columns']
                final_ops.append(op)
            elif isinstance(op['main_statement'], exp.Create):
                if model_id not in inserts: # Only add CREATE if no INSERT exists for this model
                    final_ops.append(op)
        
        return final_ops

    def _extract_statement_info(self, statement: Expression, file_model_id: str, file_path_str: str) -> Optional[Dict]:
        """Extracts details from a single CREATE TABLE or INSERT statement."""
        try:
            target_schema, target_table, op_type = None, None, None
            target_columns: List[str] = []

            if isinstance(statement, exp.Create) and statement.kind and statement.kind.upper() == 'TABLE':
                op_type = 'CREATE'
                table_expr = statement.this.find(exp.Table)
                if not table_expr: return None
                target_schema, target_table = table_expr.db, table_expr.name
                
                # Columns from CREATE TABLE (...)
                schema_def = statement.this.find(exp.Schema)
                if schema_def and schema_def.expressions:
                    target_columns = [col.this.name for col in schema_def.expressions if isinstance(col, exp.ColumnDef)]
                
                # Columns from CREATE TABLE AS SELECT ...
                elif isinstance(statement.expression, (exp.Select, exp.Union)):
                    select_part = statement.expression.find(exp.Select)
                    if select_part:
                        target_columns = [str(col.alias_or_name) for col in select_part.expressions if col.alias_or_name]

            elif isinstance(statement, exp.Insert):
                op_type = 'INSERT'
                table_expr = statement.this.find(exp.Table)
                if not table_expr: return None
                target_schema, target_table = table_expr.db, table_expr.name

                # Extract target columns if explicitly listed: INSERT INTO tbl (col1, col2) ...
                columns_in_target = None
                
                # Columns from INSERT INTO tbl (col1, col2)
                cols_spec = statement.this

                if isinstance(cols_spec, exp.Schema) and cols_spec.expressions:
                     # Check if the expressions following the table are columns
                     if all(isinstance(e, exp.Identifier) for e in cols_spec.expressions):
                         columns_in_target = cols_spec.expressions
                elif isinstance(cols_spec, exp.Tuple): # Often used for column lists directly after table name
                     columns_in_target = cols_spec.expressions
                # Check the 'alias' argument which sometimes holds the column list
                elif 'alias' in statement.args and isinstance(statement.args['alias'], exp.Tuple):
                     columns_in_target = statement.args['alias'].expressions

                if columns_in_target:
                    target_columns = [
                        col.name for col in columns_in_target if isinstance(col, exp.Identifier)
                    ]

                # Columns from SELECT statement
                elif isinstance(statement.expression, (exp.Select, exp.Union)):
                    select_part = statement.expression.find(exp.Select)
                    if select_part:
                        target_columns = [str(col.alias_or_name) for col in select_part.expressions if col.alias_or_name]

            if op_type and target_table:
                target_schema = target_schema or '_temp'
                return {
                    "schema": target_schema,
                    "name": target_table,
                    "source_type": 'model',
                    "file_path": file_path_str,
                    "columns": [col for col in target_columns if isinstance(col, str)],
                    "main_statement": statement,
                    "model_id": NameUtils.format_node_id(TBL_PREFIX, target_schema, target_table)
                }

        except Exception as e:
            self.logger.error(f"Error processing statement in {file_path_str}: {e}", exc_info=True)
        return None

class DependencyAnalyzer:
    """Analyzes table-to-table and column-to-column dependencies in parsed models."""
    
    def __init__(self, cfg: ConfigManager):
        self.config = cfg
        self.logger = logging.getLogger('sql_analyzer.analyzer')

    def analyze_dependencies(self, models: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Runs all dependency analysis phases."""
        models, new_sources = self.analyze_table_dependencies(models)
        models = self.analyze_column_dependencies(models)
        return models, new_sources
        
    def analyze_table_dependencies(self, models: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Finds table-level dependencies for each model."""
        self.logger.info("--- Phase 3: Analyzing table-to-table dependencies ---")
        existing_model_ids = {m['model_id'] for m in models}
        undiscovered_source_ids = set()

        for model in models:
            if model.get("source_type") != "model":
                continue
            
            model['table_dependency'] = []
            main_statement = model.get("main_statement")
            if not main_statement: continue
                
            try:
                # Find all CTE to exclude then from the list of source tables
                with_scope = main_statement.find(exp.With)
                ctes = {cte.alias_or_name for cte in with_scope.expressions} if with_scope else set()
                
                source_tables = [
                    tbl for tbl in main_statement.find_all(exp.Table)
                    if NameUtils.normalize_name(tbl.name) not in {NameUtils.normalize_name(c) for c in ctes}
                ]

                processed_upstreams = set()
                for table_expr in source_tables:
                    source_schema = table_expr.db if table_expr.db else '_temp'
                    source_table = table_expr.name

                    # Ignore self-references
                    if NameUtils.normalize_name(source_schema) == NameUtils.normalize_name(model['schema']) and \
                       NameUtils.normalize_name(source_table) == NameUtils.normalize_name(model['name']):
                        continue

                    upstream_id = NameUtils.format_node_id(TBL_PREFIX, source_schema, source_table)
                    if upstream_id in processed_upstreams:
                        continue
                    
                    model['table_dependency'].append(upstream_id)
                    processed_upstreams.add(upstream_id)

                    if upstream_id not in existing_model_ids:
                        undiscovered_source_ids.add(upstream_id)
            except Exception as e:
                 self.logger.error(f"Error finding table dependencies for {model['model_id']}: {e}", exc_info=True)
        
        new_source_models = []
        if undiscovered_source_ids:
            self.logger.info(f"Found {len(undiscovered_source_ids)} new source tables from queries.")
            for model_id in undiscovered_source_ids:
                parsed_id = NameUtils.parse_node_id(model_id)
                new_source_models.append({
                    "schema": parsed_id["schema"], "name": parsed_id["table"],
                    "source_type": 'source_table', "model_id": model_id, "columns": []
                })

        return models, new_source_models

    def analyze_column_dependencies(self, models: List[Dict]) -> List[Dict]:
        """Finds column-level dependencies for each model."""
        self.logger.info("--- Phase 4: Analyzing column-to-column dependencies ---")
        for model in models:
            if model.get("source_type") != "model" or not model.get('columns'):
                continue
            
            model['column_dependency'] = {}
            main_statement = model.get("main_statement")
            if not main_statement: continue

            # Map target columns to their corresponding SELECT expressions
            try:
                target_to_source_map = dict(zip(model['columns'], main_statement.named_selects))
            except Exception as e:
                self.logger.warning(f"Could not map columns for {model['model_id']}: {e}")
                continue
            
            for col_name in model['columns']:
                select_col_name = target_to_source_map.get(col_name) # Replace searched column's name by a name from INSERT/CREATE statement
                if not select_col_name: continue

                target_col_id = NameUtils.format_node_id(COL_PREFIX, model['schema'], model['name'], col_name)
                try:
                    lineage_node = sqlglot_lineage(
                        column=select_col_name,
                        sql=main_statement,
                        dialect=self.config.sql_dialect
                    )
                    source_col_ids = self._find_source_cols_from_lineage_node(lineage_node)
                    if source_col_ids:
                         model['column_dependency'][col_name] = [
                             {'target_col_id': target_col_id, 'source_col_id': src_id} for src_id in source_col_ids
                         ]
                except Exception as e:
                    self.logger.warning(f"Could not analyze lineage for column '{col_name}' in {model['model_id']}: {e}")
        return models

    def _find_source_cols_from_lineage_node(self, node: Optional[LineageNode]) -> List[str]:
        """
        Recursively traverses the lineage tree to find leaf column nodes (the true sources).
        This is the corrected version.
        """
        if not node:
            return []
        
        source_cols = []
        
        def traverse(current_node: LineageNode, visited: set):
            if id(current_node) in visited:
                return
            visited.add(id(current_node))

            # A leaf node has no further downstream dependencies, it's a source.
            if not current_node.downstream:
                # The source expression, usually a Table
                source_expr = current_node.source
                
                # We only care about columns that come from actual tables
                if isinstance(source_expr, exp.Table):
                    table_name = source_expr.name  # current_node.name is an alias of a table
                    schema_name = str(current_node.source).split('.')[0] if '.' in str(current_node.source) else '_temp'
                    # The column name at the source. It might be qualified like 'table.col'.
                    # Get the clean column name, e.g., 'col' from 'table.col'
                    clean_col_name = current_node.name.split('.')[-1] if '.' in current_node.name else current_node.name

                    source_id = NameUtils.format_node_id(
                        node_type=COL_PREFIX,
                        schema=schema_name, # Can be None, format_node_id will handle it
                        name=table_name,
                        column=clean_col_name
                    )
                    if source_id not in source_cols:
                        source_cols.append(source_id)

            # Recurse down the tree
            for child in current_node.downstream:
                traverse(child, visited)

        traverse(node, set())
        return source_cols


class LineageGraph:
    """Manages the NetworkX graph, including building, saving, and loading."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.logger = logging.getLogger('sql_analyzer.graph')

    def build_from_models(self, models: List[Dict]):
        """Builds the entire graph from a list of model definitions."""
        self.logger.info("--- Phase 5: Building dependency graph ---")
        self._draw_nodes(models)
        self._draw_edges(models)
        self.logger.info(f"Graph built. Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}.")

    def _draw_nodes(self, models: List[Dict]):
        """Adds table and column nodes to the graph."""
        for model in models:
            try:
                self.graph.add_node(
                    model['model_id'], type=TBL_PREFIX, schema=model['schema'], name=model['name'],
                    source_type=model.get('source_type'), file_path=model.get('file_path')
                )
                for column in model.get('columns', []):
                    col_id = NameUtils.format_node_id(COL_PREFIX, model['schema'], model['name'], column)
                    self.graph.add_node(
                        col_id, type=COL_PREFIX, schema=model['schema'], table=model['name'], column=column
                    )
            except Exception as e:
                self.logger.error(f"Error adding nodes for model {model.get('model_id')}: {e}")

    def _draw_edges(self, models: List[Dict]):
        """Adds dependency edges to the graph."""
        for model in models:
            model_id = model['model_id']
            # Table -> Column containment
            for column in model.get('columns', []):
                col_id = NameUtils.format_node_id(COL_PREFIX, model['schema'], model['name'], column)
                if self.graph.has_node(model_id) and self.graph.has_node(col_id):
                    self.graph.add_edge(model_id, col_id, type='contains_column')

            # Table -> Table dependency
            for upstream_id in model.get('table_dependency', []):
                if self.graph.has_node(model_id) and self.graph.has_node(upstream_id):
                    self.graph.add_edge(model_id, upstream_id, type='table_dependency')

            # Column -> Column dependency
            for deps in model.get('column_dependency', {}).values():
                for dep in deps:
                    target, source = dep['target_col_id'], dep['source_col_id']
                    if self.graph.has_node(target) and self.graph.has_node(source):
                        self.graph.add_edge(target, source, type='column_dependency')
    
    def save_state(self, state_file: Path):
        """Saves the graph to a JSON file."""
        self.logger.info(f"Saving graph state to {state_file}...")
        try:
            graph_data = json_graph.node_link_data(self.graph)
            with state_file.open('w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            self.logger.info("Graph state saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save graph state: {e}", exc_info=True)

    @classmethod
    def load_state(cls, state_file: Path) -> Optional['LineageGraph']:
        """Loads a graph from a JSON file."""
        if not state_file.exists():
            logging.getLogger('sql_analyzer.graph').info(f"State file {state_file} not found. No previous state.")
            return None
        
        logging.getLogger('sql_analyzer.graph').info(f"Loading graph state from {state_file}...")
        try:
            with state_file.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            instance = cls()
            instance.graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
            logging.getLogger('sql_analyzer.graph').info(f"Graph state loaded. Nodes: {instance.graph.number_of_nodes()}, Edges: {instance.graph.number_of_edges()}.")
            return instance
        except Exception as e:
            logging.getLogger('sql_analyzer.graph').error(f"Failed to load graph state: {e}", exc_info=True)
            return None


class ChangeDetector:
    """Compares two LineageGraph instances and reports significant changes."""

    def __init__(self, old_graph: Optional[LineageGraph], new_graph: LineageGraph):
        self.old_graph = old_graph.graph if old_graph else nx.DiGraph()
        self.new_graph = new_graph.graph
        self.logger = logging.getLogger('sql_analyzer.detector')

    def report_changes(self):
        """Finds and logs additions, removals, and their impact."""
        if not list(self.old_graph.nodes):
            self.logger.info("Previous state not found — skipping comparison.")
            return
            
        self.logger.info("-" * 30)
        self.logger.info("Comparing with previous state:")

        old_nodes, new_nodes = set(self.old_graph.nodes), set(self.new_graph.nodes)
        added_nodes = new_nodes - old_nodes
        removed_nodes = old_nodes - new_nodes

        old_edges, new_edges = set(self.old_graph.edges), set(self.new_graph.edges)
        added_edges = new_edges - old_edges
        removed_edges = old_edges - new_edges

        if not any([added_nodes, removed_nodes, added_edges, removed_edges]):
            self.logger.info("No structural changes detected.")
            return

        if added_nodes: self.logger.info(f"Added nodes ({len(added_nodes)}): {sorted(list(added_nodes))}")
        if removed_nodes: self.logger.info(f"Deleted nodes ({len(removed_nodes)}): {sorted(list(removed_nodes))}")
        if added_edges: self.logger.info(f"Added dependencies ({len(added_edges)}): {sorted([f'{u} -> {v}' for u, v in added_edges])}")
        if removed_edges: self.logger.info(f"Removed dependencies ({len(removed_edges)}): {sorted([f'{u} -> {v}' for u, v in removed_edges])}")

        self._analyze_impact(removed_nodes, removed_edges)
        self.logger.info("-" * 30)

    def _analyze_impact(self, removed_nodes: Set[str], removed_edges: Set[Tuple[str, str]]):
        """Analyzes the downstream impact of removed nodes and edges."""
        
        # Find nodes in the new graph that were affected by removals
        directly_affected = set()
        for u, v in removed_edges:
            if u in self.new_graph and v in self.new_graph:
                directly_affected.add(u) # A dependency was removed
        
        for node in removed_nodes:
            # Find nodes that USED to depend on the removed node
            if node in self.old_graph:
                for predecessor in self.old_graph.predecessors(node):
                    if predecessor in self.new_graph:
                        directly_affected.add(predecessor)

        if not directly_affected:
            self.logger.info("Removed elements did not affect any existing models.")
            return

        self.logger.info("--- Analyzing impact of removed/modified dependencies ---")
        all_impacted_nodes = set()
        for node_id in directly_affected:
            all_impacted_nodes.add(node_id)
            # Find all nodes that depend on the directly affected node
            # In our graph, ancestors are consumers (dependents)
            if node_id in self.new_graph:
                downstream_dependents = nx.ancestors(self.new_graph, node_id)
                all_impacted_nodes.update(downstream_dependents)

        impacted_tables = {n for n in all_impacted_nodes if self.new_graph.nodes[n].get('type') == TBL_PREFIX}
        impacted_columns = {n for n in all_impacted_nodes if self.new_graph.nodes[n].get('type') == COL_PREFIX}
        
        if impacted_tables: self.logger.info(f"  Impacted tables ({len(impacted_tables)}): {sorted(list(impacted_tables))}")
        if impacted_columns: self.logger.info(f"  Impacted columns ({len(impacted_columns)}): {sorted(list(impacted_columns))}")


class SQLAnalyzer:
    """Orchestrates the entire SQL dependency analysis process."""
    
    def __init__(self):
        self.config = ConfigManager()
        self.logger = logging.getLogger('sql_analyzer')
        # Dependency Injection: The analyzer uses components but doesn't create them.
        self.source_parser = SourceModelParser(self.config)
        self.sql_parser = SqlModelParser(self.config)
        self.dependency_analyzer = DependencyAnalyzer(self.config)

    def run(self):
        """Executes the full analysis pipeline."""
        self.logger.info("=" * 50)
        self.logger.info("Starting SQL model dependency analyzer...")
        self.logger.info(f"Models directory: {self.config.sql_models_dir}")
        self.logger.info(f"State file: {self.config.state_file}")
        self.logger.info(f"SQL dialect: {self.config.sql_dialect}")
        self.logger.info("=" * 50)

        # 1. Load previous state
        old_graph = LineageGraph.load_state(self.config.state_file)

        # 2. Parse all models from all sources
        source_models = self.source_parser.parse()
        sql_models = self.sql_parser.parse()
        models = source_models + sql_models

        # 3. Analyze dependencies
        models, new_source_models = self.dependency_analyzer.analyze_table_dependencies(models)
        all_models = models + new_source_models
        all_models = self.dependency_analyzer.analyze_column_dependencies(all_models)
        
        # 4. Build the new graph
        current_graph = LineageGraph()
        current_graph.build_from_models(all_models)

        # 5. Compare and report changes
        change_detector = ChangeDetector(old_graph, current_graph)
        change_detector.report_changes()

        # 6. Save the new state
        current_graph.save_state(self.config.state_file)

        self.logger.info("Dependency analysis completed successfully.")


if __name__ == "__main__":
    try:
        analyzer = SQLAnalyzer()
        analyzer.run()
    except Exception as e:
        logging.getLogger('sql_analyzer').critical(f"A critical error occurred: {e}", exc_info=True)
        exit(1)