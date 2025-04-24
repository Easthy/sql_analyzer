# -*- coding: utf-8 -*-
import os
import json
import logging
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
warnings.simplefilter(action='ignore', category=FutureWarning) # Suppress pandas future warning from networkx/json

# --- Импорт config ---
try:
    import config
except ImportError:
    print("Ошибка: Файл конфигурации 'config.py' не найден.")
    print("Создайте файл 'config.py' с необходимыми настройками.")
    exit(1)
except Exception as e: # Catch broader errors during config import/access
    print(f"Ошибка при импорте или доступе к config.py: {e}")
    exit(1)

# --- Проверка наличия необходимых атрибутов в config ---
required_configs = ['SQL_MODELS_DIR', 'STATE_FILE', 'SQL_DIALECT', 'SQL_FILE_EXTENSION']
missing_configs = [cfg for cfg in required_configs if not hasattr(config, cfg)]
if missing_configs:
    print(f"Ошибка: В config.py отсутствуют следующие обязательные параметры: {', '.join(missing_configs)}")
    exit(1)

# --- Логирование и Константы ---
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

# --- Утилиты для имен и ID ---

def normalize_name(name: Optional[str]) -> Optional[str]:
    """Приводит имя к нижнему регистру, если включена нормализация."""
    if NORMALIZE_NAMES and name:
        return name.lower()
    return name

def format_node_id(node_type: str, schema: Optional[str], name: Optional[str], column: Optional[str] = None) -> str:
    """Формирует уникальный ID для узла графа с опциональной нормализацией."""
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

def parse_node_id(node_id: str) -> Dict[str, Optional[str]]:
    """Разбирает ID узла на компоненты."""
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

# --- Функции работы с файлами и SQL ---
def find_sql_files(directory: Path) -> List[Path]:
    """Рекурсивно находит все файлы .sql в указанной директории."""
    logger.info(f"Поиск SQL файлов в: {directory}")
    sql_files = list(directory.rglob(f"*{getattr(config, 'SQL_FILE_EXTENSION', '.sql')}"))
    logger.info(f"Найдено {len(sql_files)} SQL файлов.")
    if not sql_files:
         logger.warning(f"В директории {directory} не найдено файлов с расширением {getattr(config, 'SQL_FILE_EXTENSION', '.sql')}")
    return sql_files

def extract_model_name_from_path(file_path: Path, root_dir: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Извлекает имя модели (schema.table) из пути к файлу.
    Использует имя файла как 'schema.table_name.sql'.
    """
    filename_stem = file_path.stem
    name_parts = filename_stem.split('.')

    if len(name_parts) < 2:
        logger.warning(f"Не удалось определить схему и таблицу из имени файла: {file_path.name}. Имя должно быть в формате 'schema.table.sql'. Пропускается.")
        return None, None

    schema = name_parts[0]
    table_name = '.'.join(name_parts[1:])

    logger.debug(f"Файл: {file_path.name}, Схема: {schema}, Таблица: {table_name}")
    return schema, table_name


def find_main_statement(sql_content: str, target_schema: str, target_table: str) -> Optional[Expression]:
    """
    Парсит SQL и находит 'основной' стейтмент (INSERT или CREATE), определяющий целевую таблицу.
    """
    expressions: List[Expression] = []
    try:
        expressions = sqlglot.parse(sql_content, read=getattr(config, 'SQL_DIALECT', None))
    except ParseError as e:
        logger.error(f"Ошибка парсинга SQL для {target_schema}.{target_table}: {e}")
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
                logger.error(f"  Ошибка: {error_info.get('description')} (строка ~{line}, столбец ~{col})")
                logger.error(f"  Контекст: ...{problem_sql}...")
        else:
             logger.error("  Детали ошибки парсинга недоступны.")
        return None
    except Exception as e:
        logger.error(f"Неожиданная ошибка при парсинге SQL для {target_schema}.{target_table}: {e}")
        logger.debug(f"SQL контент (начало):\n{sql_content[:500]}...")
        return None

    if not expressions:
        logger.warning(f"Не найдено SQL выражений в файле для {target_schema}.{target_table}")
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
                    logger.debug(f"Найдено INSERT выражение для {target_schema}.{target_table}")
                    break
                else:
                    logger.debug(f"Найден INSERT в таблицу '{schema_name}.{table_name}', но не совпадает с целевой '{target_schema}.{target_table}'.")

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
                    logger.debug(f"Найдено CREATE {kind} выражение для {target_schema}.{target_table}")
                    break
                else:
                    logger.debug(f"Найден CREATE {kind} для '{schema_name}.{table_name}', но не совпадает с целевой '{target_schema}.{target_table}'.")

    if not main_statement:
         logger.warning(f"Не найдено основное INSERT/CREATE TABLE/VIEW выражение для {target_schema}.{target_table} в файле.")
         # Optionally, try to find the *last* SELECT statement as a fallback? Risky.
         # last_select = next((e for e in reversed(expressions) if isinstance(e, exp.Select)), None)
         # if last_select:
         #    logger.warning("Используем последний SELECT как основное выражение (может быть неточно)")
         #    main_statement = last_select

    return main_statement

def find_source_columns_from_lineage(node: LineageNode, target_col_name: str, dialect: str) -> List[sqlglot.exp.Column]:
    """
    Рекурсивно обходит дерево lineage Node и собирает все выражения
    sqlglot.exp.Column, которые представляют собой конечные источники
    (листовые узлы типа Column в дереве lineage с информацией о таблице).
    """
    sources: List[sqlglot.exp.Column] = []
    processed_node_ids = set() # Для предотвращения циклов

    def traverse(current_node: LineageNode):
        node_id = id(current_node)
        if node_id in processed_node_ids:
            return
        processed_node_ids.add(node_id)

        # Сравниваем нормализованные имена, чтобы не добавить саму цель как источник случайно
        # (хотя lineage обычно ее не возвращает в листьях)
        is_target_node = normalize_name(current_node.name) == normalize_name(target_col_name)

        # Проверяем, является ли текущий узел листовым в дереве lineage
        is_leaf_node = not current_node.downstream

        if is_leaf_node:
            # Если листовой узел сам является колонкой (и не целевой)
            if isinstance(current_node.expression, sqlglot.exp.Column) and not is_target_node:
                # Добавляем, ТОЛЬКО если есть информация об источнике (table/alias)
                source_expr = current_node.expression
                if source_expr.table:
                    sources.append(source_expr)
                else:
                    # Логируем колонки без таблицы, которые пропускаем
                    logger.debug(f"Пропуск листовой колонки-источника '{source_expr.sql(dialect=dialect)}' без имени таблицы/алиаса в lineage для '{target_col_name}'.")
            else: # Листовой узел не колонка (например, литерал, функция без колонок)
                logger.debug(f"Листовой узел lineage для '{target_col_name}' не является Column: {type(current_node.expression)} [{current_node.name}]")

        # Рекурсивно обходим дочерние узлы ("downstream" зависимости)
        # Источники могут быть только в листьях дерева lineage
        for child_node in current_node.downstream:
            traverse(child_node)

    # Начинаем обход с корневого узла, который вернул lineage()
    if node:
        traverse(node)

    # Убираем дубликаты объектов Column (на всякий случай)
    unique_sources_dict = {id(expr): expr for expr in sources}
    unique_sources = list(unique_sources_dict.values())

    # Логируем найденные источники
    source_sqls = [s.sql(dialect=dialect) for s in unique_sources]
    if source_sqls:
        logger.debug(f"Найденные конечные колонки-источники для '{target_col_name}': {source_sqls}")

    return unique_sources

def resolve_table_alias(source_table_alias: Optional[str],
                         source_schema_alias: Optional[str],
                         source_tables_in_query: iter, # Iterator of sqlglot.exp.Table
                         target_schema: str # Default schema
                         ) -> Tuple[Optional[str], Optional[str]]:
    """
    Пытается разрешить алиас таблицы/схемы к реальному имени таблицы и схемы.
    Возвращает Tuple(actual_schema, actual_table_name) или (None, None).
    """
    if not source_table_alias:
        logger.debug("Alias resolution failed: No source table alias provided.")
        return None, None

    source_table_alias_norm = normalize_name(source_table_alias)
    source_schema_alias_norm = normalize_name(source_schema_alias)
    target_schema_norm = normalize_name(target_schema)

    # Создаем копию итератора, чтобы не истощить его при многократных вызовах
    source_tables_list = list(source_tables_in_query)

    for table_expr in source_tables_list:
        if not isinstance(table_expr, sqlglot.exp.Table): continue

        current_table_name = table_expr.name
        current_schema_name = table_expr.db
        current_alias = table_expr.alias

        current_table_name_norm = normalize_name(current_table_name)
        current_schema_name_norm = normalize_name(current_schema_name) # Schema from expression
        current_alias_norm = normalize_name(current_alias)

        # Определяем "эффективную" схему для текущей таблицы в запросе
        effective_schema_norm = current_schema_name_norm if current_schema_name_norm else target_schema_norm

        match = False
        # 1. Сравнение с алиасом таблицы
        if current_alias_norm and current_alias_norm == source_table_alias_norm:
            # Если схема в колонке указана (source_schema_alias_norm), она должна совпадать с эффективной схемой алиаса
            if source_schema_alias_norm is None or effective_schema_norm == source_schema_alias_norm:
                match = True
        # 2. Сравнение с именем таблицы (если алиаса нет у table_expr)
        elif not current_alias and current_table_name_norm == source_table_alias_norm:
            # Если схема в колонке указана (source_schema_alias_norm), она должна совпадать с эффективной схемой таблицы
            if source_schema_alias_norm is None or effective_schema_norm == source_schema_alias_norm:
                match = True

        if match:
            actual_table_name = current_table_name # Оригинальное имя таблицы
            actual_schema = current_schema_name if current_schema_name else target_schema # Оригинальное или target
            logger.debug(f"Resolved alias/table '{source_table_alias}' (schema hint: {source_schema_alias}) -> {actual_schema}.{actual_table_name} (based on: {table_expr.sql()})")
            return actual_schema, actual_table_name

    # Если не нашли прямого совпадения в FROM/JOIN
    logger.debug(f"Could not directly resolve alias/table '{source_table_alias}' (schema hint: {source_schema_alias}) in query sources: {[t.sql() for t in source_tables_list]}")

    # Fallback: Если схема в колонке НЕ была указана, и имя совпадает с именем таблицы (без алиаса) из FROM
    # Это может поймать случаи типа `SELECT t1.col FROM table1 t1 JOIN table2 ON ...` где в lineage приходит `table2.col`
    if source_schema_alias is None:
         for table_expr in source_tables_list:
             if not isinstance(table_expr, sqlglot.exp.Table): continue
             # Если имя таблицы совпадает и у нее нет алиаса
             if normalize_name(table_expr.name) == source_table_alias_norm and not table_expr.alias:
                 actual_table_name = table_expr.name
                 actual_schema = table_expr.db if table_expr.db else target_schema
                 logger.debug(f"Fallback resolution for '{source_table_alias}' -> {actual_schema}.{actual_table_name} (unaliased table match)")
                 return actual_schema, actual_table_name

    logger.warning(f"Failed to resolve alias/table '{source_table_alias}' (schema hint: {source_schema_alias})")
    return None, None


def build_dependency_graph(sql_files: List[Path], root_dir: Path) -> Tuple[nx.DiGraph, Set[str]]:
    """
    Строит граф зависимостей моделей и колонок.
    """
    graph = nx.DiGraph()
    known_models: Set[str] = set() # Set of 'tbl:schema.name' for models defined by files
    model_definitions: Dict[str, Dict[str, Any]] = {} # model_id -> {file_path, schema, table_name}

    # 1. Идентификация моделей по файлам
    logger.info("--- Фаза 1: Идентификация моделей ---")
    for file_path in sql_files:
        schema, table_name = extract_model_name_from_path(file_path, root_dir)
        if schema and table_name:
            try:
                model_id = format_node_id(TBL_PREFIX, schema, table_name)
                if model_id in model_definitions:
                     logger.warning(f"Дублирующееся определение для модели {model_id} найдено в {file_path.name} и {model_definitions[model_id]['file_path'].name}. Используется последнее: {file_path.name}")
                known_models.add(model_id)
                model_definitions[model_id] = {"file_path": file_path, "orig_schema": schema, "orig_table_name": table_name}
            except ValueError as e:
                logger.error(f"Ошибка форматирования ID для файла {file_path.name} ({schema}.{table_name}): {e}")
        else:
             logger.warning(f"Пропуск файла (не удалось определить модель): {file_path.name}")

    logger.info(f"Обнаружено {len(known_models)} моделей с SQL определениями.")

    # 2. Построение графа: узлы таблиц, колонок, зависимости таблиц и колонок
    logger.info("--- Фаза 2: Построение графа зависимостей ---")
    processed_files = 0
    total_files = len(model_definitions)
    for model_id, model_info in model_definitions.items():
        processed_files += 1
        file_path = model_info["file_path"]
        target_schema = model_info["orig_schema"]
        target_table = model_info["orig_table_name"]
        relative_path = str(file_path.relative_to(root_dir)) if root_dir in file_path.parents else str(file_path)

        logger.info(f"[{processed_files}/{total_files}] Анализ: {relative_path} (Модель: {target_schema}.{target_table})")

        try:
            sql_content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            continue

        # Находим основное выражение (INSERT/CREATE)
        main_statement = find_main_statement(sql_content, target_schema, target_table)
        if not main_statement:
            logger.warning(f"Пропуск {model_id}, не найдено основное выражение INSERT/CREATE.")
            continue

        # Добавляем узел для текущей модели (таблицы)
        try:
            parsed_model_id_info = parse_node_id(model_id)
            node_attrs = {
                "type": TBL_PREFIX,
                "schema": parsed_model_id_info['schema'],
                "name": parsed_model_id_info['table'],
                "source_type": 'model',
                "file_path": relative_path
            }
            graph.add_node(model_id, **node_attrs)
            logger.debug(f"Добавлен узел модели: {model_id}")
        except (ValueError, KeyError) as e:
            logger.error(f"Ошибка парсинга или добавления узла модели {model_id}: {e}")
            continue

        # --- Зависимости на уровне таблиц ---
        source_tables_in_query = []
        try:
            # Find all tables used, excluding CTEs defined within the main statement
            all_tables = list(main_statement.find_all(sqlglot.exp.Table))
            # Ищем CTE
            ctes_in_scope = set()
            with_scope = main_statement.find(sqlglot.exp.With)
            if with_scope:
                # Normalize CTE names for comparison
                ctes_in_scope = {normalize_name(cte.alias_or_name) for cte in with_scope.expressions if cte.alias_or_name}

            # Фильтруем таблицы, оставляя только те, что не являются CTE
            source_tables_in_query = [
                tbl for tbl in all_tables
                if normalize_name(tbl.name) not in ctes_in_scope
            ]
            if ctes_in_scope:
                logger.debug(f"Обнаружены CTE: {ctes_in_scope} в {model_id}")

        except Exception as e:
            logger.error(f"Ошибка поиска таблиц-источников в {file_path.name} для {model_id}: {e}")

        processed_upstream_models = set()
        target_schema_norm = normalize_name(target_schema)
        target_table_norm = normalize_name(target_table)

        for table_expr in source_tables_in_query:
            source_table_name = table_expr.name
            source_table_name_norm = normalize_name(source_table_name)

            # Схема из выражения или целевая
            source_schema = table_expr.db if table_expr.db else target_schema
            source_schema_norm = normalize_name(source_schema)

            # Игнорируем ссылку на саму себя
            if source_schema_norm == target_schema_norm and source_table_name_norm == target_table_norm:
                continue

            try:
                upstream_model_id = format_node_id(TBL_PREFIX, source_schema, source_table_name)
            except ValueError as e:
                logger.error(f"Ошибка форматирования ID для таблицы-источника {source_schema}.{source_table_name} в {model_id}: {e}")
                continue

            if upstream_model_id in processed_upstream_models:
                continue
            processed_upstream_models.add(upstream_model_id)

            # Добавляем узел источника
            if not graph.has_node(upstream_model_id):
                is_known = upstream_model_id in known_models
                source_type = 'model' if is_known else 'source_table'
                try:
                    parsed_upstream_id = parse_node_id(upstream_model_id)
                    up_attrs = {
                        "type": TBL_PREFIX,
                        "schema": parsed_upstream_id['schema'],
                        "name": parsed_upstream_id['table'],
                        "source_type": source_type
                    }
                    graph.add_node(upstream_model_id, **up_attrs)
                    logger.debug(f"Добавлен узел источника: {upstream_model_id} (Тип: {source_type})")
                except (ValueError, KeyError) as e:
                     logger.error(f"Ошибка добавления узла-источника {upstream_model_id}: {e}")
                     continue # Пропускаем ребро

            # Добавляем ребро зависимости
            if graph.has_node(upstream_model_id):
                 graph.add_edge(model_id, upstream_model_id, type='table_dependency')
                 logger.debug(f"Добавлена зависимость таблицы: {model_id} -> {upstream_model_id}")


        # --- Анализ зависимостей колонок ---
        target_columns: List[str] = []
        select_expressions: List[Expression] = []

        try:
            # Код извлечения target_columns и select_expressions остался прежним
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
                 logger.warning(f"Некоторые целевые колонки не являются строками или None и были пропущены в {model_id}")

        except Exception as e:
             logger.error(f"Ошибка извлечения целевых колонок/выражений для {model_id} из {file_path.name}: {e}", exc_info=True)
             target_columns = []
             select_expressions = []


        if not target_columns:
            logger.warning(f"Не удалось извлечь целевые колонки для {model_id}. Анализ lineage для колонок не будет выполнен.")
        else:
            logger.debug(f"Целевые колонки для {model_id} (перед lineage): {target_columns}")
            if select_expressions and len(target_columns) != len(select_expressions):
                 logger.warning(f"Число целевых колонок ({len(target_columns)}) не совпадает с числом выражений в SELECT ({len(select_expressions)}) для {model_id}. Lineage может быть неточным.")

            parsed_target_id_info = parse_node_id(model_id)

            # Обрабатываем каждую целевую колонку
            for i, col_name in enumerate(target_columns):
                # col_name - оригинальное имя из SQL
                target_col_sql_expression: Optional[Expression] = None
                if i < len(select_expressions):
                     target_col_sql_expression = select_expressions[i]

                try:
                    target_col_id = format_node_id(COL_PREFIX, parsed_target_id_info['schema'], parsed_target_id_info['table'], col_name)
                    col_name_norm = normalize_name(col_name)

                    # Добавляем узел колонки цели
                    col_attrs = {
                        "type": COL_PREFIX,
                        "schema": parsed_target_id_info['schema'],
                        "table": parsed_target_id_info['table'],
                        "column": col_name_norm
                    }
                    graph.add_node(target_col_id, **col_attrs)
                    graph.add_edge(model_id, target_col_id, type='contains_column')
                    logger.debug(f"Добавлен узел/ребро для колонки: {target_col_id}")

                    # --- Lineage Analysis ---
                    lineage_target_column = col_name # Используем имя колонки как цель для lineage
                    lineage_sql_statement = main_statement # Весь INSERT или CREATE AS SELECT

                    # Получаем корневой узел дерева lineage для целевой колонки
                    lineage_result_node: Optional[LineageNode] = None
                    try:
                        lineage_result_node = sqlglot_lineage(
                            column=lineage_target_column,
                            sql=lineage_sql_statement,
                            dialect=getattr(config, 'SQL_DIALECT', None),
                            # schema=... # Можно передать схему для лучшего разрешения
                        )
                    except NotImplementedError as nie:
                        logger.warning(f"Lineage не реализован для части выражения, связанного с колонкой '{col_name}' в {model_id}. Ошибка: {nie}")
                    except KeyError as ke:
                         # Часто возникает, если sqlglot не может разрешить имя колонки/таблицы
                         logger.warning(f"Ошибка KeyError при lineage для колонки '{col_name}' в {model_id}: {ke}. Возможно, не удалось разрешить имя.")
                    except Exception as e:
                         logger.error(f"Ошибка выполнения sqlglot.lineage для колонки '{col_name}' в {model_id}: {e}", exc_info=False) # Убрал traceback по умолчанию

                    if lineage_result_node:
                        # Используем обновленную функцию для поиска колонок-источников в дереве lineage
                        source_column_expressions: List[sqlglot.exp.Column] = find_source_columns_from_lineage(
                            lineage_result_node,
                            lineage_target_column, # Имя целевой колонки для сравнения
                            getattr(config, 'SQL_DIALECT', None)
                        )

                        processed_source_col_ids = set()
                        # Обрабатываем найденные колонки-источники
                        for source_expr in source_column_expressions:
                            source_col_name = source_expr.name
                            source_table_alias = source_expr.table # Имя таблицы/алиаса из выражения колонки
                            source_schema_alias = source_expr.db   # Имя схемы из выражения колонки

                            # Разрешаем алиас/имя таблицы к реальной таблице/схеме
                            # Передаем итератор source_tables_in_query (таблицы из FROM/JOIN без CTE)
                            actual_source_schema, actual_source_table_name = resolve_table_alias(
                                source_table_alias,
                                source_schema_alias,
                                iter(source_tables_in_query),
                                target_schema # Схема целевой таблицы как дефолтная
                            )

                            if actual_source_table_name and actual_source_schema:
                                try:
                                    source_col_id = format_node_id(COL_PREFIX, actual_source_schema, actual_source_table_name, source_col_name)

                                    if source_col_id in processed_source_col_ids:
                                        continue
                                    processed_source_col_ids.add(source_col_id)

                                    source_col_name_norm = normalize_name(source_col_name)
                                    parsed_source_table_id_info = parse_node_id(format_node_id(TBL_PREFIX, actual_source_schema, actual_source_table_name))

                                    # Добавляем узел колонки-источника
                                    if not graph.has_node(source_col_id):
                                        src_col_attrs = {
                                            "type": COL_PREFIX,
                                            "schema": parsed_source_table_id_info['schema'],
                                            "table": parsed_source_table_id_info['table'],
                                            "column": source_col_name_norm
                                        }
                                        graph.add_node(source_col_id, **src_col_attrs)
                                        logger.debug(f"Добавлен узел колонки источника: {source_col_id}")

                                        # Ребро от таблицы-источника к её колонке
                                        upstream_model_id = format_node_id(TBL_PREFIX, parsed_source_table_id_info['schema'], parsed_source_table_id_info['table'])
                                        if not graph.has_node(upstream_model_id):
                                            # Добавляем узел таблицы, если пропустили
                                            is_known = upstream_model_id in known_models
                                            source_type = 'model' if is_known else 'source_table'
                                            up_attrs = {
                                                "type": TBL_PREFIX,
                                                "schema": parsed_source_table_id_info['schema'],
                                                "name": parsed_source_table_id_info['table'],
                                                "source_type": source_type
                                            }
                                            graph.add_node(upstream_model_id, **up_attrs)
                                            logger.debug(f"Добавлен недостающий узел источника: {upstream_model_id} (Тип: {source_type})")

                                        if graph.has_node(upstream_model_id) and not graph.has_edge(upstream_model_id, source_col_id):
                                            graph.add_edge(upstream_model_id, source_col_id, type='contains_column')
                                            logger.debug(f"Добавлено ребро содержит_колонку: {upstream_model_id} -> {source_col_id}")

                                    # Добавляем ребро зависимости колонок: target_col -> source_col
                                    if graph.has_node(source_col_id):
                                         graph.add_edge(target_col_id, source_col_id, type='column_dependency')
                                         logger.debug(f"Добавлена зависимость колонок: {target_col_id} -> {source_col_id}")
                                    else:
                                         logger.warning(f"Не удалось добавить ребро {target_col_id} -> {source_col_id}, узел источника не найден.")

                                except (ValueError, KeyError) as e:
                                     logger.error(f"Ошибка обработки/добавления узла/ребра для колонки-источника {actual_source_schema}.{actual_source_table_name}.{source_col_name}: {e}")
                            else:
                                logger.warning(f"Не удалось разрешить источник для колонки '{source_expr.sql()}' (из lineage для '{col_name}') в {model_id}. Исходный алиас/таблица: '{source_table_alias}', схема: '{source_schema_alias}'.")
                        # Конец цикла по source_column_expressions
                    else:
                        logger.debug(f"Lineage не вернул результат для колонки '{col_name}' в {model_id}.")

                except (ValueError, KeyError) as e:
                     logger.error(f"Ошибка обработки целевой колонки '{col_name}' или ее зависимостей в {model_id}: {e}")
                except Exception as e:
                     logger.error(f"Неожиданная ошибка при анализе lineage для колонки '{col_name}' в {model_id}: {e}", exc_info=True)
            # Конец цикла for col_name in target_columns

    logger.info(f"Построение графа завершено. Узлов: {graph.number_of_nodes()}, Ребер: {graph.number_of_edges()}")
    return graph, known_models

# --- Функции сохранения/загрузки и сравнения графов ---
def save_graph_state(graph: nx.DiGraph, state_file: Path):
    """Сохраняет граф в JSON файл."""
    logger.info(f"Сохранение состояния графа в {state_file}...")
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
                    logger.debug(f"Конвертация атрибута узла '{node}' [{key}]: {type(value)} -> str")
                    data[key] = str(value)
        for u, v, data in export_graph.edges(data=True):
             for key, value in list(data.items()):
                 if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    logger.debug(f"Конвертация атрибута ребра '({u}, {v})' [{key}]: {type(value)} -> str")
                    data[key] = str(value)

        graph_data = json_graph.node_link_data(export_graph)
        with state_file.open('w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
        logger.info("Состояние графа успешно сохранено.")
    except TypeError as te:
         logger.error(f"Ошибка сериализации графа в JSON: {te}.", exc_info=True)
    except Exception as e:
        logger.error(f"Не удалось сохранить состояние графа: {e}", exc_info=True)

def load_graph_state(state_file: Path) -> Optional[nx.DiGraph]:
    """Загружает граф из JSON файла."""
    state_file_path = Path(state_file) # Убедимся, что это Path объект
    if not state_file_path.exists():
        logger.info(f"Файл состояния {state_file_path} не найден. Предыдущее состояние отсутствует.")
        return None
    logger.info(f"Загрузка состояния графа из {state_file_path}...")
    try:
        with state_file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)
        # Use multigraph=False since we don't expect parallel edges of the same type
        graph = json_graph.node_link_graph(data, directed=True, multigraph=False)
        logger.info(f"Состояние графа успешно загружено. Узлов: {graph.number_of_nodes()}, Ребер: {graph.number_of_edges()}")
        return graph
    except json.JSONDecodeError as jde:
        logger.error(f"Ошибка декодирования JSON из файла {state_file_path}: {jde}")
        return None
    except Exception as e:
        logger.error(f"Не удалось загрузить состояние графа из {state_file_path}: {e}", exc_info=True)
        return None

def find_affected_downstream(graph: nx.DiGraph, changed_node_id: str) -> Dict[str, Set[str]]:
    """
    Находит все нижестоящие узлы (модели и колонки), зависящие от измененного узла.
    Использует обратный обход графа (от источника к потребителям).
    """
    affected: Dict[str, Set[str]] = {"tables": set(), "columns": set()}

    # Нормализуем ID для поиска в графе, если включена нормализация
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
                logger.debug(f"Нормализованный ID для поиска: {search_node_id} (из {changed_node_id})")
        except ValueError as e:
             logger.warning(f"Не удалось нормализовать ID '{changed_node_id}' для поиска: {e}. Поиск будет выполнен по оригинальному ID.")
             search_node_id = changed_node_id # Ищем как есть

    if not graph.has_node(search_node_id):
        logger.error(f"Узел '{search_node_id}' не найден в графе.")
        # Попробуем найти без нормализации, если она применялась
        if search_node_id != changed_node_id and graph.has_node(changed_node_id):
            logger.warning(f"Узел '{search_node_id}' не найден, но найден '{changed_node_id}'. Используем его.")
            search_node_id = changed_node_id
        else:
            return affected

    # nx.ancestors(graph, node) находит все узлы X, такие что есть путь X -> ... -> node.
    # Поскольку ребра у нас target -> source, ancestors(graph, source_node) найдет все target_nodes.
    downstream_dependents = nx.ancestors(graph, search_node_id)

    logger.info(f"Поиск зависимых от {search_node_id}...")
    for node_id in downstream_dependents:
        try:
            node_data = graph.nodes[node_id]
            node_type = node_data.get("type")
            if node_type == TBL_PREFIX:
                affected["tables"].add(node_id)
            elif node_type == COL_PREFIX:
                affected["columns"].add(node_id)
        except KeyError:
             logger.warning(f"Не найдены данные для узла {node_id} при поиске зависимых.")

    # Добавим сам измененный узел и его непосредственные компоненты
    try:
        parsed_search_id = parse_node_id(search_node_id)
        if parsed_search_id['type'] == COL_PREFIX:
            affected["columns"].add(search_node_id)
            # Также добавим таблицу, к которой относится измененная колонка
            table_id = format_node_id(TBL_PREFIX, parsed_search_id['schema'], parsed_search_id['table'])
            if graph.has_node(table_id):
                 affected["tables"].add(table_id)
        elif parsed_search_id['type'] == TBL_PREFIX:
             affected["tables"].add(search_node_id)
             # Если изменилась таблица, то все её колонки (в текущем графе) тоже затронуты
             # Ищем через predecessors, т.к. ребра table -> column
             for predecessor_id in graph.predecessors(search_node_id):
                  edge_data = graph.get_edge_data(predecessor_id, search_node_id)
                  # ОШИБКА ЛОГИКИ ЗДЕСЬ: ребра от таблицы к колонке идут contains_column(table_id, col_id)
                  # Значит надо искать successors(table_id)
             for successor_id in graph.successors(search_node_id):
                 edge_data = graph.get_edge_data(search_node_id, successor_id)
                 if edge_data and edge_data.get("type") == 'contains_column':
                      if graph.nodes[successor_id].get("type") == COL_PREFIX:
                           affected["columns"].add(successor_id)

    except (ValueError, KeyError) as e:
        logger.warning(f"Ошибка при самодобавлении/добавлении компонентов узла {search_node_id} в affected: {e}")


    logger.info(f"Найдено {len(affected['tables'])} зависимых таблиц и {len(affected['columns'])} зависимых колонок для '{search_node_id}'.")
    return affected

def find_significat_changes(previous_graph: nx.DiGraph, current_graph: nx.DiGraph):
    # 3. Сравниваем состояния и анализируем ИМПАКТ УДАЛЕНИЙ/ИЗМЕНЕНИЙ
    if previous_graph:
        logger.info("-" * 30)
        logger.info("Сравнение с предыдущим состоянием:")

        current_nodes = set(current_graph.nodes)
        previous_nodes = set(previous_graph.nodes)
        added_nodes = current_nodes - previous_nodes
        removed_nodes = previous_nodes - current_nodes

        current_edges = set(current_graph.edges)
        previous_edges = set(previous_graph.edges)
        added_edges = current_edges - previous_edges
        removed_edges = previous_edges - current_edges

        # --- Базовый отчет об изменениях ---
        has_changes = added_nodes or removed_nodes or added_edges or removed_edges
        if has_changes:
             if added_nodes:
                 logger.info(f"Добавлено узлов ({len(added_nodes)}): {sorted(list(added_nodes))}")
             if removed_nodes:
                 logger.info(f"Удалено узлов ({len(removed_nodes)}): {sorted(list(removed_nodes))}")
             if added_edges:
                  logger.info(f"Добавлено зависимостей ({len(added_edges)}): {sorted([(u,v) for u,v in added_edges])}")
             if removed_edges:
                  logger.info(f"Удалено зависимостей ({len(removed_edges)}): {sorted([(u,v) for u,v in removed_edges])}")
        else:
            logger.info("Структурных изменений (узлы, ребра) не обнаружено.")
            save_graph_state(current_graph, config.STATE_FILE)
            logger.info("Анализ зависимостей завершен.")
            return

        # --- Анализ влияния удаленных элементов ---
        impacted_by_removal_tables = set()
        impacted_by_removal_columns = set()
        directly_affected_targets = set() # Узлы в ТЕКУЩЕМ графе, чьи зависимости ИСЧЕЗЛИ

        # Ищем цели (predecessors), которые в previous_graph указывали на удаленные узлы
        for removed_node_id in removed_nodes:
            if removed_node_id in previous_graph:
                 # Ищем узлы, которые *ранее* зависели от удаленного узла
                 for predecessor_id in previous_graph.predecessors(removed_node_id):
                     # Если этот зависящий узел все еще существует
                     if predecessor_id in current_graph:
                         edge_data = previous_graph.get_edge_data(predecessor_id, removed_node_id)
                         edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
                         logger.debug(f"Узел {predecessor_id} (существует) ранее зависел ({edge_type}) от удаленного узла {removed_node_id}.")
                         directly_affected_targets.add(predecessor_id)

        # Ищем цели (u), чьи ребра (u, v) были удалены, но оба узла u и v существуют
        for u, v in removed_edges:
            if u in current_graph and v in current_graph:
                 if previous_graph.has_edge(u, v): # Убедимся, что ребро действительно было
                      edge_data = previous_graph.get_edge_data(u, v)
                      edge_type = edge_data.get('type', 'unknown') if edge_data else 'unknown'
                      logger.debug(f"Зависимость ({edge_type}) узла {u} от {v} была удалена (оба узла существуют).")
                      directly_affected_targets.add(u) # Узел 'u' затронут изменением

        # Теперь находим полный downstream impact для directly_affected_targets в ТЕКУЩЕМ графе
        if directly_affected_targets:
            logger.info("--- Анализ влияния удаленных/измененных зависимостей ---")
            final_impacted_nodes = set() # Собираем ВСЕ затронутые узлы (прямо и косвенно)

            for target_id in directly_affected_targets:
                 # Добавляем сам узел, чья зависимость изменилась
                 final_impacted_nodes.add(target_id)

                 # Ищем всех, кто зависит от этого измененного узла в ТЕКУЩЕМ графе
                 # Используем nx.ancestors, т.к. ребра target -> source
                 try:
                      if current_graph.has_node(target_id):
                          downstream_dependents = nx.ancestors(current_graph, target_id)
                          logger.debug(f"Поиск downstream от {target_id}: {downstream_dependents}")
                          final_impacted_nodes.update(downstream_dependents)
                      else:
                          # Этого не должно происходить по логике выше, но на всякий случай
                          logger.warning(f"Узел {target_id}, помеченный как directly affected, отсутствует в текущем графе при поиске downstream.")

                 except nx.NetworkXError as ne:
                      logger.error(f"Ошибка при поиске зависимых от {target_id} в текущем графе: {ne}")


            # Категоризируем затронутые узлы
            for node_id in final_impacted_nodes:
                # Берем данные из текущего графа, если узел существует
                if node_id in current_graph:
                    try:
                        node_data = current_graph.nodes[node_id]
                        node_type = node_data.get("type")
                        if node_type == TBL_PREFIX:
                            impacted_by_removal_tables.add(node_id)
                        elif node_type == COL_PREFIX:
                            impacted_by_removal_columns.add(node_id)
                    except KeyError:
                        logger.warning(f"Не найдены данные для существующего узла {node_id} в текущем графе.")
                else:
                    # Если узел был затронут, но теперь удален (например, target_id сам был удален каскадно)
                    # Попробуем взять данные из предыдущего графа
                    if node_id in previous_graph:
                         try:
                            node_data = previous_graph.nodes[node_id]
                            node_type = node_data.get("type")
                            if node_type == TBL_PREFIX:
                                impacted_by_removal_tables.add(f"{node_id} (удален)")
                            elif node_type == COL_PREFIX:
                                impacted_by_removal_columns.add(f"{node_id} (удален)")
                         except KeyError:
                             logger.warning(f"Не найдены данные для удаленного узла {node_id} в предыдущем графе.")
                    else:
                        # Совсем странный случай
                        logger.warning(f"Затронутый узел {node_id} не найден ни в текущем, ни в предыдущем графе.")


            if impacted_by_removal_tables or impacted_by_removal_columns:
                logger.info("Обнаружено влияние на следующие объекты из-за удаленных/измененных зависимостей:")
                if impacted_by_removal_tables:
                    logger.info(f"  Затронутые таблицы ({len(impacted_by_removal_tables)}): {sorted(list(impacted_by_removal_tables))}")
                if impacted_by_removal_columns:
                    logger.info(f"  Затронутые колонки ({len(impacted_by_removal_columns)}): {sorted(list(impacted_by_removal_columns))}")
            else:
                 # Если directly_affected_targets был, но итоговый импакт пуст (например, зависели только удаленные узлы)
                 if directly_affected_targets:
                     logger.info("Изменения зависимостей затронули только удаленные объекты.")
                 else: # Эта ветка не должна достигаться, если has_changes=True и directly_affected_targets пуст
                      logger.info("Удаленные/измененные узлы/ребра не привели к разрыву известных зависимостей у существующих объектов.")

        logger.info("-" * 30)

    else:
        logger.info("Предыдущее состояние не найдено, сравнение и анализ влияния не выполняются.")

def main():
    logger.info("="*50)
    logger.info("Запуск анализатора зависимостей SQL моделей...")
    logger.info(f"Используется директория моделей: {config.SQL_MODELS_DIR}")
    logger.info(f"Файл состояния: {config.STATE_FILE}")
    logger.info(f"Нормализация имен: {NORMALIZE_NAMES}")
    logger.info(f"SQL диалект: {getattr(config, 'SQL_DIALECT', 'Не указан')}")
    logger.info("="*50)

    if not config.SQL_MODELS_DIR.is_dir():
        logger.error(f"Папка с SQL моделями не найдена: {config.SQL_MODELS_DIR}")
        logger.error("Проверьте путь в config.py и убедитесь, что папка существует.")
        return

    sql_files = find_sql_files(config.SQL_MODELS_DIR)
    if not sql_files:
        logger.warning("Не найдено SQL файлов для анализа.")
        if not Path(config.STATE_FILE).exists():
            logger.info("Создание пустого файла состояния.")
            save_graph_state(nx.DiGraph(), Path(config.STATE_FILE))
        logger.info("Завершение работы.")
        return

    # 1. Загружаем предыдущее состояние
    previous_graph = load_graph_state(config.STATE_FILE)

    # 2. Строим текущее состояние
    current_graph, _ = build_dependency_graph(sql_files, config.SQL_MODELS_DIR)

    find_significat_changes(previous_graph, current_graph)

    # 4. Сохранить текущее состояние графа
    save_graph_state(current_graph, config.STATE_FILE)

    logger.info("Анализ зависимостей завершен.")

if __name__ == "__main__":
    main()
