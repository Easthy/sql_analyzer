from sql_analyzer import NameUtils, TBL_PREFIX, COL_PREFIX

def test_format_node_id_table():
    NameUtils._normalize_enabled = True
    assert NameUtils.format_node_id(TBL_PREFIX, "SchemaA", "TableB") == "tbl:schemaa.tableb"

def test_format_node_id_column():
    NameUtils._normalize_enabled = True
    assert NameUtils.format_node_id(COL_PREFIX, "SchemaA", "TableB", "ColumnC") == "col:schemaa.tableb.columnc"

def test_parse_node_id():
    node_id = "col:my_schema.my_table.my_column"
    expected = {"type": "col", "schema": "my_schema", "table": "my_table", "column": "my_column"}
    assert NameUtils.parse_node_id(node_id) == expected