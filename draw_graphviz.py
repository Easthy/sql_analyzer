import json
from graphviz import Digraph
from networkx.readwrite import json_graph
import warnings

# Suppress FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
TABLE_BORDER_COLOR = '#0d00ff'  # table border
COLUMN_BORDER_COLOR = '#0d00ff'
CLUSTER_BORDER_COLOR = '#00c4b4'
CLUSTER_FILL_COLOR = '#d6fff1:#c2ffea'
TABLE_NODE_COLOR = '#adffff'   # background table node
COLUMN_NODE_COLOR = '#bdff61'  # background column node
TABLE_NODE_SHAPE = 'Mrecord'
COLUMN_NODE_SHAPE = 'octagon'
FONT_SIZE = '10'
TABLE_EDGE_COLOR = '#0099f0'  # table to table
COLUMN_EDGE_COLOR = '#0d00ff'  # table to col
COLUMN_DEPENDENCY_COLOR = '#ff990a'  # col to col
OTHER_EDGE_COLOR = '#c40000'
RANKDIR = 'TB'

def clean_node_id(node_id):
    """Create valid Graphviz node IDs by replacing problematic characters"""
    if isinstance(node_id, dict):
        # Handle case when node_id is a dictionary (from node attributes)
        return node_id['id'].replace(':', '_').replace('.', '_')
    return str(node_id).replace(':', '_').replace('.', '_')


# Load and process the JSON data
with open("dependency_state.json") as f:
    data = json.load(f)

# Create the full graph with explicit edges parameter
G = json_graph.node_link_graph(data, directed=True, multigraph=False, edges='links')

# Create Graphviz digraph
dot = Digraph(
    format='png',
    engine='dot',
    graph_attr={
        'rankdir': RANKDIR,
        'splines': 'curved',
        'overlap': 'false',
        'fontsize': FONT_SIZE,
        'nodesep': '0.7',
        'ranksep': '1.1',
        'maxiter': '1000',
        'remincross': 'true'
    },
    node_attr={
        'fontname': 'Arial',
        'fontsize': FONT_SIZE,
    },
    edge_attr={
        'fontname': 'Arial',
        'fontsize': FONT_SIZE,
    }
)

# First pass: add all nodes with proper styling
for node in G.nodes():
    attrs = G.nodes[node]
    node_id = clean_node_id(node)

    if attrs['type'] == 'tbl':
        label = f"{attrs.get('schema', '')}.{attrs.get('name', '')}"
        dot.node(
            node_id,
            label=label,
            shape=TABLE_NODE_SHAPE,
            style='filled',
            fillcolor=TABLE_NODE_COLOR,
            color=TABLE_BORDER_COLOR,
            penwidth='1.7'
        )
    elif attrs['type'] == 'col':
        label = f"{attrs.get('column', '')}"
        dot.node(
            node_id,
            label=label,
            shape=COLUMN_NODE_SHAPE,
            style='filled',
            fillcolor=COLUMN_NODE_COLOR,
            color=COLUMN_BORDER_COLOR,
            penwidth='1.0'
        )

# Second pass: add edges with proper styling
for u, v, attrs in G.edges(data=True):
    u_id = clean_node_id(u)
    v_id = clean_node_id(v)

    if attrs.get('type') == 'table_dependency':
        dot.edge(
            u_id, v_id,
            color=TABLE_EDGE_COLOR,
            penwidth='1.5',
            arrowsize='1.0'
        )
    elif attrs.get('type') == 'contains_column':
        dot.edge(
            u_id, v_id,
            color=COLUMN_EDGE_COLOR,
            penwidth='1.0',
            arrowsize='0.8',
            style='dashed'
        )
    elif attrs.get('type') == 'column_dependency':
        dot.edge(
            u_id, v_id,
            color=COLUMN_DEPENDENCY_COLOR,
            penwidth='1.2',
            arrowsize='0.9',
            style='solid'
        )
    else:
        dot.edge(
            u_id, v_id,
            color=OTHER_EDGE_COLOR,
            penwidth='1.2',
            arrowsize='1.0',
            style='dotted'
        )

# Third pass: group columns with their tables using subgraphs
# Create a mapping of tables to their columns
table_columns = {}
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'contains_column':
        table_id = clean_node_id(u)
        col_id = clean_node_id(v)
        if table_id not in table_columns:
            table_columns[table_id] = []
        table_columns[table_id].append(col_id)

# Create clusters for each table with its columns
for table_id, columns in table_columns.items():
    with dot.subgraph(name=f'cluster_{table_id}') as c:
        c.attr(
            style='rounded,filled', 
            color=CLUSTER_BORDER_COLOR,
            fillcolor=CLUSTER_FILL_COLOR,
            gradientangle='90', 
            labeljust='l', 
            fontsize='9'
        )
        c.node(table_id)
        for col_id in columns:
            c.node(col_id)

# Render and save the graph
print("Rendering graph...")
dot.render('dependency_graph', cleanup=True, format='png')
print("Graph saved as dependency_graph.png")