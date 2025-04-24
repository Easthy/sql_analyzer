import json
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph
import warnings
import math
import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
TABLE_NODE_COLOR = 'lightblue'
COLUMN_NODE_COLOR = 'lightgreen'
TABLE_NODE_SIZE = 1000
COLUMN_NODE_SIZE = 300
TABLE_EDGE_COLOR = 'black' # Make table dependencies more prominent
COLUMN_EDGE_COLOR = 'gray'
FONT_SIZE = 7
COLUMN_OFFSET_RADIUS = 0.15
FIGURE_SIZE = (20, 15)
LAYOUT_K = 0.9
LAYOUT_ITERATIONS = 150
TABLE_ARROW_SIZE = 15 # Slightly larger arrows for table dependencies
COLUMN_ARROW_SIZE = 10
# ---
# Load the JSON data
# Use the JSON provided in the prompt directly as a string for reproducibility
# Загрузим твой JSON (можно из файла или строки)
with open("dependency_state.json") as f:
    data = json.load(f)
# Or load from file:
# with open("dependency_state.json") as f:
#     data = json.load(f)

# Create the full graph
G = json_graph.node_link_graph(data, directed=True, multigraph=False)

# Separate nodes by type
table_nodes = {node for node, attrs in G.nodes(data=True) if attrs['type'] == 'tbl'}
column_nodes = {node for node, attrs in G.nodes(data=True) if attrs['type'] == 'col'}

# Create a subgraph containing only tables and their dependencies FOR LAYOUT
G_tables_layout = nx.Graph() # Use undirected for layout purposes
for tbl_node in table_nodes:
    G_tables_layout.add_node(tbl_node)
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'table_dependency' and u in table_nodes and v in table_nodes:
        G_tables_layout.add_edge(u, v)

# --- Step 1: Calculate positions for tables only ---
print("Calculating table layout...")
pos_tables = nx.spring_layout(
    G_tables_layout, # Use the layout graph
    k=LAYOUT_K / math.sqrt(G_tables_layout.number_of_nodes()) if G_tables_layout.number_of_nodes() > 0 else LAYOUT_K,
    iterations=LAYOUT_ITERATIONS,
    seed=42
)
print("Table layout calculated.")

# --- Step 2: Calculate positions for columns relative to their tables ---
pos = pos_tables.copy()
table_columns = {tbl: [] for tbl in table_nodes}
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'contains_column' and u in table_nodes and v in column_nodes:
        table_columns[u].append(v)

print("Positioning columns...")
for table_node, columns in table_columns.items():
    if not columns or table_node not in pos: # Check if table has position
        continue
    table_pos = pos[table_node]
    num_columns = len(columns)
    angle_step = 2 * math.pi / num_columns if num_columns > 1 else 0
    start_angle = -math.pi / 2
    for i, col_node in enumerate(columns):
        angle = start_angle + i * angle_step
        col_x = table_pos[0] + COLUMN_OFFSET_RADIUS * math.cos(angle)
        col_y = table_pos[1] + COLUMN_OFFSET_RADIUS * math.sin(angle)
        pos[col_node] = np.array([col_x, col_y])
print("Columns positioned.")

# --- Step 3: Prepare for Drawing ---
node_colors = [TABLE_NODE_COLOR if node in table_nodes else COLUMN_NODE_COLOR for node in G.nodes()]
node_sizes = [TABLE_NODE_SIZE if node in table_nodes else COLUMN_NODE_SIZE for node in G.nodes()]

def get_short_name(node_id, attrs):
    if attrs['type'] == 'tbl':
        return f"{attrs.get('schema', '')}.{attrs.get('name', '')}"
    elif attrs['type'] == 'col':
        return f"{attrs.get('table', '')}.{attrs.get('column', '')}"
    return node_id

labels = {node: get_short_name(node, G.nodes[node]) for node in G.nodes()}

# Separate edges by type for drawing
table_dependency_edges = [
    (u, v) for u, v, attrs in G.edges(data=True)
    if attrs.get('type') == 'table_dependency'
]
contains_column_edges = [
    (u, v) for u, v, attrs in G.edges(data=True)
    if attrs.get('type') == 'contains_column'
]
other_edges = [
    (u, v) for u, v, attrs in G.edges(data=True)
    if attrs.get('type') not in ['table_dependency', 'contains_column']
]


# --- Step 4: Draw the Graph ---
print("Drawing graph...")
plt.figure(figsize=FIGURE_SIZE)

# Draw TABLE DEPENDENCY edges (with arrows)
nx.draw_networkx_edges(
    G, pos,
    edgelist=table_dependency_edges,
    edge_color=TABLE_EDGE_COLOR,
    width=1.5, # Slightly thicker
    alpha=0.8,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=TABLE_ARROW_SIZE, # Use specific arrow size
    connectionstyle='arc3,rad=0.1',
    node_size=node_sizes # Important for arrow placement relative to nodes
)

# Draw CONTAINS COLUMN edges (optional arrows, maybe thinner/lighter)
nx.draw_networkx_edges(
    G, pos,
    edgelist=contains_column_edges,
    edge_color=COLUMN_EDGE_COLOR,
    width=0.8, # Thinner
    alpha=0.6,
    arrows=True, # Keep arrows if desired, or set to False
    arrowstyle='-|>',
    arrowsize=COLUMN_ARROW_SIZE, # Use specific arrow size
    connectionstyle='arc3,rad=0.05', # Less curve perhaps
    node_size=node_sizes # Important for arrow placement
)

# Draw any OTHER edge types if they exist
if other_edges:
    nx.draw_networkx_edges(
        G, pos,
        edgelist=other_edges,
        edge_color='red', # Make them stand out if unexpected
        alpha=0.7,
        arrows=True,
        arrowstyle='-|>',
        arrowsize=12,
        node_size=node_sizes
    )


# Draw nodes (on top of edges)
nx.draw_networkx_nodes(
    G, pos,
    node_size=node_sizes,
    node_color=node_colors
)

# Draw labels
nx.draw_networkx_labels(
    G, pos,
    labels=labels,
    font_size=FONT_SIZE,
    font_weight='bold'
)

plt.title("Dependency Graph (Tables Spaced, Columns Clustered)")
plt.axis('off')
plt.tight_layout()
print("Saving graph...")
plt.savefig("graph_output.png", dpi=300, bbox_inches='tight')
print("Graph saved as graph_output.png")
# plt.show()