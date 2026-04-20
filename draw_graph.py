import json
import math
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Configuration ---
TABLE_NODE_COLOR = '#9ecae1'
COLUMN_NODE_COLOR = '#c7e9c0'
TABLE_NODE_SIZE = 1100
COLUMN_NODE_SIZE = 90
TABLE_EDGE_COLOR = '#1f3a93'
COLUMN_EDGE_COLOR = '#9ab0a8'
COLUMN_DEP_EDGE_COLOR = '#ff7f0e'
OTHER_EDGE_COLOR = '#d62728'

TABLE_FONT_SIZE = 9
COLUMN_FONT_SIZE = 3.5

LAYOUT_ENGINE = 'sfdp'

# Минимальная дуга между соседними колонками на кольце (в единицах позиций).
# Колонки распределяются по нескольким кольцам так, чтобы эта дуга соблюдалась.
COLUMN_ARC_SPACING = 0.18
# Базовый радиус первого кольца колонок (доля от среднего расстояния таблица-таблица)
COLUMN_BASE_RADIUS_FRAC = 0.18
# Шаг между соседними кольцами колонок
COLUMN_RING_STEP = 0.11
# Потолок радиуса облака (доля от расстояния таблица-таблица, чтобы не налезать на соседей)
COLUMN_MAX_RADIUS_FRAC = 0.48
# Насколько сдвигать подпись наружу от узла колонки (доля от радиуса её кольца)
COLUMN_LABEL_OUTSET = 0.06

FIGURE_SIZE = (70, 70)
DPI = 200

TABLE_ARROW_SIZE = 20
COLUMN_ARROW_SIZE = 6
COLUMN_DEP_ARROW_SIZE = 12
# ---


def graphviz_layout(graph: nx.Graph, weights: dict, engine: str = 'sfdp'):
    """Layout через pydot+graphviz. weights[node] — относительный размер узла.

    Размер передаётся через width/height, чтобы sfdp разносил «толстые» таблицы дальше.
    """
    import pydot

    dot = pydot.Dot(graph_type='graph', overlap='prism', splines='false')
    dot.set('sep', '+25')
    dot.set('K', '1.3')
    dot.set('repulsiveforce', '2.5')

    node_id_map = {n: f'n{i}' for i, n in enumerate(graph.nodes())}
    for node, nid in node_id_map.items():
        w = weights.get(node, 1.0)
        dot.add_node(pydot.Node(
            nid, label='""', shape='circle', fixedsize='true',
            width=str(w), height=str(w),
        ))
    for u, v in graph.edges():
        dot.add_edge(pydot.Edge(node_id_map[u], node_id_map[v]))

    plain = dot.create(prog=engine, format='plain').decode()
    pos = {}
    inv_map = {v: k for k, v in node_id_map.items()}
    for line in plain.splitlines():
        parts = line.split()
        if parts and parts[0] == 'node' and len(parts) >= 4:
            nid = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            if nid in inv_map:
                pos[inv_map[nid]] = np.array([x, y])
    return pos


print('Loading dependency_state.json ...')
with open('dependency_state.json') as f:
    data = json.load(f)

G = json_graph.node_link_graph(data, directed=True, multigraph=False, edges='links')
print(f'Total nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}')

table_nodes = {n for n, a in G.nodes(data=True) if a.get('type') == 'tbl'}
column_nodes = {n for n, a in G.nodes(data=True) if a.get('type') == 'col'}
print(f'Tables: {len(table_nodes)}, Columns: {len(column_nodes)}')

# Колонки каждой таблицы
table_columns = defaultdict(list)
col_to_tbl = {}
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'contains_column' and u in table_nodes and v in column_nodes:
        table_columns[u].append(v)
        col_to_tbl[v] = u

# --- Step 1: layout таблиц с учётом их «размера» (числа колонок) ---
G_tables_layout = nx.Graph()
G_tables_layout.add_nodes_from(table_nodes)
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'table_dependency' and u in table_nodes and v in table_nodes:
        G_tables_layout.add_edge(u, v)
for u, v, attrs in G.edges(data=True):
    if attrs.get('type') == 'column_dependency':
        tu, tv = col_to_tbl.get(u), col_to_tbl.get(v)
        if tu and tv and tu != tv:
            G_tables_layout.add_edge(tu, tv)

# width в sfdp: от 1 до ~5 дюймов в зависимости от числа колонок
max_cols = max((len(c) for c in table_columns.values()), default=1)
weights = {}
for t in table_nodes:
    n = len(table_columns.get(t, []))
    weights[t] = 1.0 + 4.0 * math.sqrt(n) / math.sqrt(max_cols)

print(f'Computing layout for {G_tables_layout.number_of_nodes()} tables '
      f'(max cols/table = {max_cols}) using graphviz {LAYOUT_ENGINE} ...')
try:
    pos_tables = graphviz_layout(G_tables_layout, weights, engine=LAYOUT_ENGINE)
except Exception as exc:
    print(f'Graphviz layout failed ({exc}), falling back to spring_layout')
    pos_tables = nx.spring_layout(
        G_tables_layout,
        k=3.0 / math.sqrt(max(1, G_tables_layout.number_of_nodes())),
        iterations=300,
        seed=50,
    )

# Нормируем/масштабируем позиции таблиц в диапазон ~[-100; 100]
if pos_tables:
    xs = np.array([p[0] for p in pos_tables.values()])
    ys = np.array([p[1] for p in pos_tables.values()])
    span = max(xs.max() - xs.min(), ys.max() - ys.min()) or 1.0
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    for k, (x, y) in pos_tables.items():
        pos_tables[k] = np.array([(x - cx) / span, (y - cy) / span]) * 200.0

# Среднее расстояние до ближайшей таблицы — калибровка радиуса колонок
if len(pos_tables) > 1:
    from scipy.spatial import cKDTree
    arr = np.array(list(pos_tables.values()))
    tree = cKDTree(arr)
    dists, _ = tree.query(arr, k=2)
    avg_nn = float(np.mean(dists[:, 1]))
else:
    avg_nn = 10.0
print(f'Avg nearest-neighbor table distance: {avg_nn:.3f}')

# --- Step 2: колонки — несколько колец вокруг таблицы ---
# Для каждого кольца подбираем радиус так, чтобы угловой шаг между
# колонками по дуге был не меньше COLUMN_ARC_SPACING. Если места не хватает —
# добавляем новое кольцо. Радиус ограничен COLUMN_MAX_RADIUS_FRAC * avg_nn.
pos = dict(pos_tables)
label_pos = {}  # отдельные позиции для подписей колонок — чуть наружу

base_r = COLUMN_BASE_RADIUS_FRAC * avg_nn
ring_step = COLUMN_RING_STEP * avg_nn
max_r = COLUMN_MAX_RADIUS_FRAC * avg_nn


def distribute_into_rings(num_cols: int):
    """Вернуть список (radius, count) так, чтобы все num_cols разместились."""
    rings = []
    remaining = num_cols
    idx = 0
    while remaining > 0:
        r = base_r + idx * ring_step
        if r > max_r:
            r = max_r
        capacity = max(3, int((2 * math.pi * r) / COLUMN_ARC_SPACING))
        take = min(capacity, remaining)
        rings.append((r, take))
        remaining -= take
        idx += 1
        if r >= max_r and remaining > 0:
            # Дальше растягивать некуда — сжимаем последнее кольцо
            last_r, last_take = rings[-1]
            rings[-1] = (last_r, last_take + remaining)
            remaining = 0
    return rings


for tbl, cols in table_columns.items():
    if tbl not in pos or not cols:
        continue
    tx, ty = pos[tbl]
    cols = sorted(cols, key=lambda c: G.nodes[c].get('column', ''))
    rings = distribute_into_rings(len(cols))
    col_iter = iter(cols)
    ring_idx = 0
    for radius, count in rings:
        # на каждом следующем кольце смещаем угол на половину шага,
        # чтобы колонки соседних колец не лежали на одном луче
        angle_shift = math.pi / max(count, 1) * ring_idx
        step = 2 * math.pi / max(count, 1)
        for i in range(count):
            col = next(col_iter)
            a = -math.pi / 2 + i * step + angle_shift
            cx = tx + radius * math.cos(a)
            cy = ty + radius * math.sin(a)
            pos[col] = np.array([cx, cy])
            outset = radius * COLUMN_LABEL_OUTSET
            label_pos[col] = np.array([tx + (radius + outset) * math.cos(a),
                                       ty + (radius + outset) * math.sin(a)])
        ring_idx += 1

# --- Step 3: подготовка отрисовки ---

def table_label(a):
    return f"{a.get('schema', '')}.{a.get('name', '')}"

def column_label(a):
    return a.get('column', '')

table_labels = {n: table_label(G.nodes[n]) for n in table_nodes}
column_labels = {n: column_label(G.nodes[n]) for n in column_nodes}

table_dep_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get('type') == 'table_dependency']
contains_col_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get('type') == 'contains_column']
col_dep_edges = [(u, v) for u, v, a in G.edges(data=True) if a.get('type') == 'column_dependency']
other_edges = [
    (u, v) for u, v, a in G.edges(data=True)
    if a.get('type') not in ('table_dependency', 'contains_column', 'column_dependency')
]


# --- Step 4: рисуем ---
print('Drawing graph...')
fig, ax = plt.subplots(figsize=FIGURE_SIZE)
ax.set_aspect('equal')
ax.axis('off')

# contains_column — фоновые
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edgelist=contains_col_edges,
    edge_color=COLUMN_EDGE_COLOR,
    width=0.25, alpha=0.25, arrows=False,
)

# column_dependency — с заметной кривизной, чтобы параллельные не сливались
if col_dep_edges:
    # Разбросаем rad по рёбрам между одной и той же парой таблиц, чтобы
    # пучки расходились веером
    pair_counter = defaultdict(int)
    for u, v in col_dep_edges:
        pair_counter[(col_to_tbl.get(u), col_to_tbl.get(v))] += 1

    pair_index = defaultdict(int)
    # Рисуем поштучно, чтобы разбрасывать rad
    for u, v in col_dep_edges:
        key = (col_to_tbl.get(u), col_to_tbl.get(v))
        total = pair_counter[key]
        k = pair_index[key]
        pair_index[key] += 1
        if total <= 1:
            rad = 0.12
        else:
            # rad от -0.35 до +0.35, распределяя по дуге
            rad = -0.35 + 0.7 * (k + 0.5) / total
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[(u, v)],
            edge_color=COLUMN_DEP_EDGE_COLOR,
            width=0.55, alpha=0.6,
            arrows=True, arrowstyle='-|>', arrowsize=COLUMN_DEP_ARROW_SIZE,
            connectionstyle=f'arc3,rad={rad:.3f}',
            node_size=COLUMN_NODE_SIZE,
        )

# table_dependency поверх
nx.draw_networkx_edges(
    G, pos, ax=ax,
    edgelist=table_dep_edges,
    edge_color=TABLE_EDGE_COLOR,
    width=2.0, alpha=0.85,
    arrows=True, arrowstyle='-|>', arrowsize=TABLE_ARROW_SIZE,
    connectionstyle='arc3,rad=0.08',
    node_size=TABLE_NODE_SIZE,
)

if other_edges:
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=other_edges,
        edge_color=OTHER_EDGE_COLOR,
        width=1.0, alpha=0.8,
        arrows=True, arrowstyle='-|>', arrowsize=12,
        connectionstyle='arc3,rad=0.1',
    )

# Узлы: колонки под таблицами
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    nodelist=list(column_nodes),
    node_color=COLUMN_NODE_COLOR,
    node_size=COLUMN_NODE_SIZE,
    edgecolors='#4a7a3a', linewidths=0.4,
)
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    nodelist=list(table_nodes),
    node_color=TABLE_NODE_COLOR,
    node_size=TABLE_NODE_SIZE,
    edgecolors='#1f3a93', linewidths=1.1,
)

# Подписи колонок — по позициям со сдвигом наружу
for col, (lx, ly) in label_pos.items():
    ax.text(
        lx, ly, column_labels[col],
        fontsize=COLUMN_FONT_SIZE,
        color='#204020',
        ha='center', va='center',
        zorder=5,
    )

# Подписи таблиц
nx.draw_networkx_labels(
    G, pos, ax=ax,
    labels=table_labels,
    font_size=TABLE_FONT_SIZE,
    font_weight='bold',
    font_color='#0b1e4a',
)

ax.set_title(f'Dependency Graph: {len(table_nodes)} tables, {len(column_nodes)} columns',
             fontsize=16)

xs = [p[0] for p in pos.values()]
ys = [p[1] for p in pos.values()]
pad_x = (max(xs) - min(xs)) * 0.02
pad_y = (max(ys) - min(ys)) * 0.02
ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

plt.savefig('graph_output.png', dpi=DPI, bbox_inches='tight', facecolor='white')
print('Saved as graph_output.png')
