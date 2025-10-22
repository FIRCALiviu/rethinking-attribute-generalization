import math
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Load your data ---
with open('./data/concepts-things.txt', 'r', encoding='utf-8') as f:
    decoder = f.read().split()

with open('probing_norms/similar_idxs.txt', 'r', encoding='utf-8') as f:
    pairs_raw = [int(x) for x in f.read().split()]

pairs = [(pairs_raw[2*i], pairs_raw[2*i+1]) for i in range(len(pairs_raw)//2)]
nodes = {decoder[i] for i in pairs_raw}
pairs = [(decoder[a], decoder[b]) for a, b in pairs]

# --- Build graph ---
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(pairs)

# --- Beautiful connected-component visualization ---
def draw_components_tiled(
    G, 
    k=0.6, 
    padding=0.12, 
    cols=None, 
    seed=42,
    node_size=240, 
    with_labels=True,
    edge_width=1.6,
    edge_alpha=0.6,
    edge_color="#7f7f7f",
    label_bbox=True,
    savepath='cretin.png'
):
    """
    Draw a graph with connected components laid out separately and placed in a tidy grid.
    Colors are assigned so that adjacent clusters have different colors.
    """
    comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    comps.sort(key=lambda H: H.number_of_nodes(), reverse=True)
    n = len(comps)
    if n == 0:
        return

    if cols is None:
        cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    # Meta-graph to color adjacent grid cells differently
    meta_graph = nx.Graph()
    meta_graph.add_nodes_from(range(n))
    for i in range(n):
        r, c = divmod(i, cols)
        if c + 1 < cols and i + 1 < n:
            meta_graph.add_edge(i, i + 1)
        if r + 1 < rows and i + cols < n:
            meta_graph.add_edge(i, i + cols)

    # Greedy color on the meta grid with a soft palette
    colors = {}
    color_map = plt.cm.get_cmap('Set3', 12)
    for node in meta_graph.nodes():
        neighbor_colors = {colors[nb] for nb in meta_graph.neighbors(node) if nb in colors}
        for color_index in range(12):
            if color_index not in neighbor_colors:
                colors[node] = color_index
                break

    # Local layouts (normalize to [0,1]x[0,1] for each component)
    local_positions = []
    for H in comps:
        # Kamada-Kawai often yields straighter, more even edges for small comps
        if H.number_of_nodes() <= 30:
            pos = nx.kamada_kawai_layout(H)
        else:
            k_eff = k * (0.75 if H.number_of_nodes() <= 2 else 1.0)
            pos = nx.spring_layout(H, k=k_eff, seed=seed)
        xs, ys = zip(*pos.values())
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        dx = (maxx - minx) or 1.0
        dy = (maxy - miny) or 1.0
        pos = {u: ((p[0]-minx)/dx, (p[1]-miny)/dy) for u, p in pos.items()}
        local_positions.append(pos)

    # Place components on a tidy grid
    global_pos = {}
    cell_w, cell_h = 1.0, 1.0
    pad = padding
    scale_w = (1 - 2*pad) * cell_w
    scale_h = (1 - 2*pad) * cell_h

    for i, (H, pos) in enumerate(zip(comps, local_positions)):
        r, c = divmod(i, cols)
        ox, oy = c*cell_w, (rows-1-r)*cell_h
        for u, (x, y) in pos.items():
            X = ox + pad*cell_w + x*scale_w
            Y = oy + pad*cell_h + y*scale_h
            global_pos[u] = (X, Y)

    # --- Drawing ---
    fig_w = min(16, 4*cols)
    fig_h = min(12, 3*rows)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    ax.set_axis_off()
    ax.set_aspect('equal')

    # Edges first (rounded caps/joins for smoother-looking lines)
    edge_collection = nx.draw_networkx_edges(
        G, global_pos, ax=ax,
        width=edge_width, alpha=edge_alpha, edge_color=edge_color
    )
    if isinstance(edge_collection, mpl.collections.LineCollection):
        edge_collection.set_capstyle('round')
        edge_collection.set_joinstyle('round')

    # Nodes by component color
    for i, H in enumerate(comps):
        comp_color = color_map(colors[i])
        nx.draw_networkx_nodes(
            H, global_pos, ax=ax,
            node_size=node_size,
            linewidths=0.8,
            edgecolors='white',
            node_color=[comp_color]*H.number_of_nodes()
        )

    # Labels (optional white box to keep lines from muddying text)
    if with_labels:
        label_kwargs = dict(font_size=12, font_weight='medium')
        if label_bbox:
            label_kwargs["bbox"] = dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2)
        nx.draw_networkx_labels(G, global_pos, ax=ax, **label_kwargs)

    plt.margins(0.01)
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)

# --- Run visualization ---
draw_components_tiled(G, k=0.6, padding=0.15, with_labels=True)
