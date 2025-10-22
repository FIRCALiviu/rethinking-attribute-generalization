import networkx as nx 
import matplotlib.pyplot as plt
import community as community_louvain
G = nx.Graph()

decoder =open('./data/concepts-things.txt').read().split()

pairs = open('probing_norms/similar_idxs.txt')
pairs =[int(x) for x in pairs.read().split()]

pairs_decoupled = pairs.copy()
pairs = [(pairs[2*i],pairs[2*i+1]) for i in range(len(pairs)//2)]



nodes = set(map(lambda x: decoder[x], pairs_decoupled))
pairs = map(lambda  x: (decoder[x[0]],decoder[x[1]]),pairs)
G.add_nodes_from(nodes)
G.add_edges_from(pairs)



# Find communities
partition = community_louvain.best_partition(G)
clusters = set(partition.values())
num_clusters = len(clusters)

# Assign a color to each cluster
cmap = plt.cm.tab20
cluster_color_map = {cl: cmap(i % cmap.N) for i, cl in enumerate(clusters)}

# Choose layout
pos = nx.kamada_kawai_layout(G)

plt.figure(figsize=(40, 40))

# Node colors by cluster
node_colors = [cluster_color_map[partition[n]] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=600)

# **Edge colors by cluster of the source node**
edge_colors = [
    cluster_color_map[partition[edge[0]]]
    for edge in G.edges()
]
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, alpha=0.4, width=2)

# Labels (optional)
nx.draw_networkx_labels(G, pos, font_size=10)

plt.axis('off')
plt.tight_layout()
plt.savefig("gpt cluster.png", dpi=300)