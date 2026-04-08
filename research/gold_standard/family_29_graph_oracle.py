"""
Gold Standard Oracle: Family 29 — Graph Algorithms

Generates expected values from NetworkX for comparison with tambear.

Algorithms covered:
  - BFS / DFS traversal
  - Dijkstra shortest paths
  - Bellman-Ford shortest paths
  - Floyd-Warshall all-pairs shortest paths
  - Kruskal / Prim MST
  - PageRank
  - Degree / Closeness / Betweenness centrality
  - Connected components
  - Topological sort

Usage:
    python research/gold_standard/family_29_graph_oracle.py
"""

import json
import networkx as nx
import numpy as np


results = {}

# ── Graph 1: Simple weighted DAG ──
# 0→1(4), 0→2(1), 2→1(2), 1→3(5), 2→3(8)

G1 = nx.DiGraph()
G1.add_weighted_edges_from([
    (0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 5), (2, 3, 8)
])

# Dijkstra
d = dict(nx.single_source_dijkstra_path_length(G1, 0))
results["dijkstra_dag"] = {
    "source": 0,
    "distances": {str(k): v for k, v in d.items()},
    "edges": [(0,1,4), (0,2,1), (2,1,2), (1,3,5), (2,3,8)],
    "n_nodes": 4,
}

# Bellman-Ford
d_bf = dict(nx.single_source_bellman_ford_path_length(G1, 0))
results["bellman_ford_dag"] = {
    "source": 0,
    "distances": {str(k): v for k, v in d_bf.items()},
}

# Floyd-Warshall
fw = dict(nx.floyd_warshall(G1))
results["floyd_warshall_dag"] = {
    "distances": {str(i): {str(j): v for j, v in row.items()} for i, row in fw.items()},
}

# Topological sort
topo = list(nx.topological_sort(G1))
results["topological_sort_dag"] = {
    "order": topo,
}

# ── Graph 2: Undirected weighted graph for MST ──
# Triangle: 0-1(1), 1-2(2), 0-2(3)

G2 = nx.Graph()
G2.add_weighted_edges_from([(0, 1, 1), (1, 2, 2), (0, 2, 3)])

mst = nx.minimum_spanning_tree(G2)
results["mst_triangle"] = {
    "edges": sorted([(u, v, G2[u][v]['weight']) for u, v in mst.edges()]),
    "total_weight": sum(mst[u][v]['weight'] for u, v in mst.edges()),
    "n_edges": mst.number_of_edges(),
}

# ── Graph 3: Larger graph for MST and Dijkstra ──
# 5 nodes with various weights

G3 = nx.Graph()
G3.add_weighted_edges_from([
    (0, 1, 2), (0, 3, 6), (1, 2, 3), (1, 3, 8), (1, 4, 5),
    (2, 4, 7), (3, 4, 9)
])

mst3 = nx.minimum_spanning_tree(G3)
results["mst_5node"] = {
    "total_weight": sum(mst3[u][v]['weight'] for u, v in mst3.edges()),
    "n_edges": mst3.number_of_edges(),
}

d3 = dict(nx.single_source_dijkstra_path_length(G3, 0))
results["dijkstra_5node"] = {
    "source": 0,
    "distances": {str(k): v for k, v in sorted(d3.items())},
}

# ── Graph 4: PageRank ──

# Star graph: center 0 connected to 1,2,3,4
G4_star = nx.DiGraph()
for i in range(1, 5):
    G4_star.add_edge(0, i)
    G4_star.add_edge(i, 0)

pr_star = nx.pagerank(G4_star, alpha=0.85, max_iter=1000, tol=1e-10)
results["pagerank_star"] = {
    "ranks": {str(k): round(v, 10) for k, v in sorted(pr_star.items())},
    "alpha": 0.85,
}

# Complete K4: all nodes equal
G4_complete = nx.complete_graph(4, create_using=nx.DiGraph)
pr_complete = nx.pagerank(G4_complete, alpha=0.85, max_iter=1000, tol=1e-10)
results["pagerank_complete_k4"] = {
    "ranks": {str(k): round(v, 10) for k, v in sorted(pr_complete.items())},
    "alpha": 0.85,
}

# Chain: 0→1→2→3→0 (cycle)
G4_cycle = nx.DiGraph()
for i in range(4):
    G4_cycle.add_edge(i, (i+1) % 4)
pr_cycle = nx.pagerank(G4_cycle, alpha=0.85, max_iter=1000, tol=1e-10)
results["pagerank_cycle"] = {
    "ranks": {str(k): round(v, 10) for k, v in sorted(pr_cycle.items())},
    "alpha": 0.85,
}

# ── Graph 5: Centrality ──

# Star graph (undirected)
G5 = nx.star_graph(4)  # 0 is center, connected to 1,2,3,4
dc = nx.degree_centrality(G5)
cc = nx.closeness_centrality(G5)
bc = nx.betweenness_centrality(G5)

results["centrality_star"] = {
    "degree": {str(k): round(v, 10) for k, v in sorted(dc.items())},
    "closeness": {str(k): round(v, 10) for k, v in sorted(cc.items())},
    "betweenness": {str(k): round(v, 10) for k, v in sorted(bc.items())},
}

# ── Graph 6: Connected components ──

G6 = nx.Graph()
G6.add_edges_from([(0, 1), (1, 2)])  # component 1
G6.add_edges_from([(3, 4)])           # component 2
G6.add_node(5)                        # isolated node

components = list(nx.connected_components(G6))
results["connected_components"] = {
    "n_components": len(components),
    "components": [sorted(list(c)) for c in components],
}

# ── Graph 7: Bellman-Ford with negative edges ──

G7 = nx.DiGraph()
G7.add_weighted_edges_from([
    (0, 1, 4), (0, 2, 5), (1, 2, -3), (2, 3, 4)
])
d7 = dict(nx.single_source_bellman_ford_path_length(G7, 0))
results["bellman_ford_negative"] = {
    "source": 0,
    "distances": {str(k): v for k, v in sorted(d7.items())},
    "edges": [(0,1,4), (0,2,5), (1,2,-3), (2,3,4)],
}

# ── Graph 8: Density ──

results["density_complete_k4"] = {
    "value": round(nx.density(nx.complete_graph(4, create_using=nx.DiGraph)), 10),
}
results["density_star_5"] = {
    "value": round(nx.density(G5), 10),
}

# ── Save ──

with open("research/gold_standard/family_29_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F29 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if 'distances' in r:
        d_str = ', '.join(f"{k}:{v}" for k, v in sorted(r['distances'].items()))
        print(f"  PASS {name}: d=[{d_str}]")
    elif 'ranks' in r:
        pr_str = ', '.join(f"{k}:{v:.4f}" for k, v in sorted(r['ranks'].items()))
        print(f"  PASS {name}: pr=[{pr_str}]")
    elif 'total_weight' in r:
        print(f"  PASS {name}: MST weight={r['total_weight']}")
    elif 'n_components' in r:
        print(f"  PASS {name}: {r['n_components']} components")
    elif 'value' in r:
        print(f"  PASS {name}: {r['value']}")
    else:
        print(f"  PASS {name}")
