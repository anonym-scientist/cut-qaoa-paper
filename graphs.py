import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def link_graphs(g: nx.Graph, h: nx.Graph, n_links: int) -> nx.Graph:
    graph: nx.Graph = nx.disjoint_union(g, h)
    num_nodes_g = g.number_of_nodes()
    num_nodes_h = h.number_of_nodes()
    if n_links > num_nodes_g * num_nodes_h:
        raise ValueError('The number of links is too high')
    for _ in range(n_links):
        node_g = random.randint(0, num_nodes_g - 1)
        node_h = random.randint(0, num_nodes_h - 1) + num_nodes_g
        while graph.has_edge(node_g, node_h):
            node_g = random.randint(0, num_nodes_g - 1)
            node_h = random.randint(0, num_nodes_h - 1) + num_nodes_g
        graph.add_edge(node_g, node_h)

    for u, v in graph.edges():
        graph[u][v]['weight'] = 1

    return graph


def generate_graph(graph_generator, n_links):
    g = graph_generator()
    h = graph_generator()
    if isinstance(n_links, int):
        return link_graphs(g, h, n_links)
    else:
        return [link_graphs(g, h, i) for i in n_links]


def erdos_renyi_graph_generator(n, edge_prob=0.9):
    def _erdos_renyi_graph_generator():
        graph = nx.erdos_renyi_graph(n, edge_prob)
        while not nx.is_connected(graph):
            graph = nx.erdos_renyi_graph(n, edge_prob)
        return graph

    return _erdos_renyi_graph_generator


def gnm_graph_generator(n, m):
    def _gnm_graph_generator():
        graph = nx.dense_gnm_random_graph(n, m)
        while not nx.is_connected(graph):
            graph = nx.dense_gnm_random_graph(n, m)
        return graph

    return _gnm_graph_generator


def draw_graph(g: nx.Graph, path=None, node_color=None, show=True):
    # Generate plot of the Graph
    if node_color is None:
        colors = ['r' for node in g.nodes()]
    else:
        n = g.number_of_nodes()
        colors = ['r' if node_color[n - node - 1] == '1' else 'b' for node in g.nodes()]
    pos = nx.spring_layout(g)
    fig = plt.figure()
    nx.draw_networkx(g, node_color=colors, node_size=600, alpha=1, ax=fig.add_subplot(111), pos=pos)
    if path is not None:
        fig.savefig(path)
    if show:
        plt.show()
    else:
        plt.close()


def get_graph(nodes, edge_list):
    node_list = np.arange(0, nodes, 1)
    G = nx.Graph()
    G.add_nodes_from(node_list)
    G.add_weighted_edges_from(edge_list)
    return G


def draw_graph_from_file(path, image_path=None, show=True):
    p = Path(path)
    graph = nx.read_adjlist(p)
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    if image_path is None:
        image_path = p.parent / 'graph.pdf'
    draw_graph(graph, path=image_path, show=show)


def get_graph_from_file(path) -> nx.Graph:
    graph = nx.read_adjlist(path)
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    for e in graph.edges():
        graph[e[0]][e[1]]['weight'] = 1
    return graph


if __name__ == '__main__':
    # g = nx.erdos_renyi_graph(5, 0.9)
    # h = nx.erdos_renyi_graph(5, 0.9)
    # G = link_graphs(g, h, 2)

    G = generate_graph(lambda: nx.erdos_renyi_graph(10, 0.9), 4)

    draw_graph(G)
