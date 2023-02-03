import networkx as nx

from qaoa.expectations import maxcut_obj


class ObjectiveFunction:
    def __init__(self):
        self.cached_graph = None
        self.cached_cut_size = {}
        self.sign = 1

    def cut_size(self, x: str, graph: nx.Graph) -> float:
        if x in self.cached_cut_size.keys() and hash(str(graph)) == self.cached_graph:
            return self.sign * self.cached_cut_size.get(x)
        else:
            if hash(str(graph)) != self.cached_graph:
                self.cached_graph = hash(str(graph))
                self.cached_cut_size = {}
            c = maxcut_obj(x, graph)
            self.cached_cut_size[x] = c
            return self.sign * c

    def get_number_of_unique_evaluations(self):
        return len(self.cached_cut_size)