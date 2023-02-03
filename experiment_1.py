import logging

import networkx as nx
import numpy as np

from qaoa.circuit_generation import create_qaoa_circ_parameterized
from graphs import draw_graph
from circuit_cutting import preprocess, util
from circuit_cutting.execute import Executor
from utils import mkdir

logger = logging.getLogger(__name__)


def run_experiment_1(graph, parameters, backend, evaluations=3, shots=10000, cut_shot_factor=2,
                     reduced=True, path=None, retrieve_interval: int = 0, retries: int = 0,
                     short_circuits=False):
    sub_graph_size = graph.number_of_nodes() // 2
    cuts = sum([1 if u < sub_graph_size <= v or v < sub_graph_size <= u else 0 for u, v in graph.edges])
    if reduced:
        shots_cut = cut_shot_factor * int(shots / (2 * (4 ** cuts)))
    else:
        shots_cut = cut_shot_factor * int(shots / (2 * (5 ** cuts)))
    mkdir(path)

    draw_graph(graph, f'{path}/graph')
    nx.write_adjlist(graph, f'{path}/graph.txt')

    e = Executor(backend, memory=True)
    circuit = create_qaoa_circ_parameterized(graph, 1)
    circuit_short = create_qaoa_circ_parameterized(graph, 1, cut_edges_at_the_end=True)
    sub_circuits, sub_circuits_info = preprocess.split(circuit, [sub_graph_size, sub_graph_size], reduced)

    keys, sub_circuit_list = util.dict_to_lists(sub_circuits)
    for i, params in enumerate(parameters):
        qc = circuit.bind_parameters(params)
        qc_short = circuit_short.bind_parameters(params)
        for j in range(evaluations):
            e.add(qc, name=str((params, j)), shots=shots)
            if short_circuits:
                e.add(qc_short, name=str((params, j)) + '#short', shots=shots)

    for i, params in enumerate(parameters):
        sub_circuit_list_with_params = [circ.bind_parameters(params) for circ in sub_circuit_list]
        for j in range(evaluations):
            e.add(sub_circuit_list_with_params, name=str((params, j)) + '#cut', shots=shots_cut)

    e.job_limit = np.infty
    e.execute(retrieve_interval=retrieve_interval, retries=retries)

    e.save_results(path)
