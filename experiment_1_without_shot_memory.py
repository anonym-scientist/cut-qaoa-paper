import argparse
import logging
import sys
import time

import networkx as nx
import numpy as np
import qiskit

from experiment_utils import get_parameters
from json_utils import store_kwargs_as_json
from provider_handler import get_backend_and_provider
from qaoa.circuit_generation import create_qaoa_circ_parameterized
from graphs import draw_graph, generate_graph, erdos_renyi_graph_generator
from circuit_cutting import preprocess, util
from circuit_cutting.execute import Executor
from utils import mkdir

logger = logging.getLogger(__name__)

DIR_PATH = 'experiment_param_map'


def run_experiment_1_without_shot_memory(graph, parameters, backend, evaluations=3, shots=10000, cut_shot_factor=2,
                                         reduced=True, path=None, retrieve_interval: int = 0, retries: int = 0):
    sub_graph_size = graph.number_of_nodes() // 2
    cuts = sum([1 if u < sub_graph_size <= v or v < sub_graph_size <= u else 0 for u, v in graph.edges])
    if reduced:
        shots_cut = cut_shot_factor * int(shots / (2 * (4 ** cuts)))
    else:
        shots_cut = cut_shot_factor * int(shots / (2 * (5 ** cuts)))
    mkdir(path)

    draw_graph(graph, f'{path}/graph')
    nx.write_adjlist(graph, f'{path}/graph.txt')

    e = Executor(backend)
    circuit = create_qaoa_circ_parameterized(graph, 1)
    sub_circuits, sub_circuits_info = preprocess.split(circuit, [sub_graph_size, sub_graph_size], reduced)

    keys, sub_circuit_list = util.dict_to_lists(sub_circuits)
    for i, params in enumerate(parameters):
        qc = circuit.bind_parameters(params)
        for j in range(evaluations):
            e.add(qc, name=str((params, j)), shots=shots)

    for i, params in enumerate(parameters):
        sub_circuit_list_with_params = [circ.bind_parameters(params) for circ in sub_circuit_list]
        for j in range(evaluations):
            e.add(sub_circuit_list_with_params, name=str((params, j)) + '#cut', shots=shots_cut)

    e.job_limit = np.infty
    e.execute(retrieve_interval=retrieve_interval, retries=retries)

    result = {'qaoa': [], 'cut-qaoa': []}
    for i, params in enumerate(parameters):
        qaoa_result = {'params': params, 'evaluations': []}
        cut_qaoa_result = {'params': params, 'evaluations': []}
        for j in range(evaluations):
            counts = e.get_counts_by_name(str((params, j)))
            qaoa_result['evaluations'].append(counts)
            counts_cut = e.get_counts_by_name(str((params, j)) + '#cut')
            counts_cut_dict = util.lists_to_dict(keys, counts_cut)
            counts_cut_dict_json = {}
            for fragment_id, fragment_dict in counts_cut_dict.items():
                fragment_dict_json = {}
                for key, val in fragment_dict.items():
                    fragment_dict_json[str(key)] = {'key': key, 'counts': val}
                counts_cut_dict_json[fragment_id] = fragment_dict_json
            cut_qaoa_result['evaluations'].append(counts_cut_dict_json)
        result['qaoa'].append(qaoa_result)
        result['cut-qaoa'].append(cut_qaoa_result)

    store_kwargs_as_json(path, 'result', **result)


def start(path, graph_size, edge_prop, n_links, parameters, backend=None, p=1, shots=10000,
          retrieve_interval: int = 0, retries: int = 0, reduced=False, cut_shot_factor=1, evaluations=3):
    graph_generator = erdos_renyi_graph_generator(graph_size, edge_prop)
    graph = generate_graph(graph_generator, n_links)
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')
    path = f'{path}/{backend.name()}_{time.time_ns()}'
    mkdir(path)
    store_kwargs_as_json(path, 'config', graph_size=graph_size, n_links=n_links, edge_prop=edge_prop, p=p, shots=shots,
                         backend=backend.name(), reduced=reduced, cut_shot_factor=cut_shot_factor,
                         evaluations=evaluations, parameters=parameters)
    run_experiment_1_without_shot_memory(graph, parameters, backend, evaluations, shots, cut_shot_factor, reduced, path,
                                         retrieve_interval, retries)


def start_with_graph(path, graph_path, parameters, backend=None, p=1, shots=10000,
                     retrieve_interval: int = 0, retries: int = 0, reduced=False, cut_shot_factor=1, evaluations=3):
    graph = nx.read_adjlist(graph_path)
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')
    path = f'{path}/{backend.name()}_{time.time_ns()}'
    mkdir(path)
    store_kwargs_as_json(path, 'config', graph_path=graph_path, p=p, shots=shots,
                         backend=backend.name(), reduced=reduced, cut_shot_factor=cut_shot_factor,
                         evaluations=evaluations, parameters=parameters)
    run_experiment_1_without_shot_memory(graph, parameters, backend, evaluations, shots, cut_shot_factor, reduced, path,
                                         retrieve_interval, retries)


def main():
    parser = argparse.ArgumentParser(description='Qiskit runtime experiment')
    parser.add_argument('-b', '--backend', type=str, nargs='?', default='aer_simulator',
                        help='name of the IBMQ backend')
    parser.add_argument('-g', '--graph_size', type=int, nargs='?', default=5, help='size of subgraphs')
    parser.add_argument('-e', '--edge_prob', type=float, nargs='?', default=0.5, help='edge probability')
    parser.add_argument('-l', '--links', type=int, nargs='?', default=2, help='number of links')
    parser.add_argument('--graph-path', type=str, nargs='?', default=None, help='path to a graph')
    parser.add_argument('--steps', type=int, nargs='?', default=20, help='branching factor')
    parser.add_argument('--evaluations', type=int, nargs='?', default=1,
                        help='number of evaluations per parameter')
    parser.add_argument('-f', '--factor', type=int, nargs='?', default=1,
                        help='Cut-shot-factor')
    parser.add_argument('-s', '--shots', type=int, nargs='?', default=10000, help='number of shots')
    parser.add_argument('-r', '--reduced', type=bool, nargs='?', default=True, help='reduced cut')
    parser.add_argument('--retrieve_interval', type=int, nargs='?', default=60,
                        help='Retrieve interval of jobs in seconds')
    parser.add_argument('--retries', type=int, nargs='?', default=3, help='Number of retries in case of errors')

    args = parser.parse_args()

    parameters = get_parameters(np.linspace(0, np.pi, args.steps), np.linspace(0, 2 * np.pi, 2 * args.steps))

    if args.backend == 'aer_simulator':
        backend = qiskit.Aer.get_backend('aer_simulator')
    else:
        backend, _ = get_backend_and_provider(args.backend)

    mkdir(DIR_PATH)

    if args.graph_path is None:
        start(DIR_PATH, args.graph_size, args.edge_prob, args.links, parameters, backend=backend, reduced=args.reduced,
              evaluations=args.evaluations, cut_shot_factor=args.factor, shots=args.shots,
              retrieve_interval=args.retrieve_interval, retries=args.retries)
    else:
        start_with_graph(DIR_PATH, args.graph_path, parameters,
                         backend=backend, reduced=args.reduced,
                         evaluations=args.evaluations, cut_shot_factor=1, shots=args.shots,
                         retrieve_interval=args.retrieve_interval, retries=args.retries)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logging.getLogger('circuit_cutting').setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(s_handler)
    logging.getLogger('circuit_cutting').addHandler(s_handler)

    main()
