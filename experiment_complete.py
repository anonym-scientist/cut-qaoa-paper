import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import qiskit

from experiment_1 import run_experiment_1
from experiment_2 import run_experiment_2, wait_for_finish


from experiment_utils import get_parameters
from json_utils import store_kwargs_as_json
from provider_handler import get_backend_and_provider
from graphs import generate_graph, erdos_renyi_graph_generator, draw_graph

logger = logging.getLogger(__name__)

DIR_PATH = Path('experiment_complete')


def create_experiment_directory(path, backend):
    dt = datetime.now()
    exp_id = dt.strftime('%Y-%m-%d-%H-%M-%S')
    exp_path = path / f'{backend.name()}_{exp_id}'
    exp_path.mkdir(parents=True, exist_ok=True)
    return exp_path


def start(path, graph_size, n_links, edge_prob, n_rounds, provider, backend=None, p=1, shots=10000,
          retrieve_interval: int = 0, retries: int = 0, reduced=False, cut_shot_factor=1,
          log_level='INFO', log_modules=None, algorithms=None, short_circuits=False, parameters=None):
    graph = generate_graph(erdos_renyi_graph_generator(graph_size, edge_prob), n_links)
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')
    exp_path = create_experiment_directory(path, backend)
    store_kwargs_as_json(str(exp_path.resolve()), 'config', graph_size=graph_size, n_links=n_links, edge_prob=edge_prob,
                         p=p, shots=shots, backend=backend.name(), n_rounds=n_rounds, reduced=reduced,
                         cut_shot_factor=cut_shot_factor, retrieve_interval=retrieve_interval, retries=retries,
                         algorithms=algorithms, short_circuits=short_circuits, parameters=parameters)
    _run(exp_path, graph, n_rounds, parameters, provider, backend, p, shots, retrieve_interval, retries, reduced,
         cut_shot_factor, log_level, log_modules, algorithms, short_circuits)


def start_with_graph(path, graph_path, n_rounds, provider, backend=None, p=1, shots=10000,
                     retrieve_interval: int = 0, retries: int = 0, reduced=False, cut_shot_factor=1,
                     log_level='INFO', log_modules=None,
                     algorithms=None, short_circuits=False, parameters=None):
    graph = nx.read_adjlist(graph_path)
    node_mapping = {str(n): n for n in range(graph.number_of_nodes())}
    nx.relabel_nodes(graph, node_mapping, copy=False)
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')
    exp_path = create_experiment_directory(path, backend)
    store_kwargs_as_json(str(exp_path.resolve()), 'config', graph_path=graph_path,
                         p=p, shots=shots, backend=backend.name(), n_rounds=n_rounds, reduced=reduced,
                         cut_shot_factor=cut_shot_factor, retrieve_interval=retrieve_interval, retries=retries,
                         algorithms=algorithms, short_circuits=short_circuits, parameters=parameters)
    _run(exp_path, graph, n_rounds, parameters, provider, backend, p, shots, retrieve_interval, retries, reduced,
         cut_shot_factor, log_level, log_modules, algorithms, short_circuits)


def _run(exp_path, graph, n_rounds, parameters, provider, backend=None, p=1, shots=10000,
         retrieve_interval: int = 0, retries: int = 0, reduced=False, cut_shot_factor=1,
         log_level='INFO', log_modules=None, algorithms=None, short_circuits=False):
    exp_path.mkdir(parents=True, exist_ok=True)
    draw_graph(graph, f'{exp_path.resolve()}/graph')
    nx.write_adjlist(graph, f'{exp_path.resolve()}/graph.txt')
    qaoa_execution_path = exp_path / 'qaoa_execution'
    qaoa_execution_path.mkdir(parents=True, exist_ok=True)
    param_map_path = exp_path / 'param_map'
    param_map_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(exp_path / 'config.json', qaoa_execution_path)
    shutil.copy(exp_path / 'config.json', param_map_path)

    job, program_id = run_experiment_2(path=qaoa_execution_path, graph=graph, n_rounds=n_rounds,
                                       provider=provider,
                                       backend=backend, p=p, shots=shots,
                                       retrieve_interval=1, retries=retries,
                                       reduced=reduced, cut_shot_factor=cut_shot_factor,
                                       log_level=log_level,
                                       log_modules=log_modules, algorithms=algorithms)
    run_experiment_1(graph, parameters=parameters, backend=backend, evaluations=1, shots=shots,
                     cut_shot_factor=cut_shot_factor, reduced=reduced, path=param_map_path,
                     retrieve_interval=retrieve_interval, retries=retries, short_circuits=short_circuits)
    wait_for_finish(job, retries, provider, program_id, qaoa_execution_path)


def main():
    log_modules = ['program', 'circuit_cutting']

    parser = argparse.ArgumentParser(description='Qiskit runtime experiment')
    parser.add_argument('-b', '--backend', type=str, nargs='?', default='ibmq_qasm_simulator',
                        help='name of the IBMQ backend')
    parser.add_argument('-g', '--graph_size', type=int, nargs='?', default=5, help='size of subgraphs')
    parser.add_argument('-e', '--edge_prob', type=float, nargs='?', default=0.5, help='edge probability')
    parser.add_argument('-l', '--links', type=int, nargs='?', default=2, help='number of links between subgraphs')
    parser.add_argument('--graph-path', type=str, nargs='?', default=None, help='path to a graph')
    parser.add_argument('-n', '--n_rounds', type=int, nargs='?', default=3, help='number of rounds for experiment 2')
    parser.add_argument('-s', '--shots', type=int, nargs='?', default=10000, help='number of shots')
    parser.add_argument('-f', '--factor', type=int, nargs='?', default=1, help='Cut-shot-factor')
    parser.add_argument('-r', '--reduced', type=bool, nargs='?', default=True, help='reduced cut')
    parser.add_argument('-a', '--algorithms', nargs='+', default=['qaoa', 'qaoa-short', 'cut-qaoa'], help='list of algorithms')
    parser.add_argument('--steps', type=int, nargs='?', default=20, help='steps of the parameter grid')
    parser.add_argument('--short-circuits', action='store_true', help='add parallel QAOA ansatz, default: sequential ansatz only')
    parser.add_argument('--log_level', type=str, nargs='?', default='INFO', help='level of logging')
    parser.add_argument('--retrieve_interval', type=int, nargs='?', default=60, help='Retrieve interval of jobs in seconds')
    parser.add_argument('--retries', type=int, nargs='?', default=3, help='Number of retries in case of errors')

    args = parser.parse_args()

    backend, provider = get_backend_and_provider(args.backend)

    shared_args = {'provider': provider,
                   'backend': backend,
                   'n_rounds': args.n_rounds,
                   'reduced': args.reduced,
                   'shots': args.shots,
                   'cut_shot_factor': args.factor,
                   'retrieve_interval': args.retrieve_interval,
                   'retries': args.retries,
                   'log_level': args.log_level,
                   'algorithms': args.algorithms,
                   'log_modules': log_modules,
                   'parameters': get_parameters(np.linspace(0, np.pi, args.steps),
                                                np.linspace(0, 2 * np.pi, 2 * args.steps)),
                   'short_circuits': args.short_circuits
                   }

    if args.graph_path is None:
        start(DIR_PATH, args.graph_size, args.links, args.edge_prob, **shared_args)
    else:
        start_with_graph(DIR_PATH, args.graph_path, **shared_args)


if __name__ == '__main__':
    logger.setLevel(logging.INFO)
    logging.getLogger('runtime_helpers').setLevel(logging.INFO)
    logging.getLogger('circuit_cutting').setLevel(logging.DEBUG)
    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(s_handler)
    logging.getLogger('runtime_helpers').addHandler(s_handler)
    logging.getLogger('circuit_cutting').addHandler(s_handler)

    main()
