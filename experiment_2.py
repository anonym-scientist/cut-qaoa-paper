import logging
import time

import networkx as nx
from qiskit.providers.ibmq.api.exceptions import WebsocketError

from json_utils import store_kwargs_as_json
from graphs import draw_graph
from runtime_helpers.interaction_functions import build_program, upload_program, start_program, delete_program

logger = logging.getLogger(__name__)


def run_experiment_2(path, graph, n_rounds, provider, backend=None, p=1, shots=10000,
                     retrieve_interval: int = 0,
                     retries: int = 0, reduced=False,
                     cut_shot_factor=1, log_level='INFO', log_modules=None,
                     algorithms=None):
    if log_modules is None:
        log_modules = []
    if algorithms is None:
        algorithms = ['qaoa', 'cut-qaoa']

    sub_graph_size = graph.number_of_nodes() // 2
    cuts = sum([1 if u < sub_graph_size <= v or v < sub_graph_size <= u else 0 for u, v in graph.edges])

    draw_graph(graph, f'{path.resolve()}/graph')
    nx.write_adjlist(graph, f'{path.resolve()}/graph.txt')

    opt_results = []
    opt_results_cut = []
    result = []
    result_cut = []
    cuts = cuts * p
    if reduced:
        shots_cut = cut_shot_factor * int(shots / (2 * (4 ** cuts)))
    else:
        shots_cut = cut_shot_factor * int(shots / (2 * (5 ** cuts)))

    if backend is None:
        backend_name = 'aer_simulator'
    else:
        backend_name = backend.name()

    metadata = {'name': 'runtime_multi_param', 'max_execution_time': 28800, 'description': 'A test program'}

    # program = build_program('runtime_programs/runtime_with_imports.py.py', python_paths=['.'])
    # program_id = upload_program(provider, data=program, meta_data=metadata)

    program_id = upload_program(provider, data_file='runtime_programs/runtime_single_file.py', meta_data=metadata)

    if backend is not None:
        program_options = {'backend_name': backend_name}
    else:
        program_options = {'backend_name': 'ibmq_qasm_simulator'}

    def callback(job_id, interim_result):
        print(job_id, interim_result)
        if isinstance(interim_result, dict):
            if '__final_result__' in interim_result:
                return

            iteration = interim_result['iteration']
            graph_path = path / str(iteration)
            graph_path.mkdir(parents=True, exist_ok=True)

            if '__qaoa_result__' in interim_result:
                qaoa_result = interim_result['result']
                opt_results.append(qaoa_result)
                result.append((qaoa_result['cut'], qaoa_result['cut_val']))
                store_kwargs_as_json(str(graph_path.resolve()), 'qaoa', **qaoa_result)
            elif '__qaoa_short_result__' in interim_result:
                qaoa_short_result = interim_result['result']
                opt_results.append(qaoa_short_result)
                result.append((qaoa_short_result['cut'], qaoa_short_result['cut_val']))
                store_kwargs_as_json(str(graph_path.resolve()), 'qaoa-short', **qaoa_short_result)
            elif '__cut_qaoa_result__' in interim_result:
                cut_qaoa_result = interim_result['result']
                opt_results_cut.append(cut_qaoa_result)
                result_cut.append((cut_qaoa_result['cut'], cut_qaoa_result['cut_val']))
                store_kwargs_as_json(str(graph_path.resolve()), 'cut-qaoa', **cut_qaoa_result)

    combined_inputs = {
        'edge_list': list(nx.to_edgelist(graph)),
        'n_rounds': n_rounds,
        'shots': shots,
        'shots_cut': shots_cut,
        'partitions': [sub_graph_size, sub_graph_size],
        'p': p,
        'retrieve_interval': retrieve_interval,
        'retries': retries,
        'reduced': reduced,
        'log_level': log_level,
        'log_modules': log_modules,
        'algorithms': algorithms

    }

    job = start_program(provider, program_id, program_options, combined_inputs, callback=callback)

    return job, program_id


def wait_for_finish(job, retries, provider, program_id, path):
    result_dict = {}

    for i in range(retries + 1):
        try:
            result_dict = job.result()
        except WebsocketError as wse:
            logger.info(wse.message)
            if i >= retries:
                raise wse
            time.sleep(60 * 2 ** i)
        break

    delete_program(provider, program_id)

    store_kwargs_as_json(str(path.resolve()), 'result', **result_dict)
