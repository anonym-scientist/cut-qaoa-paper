import logging

import networkx as nx
import numpy as np

from optimization.callbacks import get_cobyla_callback
from qaoa.qaoa_functions import run_qaoa, get_max_result_and_counts, run_qaoa_cut, get_max_result_and_counts_cut
from runtime_helpers.program_functions import activate_logging

logger = logging.getLogger(__name__)


def main(backend, user_messenger, **kwargs):
    """Main entry point of the program.

    Args:
        backend: Backend to submit the circuits to.
        user_messenger: Used to communicate with the program consumer.
        kwargs: User inputs.
    """
    edge_list = kwargs.pop('edge_list')
    n_rounds = kwargs.pop('n_rounds', 10)
    shots = kwargs.pop('shots', 10000)
    shots_cut = kwargs.pop('shots_cut', 10000)
    p = kwargs.pop('p', 1)
    retrieve_interval = kwargs.pop('retrieve_interval', 1)
    retries = kwargs.pop('retries', 2)
    partitions = kwargs.pop('partitions')
    reduced = kwargs.pop('reduced', True)
    log_level = kwargs.pop('log_level', 'INFO')
    log_modules = kwargs.pop('log_modules', [])
    algorithms = kwargs.pop('algorithms', ['qaoa', 'cut-qaoa'])

    graph = nx.from_edgelist(edge_list)
    for e in graph.edges():
        graph[e[0]][e[1]]['weight'] = 1

    activate_logging(user_messenger, log_level, log_modules)

    initial_param_list = []

    result_qaoa_list = []
    result_cut_qaoa_list = []
    result_qaoa_short_list = []

    for i in range(n_rounds):

        params = []
        params.extend(np.random.uniform(0, np.pi, p))  # beta parameters
        params.extend(np.random.uniform(0, 2 * np.pi, p))  # gamma parameters

        user_messenger.publish({'__params__': True, 'iteration': i, 'params': params})

        initial_param_list.append(params)

        if 'qaoa' in algorithms:
            logger.info('Start qaoa')
            optimization_path = [params]
            callback_qaoa = get_cobyla_callback(logger, optimization_path)
            r_opt = run_qaoa(graph, shots=shots, params=params, backend=backend, retrieve_interval=retrieve_interval,
                             retries=retries, callback=callback_qaoa)

            logger.info('Start qaoa get_max')
            r_max = get_max_result_and_counts(graph, shots, r_opt.x, backend, retrieve_interval=retrieve_interval,
                                              retries=retries)

            result_qaoa = {'cut': r_max[0], 'cut_val': r_max[1], 'counts': r_max[2], 'initial_params': params,
                           'optimization_path': optimization_path}
            result_qaoa.update(r_opt.__dict__)
            user_messenger.publish({'__qaoa_result__': True, 'iteration': i, 'result': result_qaoa})
            result_qaoa_list.append(result_qaoa)

        if 'qaoa-short' in algorithms:
            logger.info('Start qaoa-short')
            optimization_path = [params]
            callback_qaoa_short = get_cobyla_callback(logger, optimization_path)
            r_opt = run_qaoa(graph, shots=shots, params=params, backend=backend, retrieve_interval=retrieve_interval,
                             retries=retries, callback=callback_qaoa_short, short_circuits=True)

            logger.info('Start qaoa-short get_max')
            r_max = get_max_result_and_counts(graph, shots, r_opt.x, backend, retrieve_interval=retrieve_interval,
                                              retries=retries, cut_edges_at_the_end=True)

            result_qaoa_short = {'cut': r_max[0], 'cut_val': r_max[1], 'counts': r_max[2], 'initial_params': params,
                                 'optimization_path': optimization_path}
            result_qaoa_short.update(r_opt.__dict__)
            user_messenger.publish({'__qaoa_short_result__': True, 'iteration': i, 'result': result_qaoa_short})
            result_qaoa_short_list.append(result_qaoa_short)

        if 'cut-qaoa' in algorithms:
            logger.info('Start cut-qaoa')
            optimization_path = [params]
            callback_cut_qaoa = get_cobyla_callback(logger, optimization_path)
            r_opt = run_qaoa_cut(graph, shots=shots_cut, params=params, partitions=partitions, backend=backend,
                                 reduced=reduced,
                                 retrieve_interval=retrieve_interval, retries=retries, callback=callback_cut_qaoa)

            logger.info('Start cut-qaoa get_max')

            r_max = get_max_result_and_counts_cut(graph, shots_cut, r_opt.x, partitions, backend, reduced=reduced,
                                                  retrieve_interval=retrieve_interval, retries=retries)

            result_cut_qaoa = {'cut': r_max[0], 'cut_val': r_max[1], 'counts': r_max[2], 'initial_params': params,
                               'optimization_path': optimization_path}
            result_cut_qaoa.update(r_opt.__dict__)
            user_messenger.publish({'__cut_qaoa_result__': True, 'iteration': i, 'result': result_cut_qaoa})
            result_cut_qaoa_list.append(result_cut_qaoa)

        logger.info('Finish')

    result_dict = {'__final_result__': True,
                   'params': initial_param_list}

    if len(result_qaoa_list) > 0:
        result_dict['qaoa_results'] = result_qaoa_list
    if len(result_qaoa_short_list) > 0:
        result_dict['qaoa_short_results'] = result_qaoa_short_list
    if len(result_cut_qaoa_list) > 0:
        result_dict['cut_qaoa_results'] = result_cut_qaoa_list

    return result_dict
