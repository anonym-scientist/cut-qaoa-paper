import logging
import time
from multiprocessing import Pool
from typing import Tuple, List, Optional, Callable, Union, Dict

import networkx as nx
import numpy as np
import qiskit.providers
from qiskit import *

from optimization.optimizers import get_optimizer
from qaoa.circuit_generation import create_qaoa_circ_parameterized
from qaoa.expectations import get_expectation, get_expectation_cut, maxcut_obj, execute_cut_graph, \
    compute_expectation_cut
from qaoa.objective import ObjectiveFunction
from circuit_cutting.execute import Executor
from circuit_cutting.postprocess import QaoaPostProcessor

logger = logging.getLogger(__name__)


def brutforce_max_cut(graph):
    n = graph.number_of_nodes()
    format_str = '{0:0' + str(n) + 'b}'
    best_cut = format_str.format(0)
    best_value = maxcut_obj(best_cut, graph)
    for i in range(1, 2 ** n):
        cut = format_str.format(i)
        value = maxcut_obj(cut, graph)
        if best_value > value:
            best_cut = cut
            best_value = value
    return best_cut, best_value


def _compute_cut_value_func(args):
    graph, format_str, x = args
    cut = format_str.format(x)
    value = maxcut_obj(cut, graph)
    return cut, value


def brutforce_max_cut_multi_core(graph):
    n = graph.number_of_nodes()
    format_str = '{0:0' + str(n) + 'b}'
    with Pool() as p:
        values = p.map(_compute_cut_value_func, ((graph, format_str, i) for i in range(2 ** n)))
    return min(values, key=lambda x: x[1])


def run_qaoa(graph: nx.Graph,
             shots: int,
             params: Union[List[float], np.ndarray],
             backend: Optional[qiskit.providers.Backend] = None,
             method: str = 'COBYLA',
             retrieve_interval: int = 0,
             retries: int = 0,
             callback: Callable[[List[float]], None] = None,
             minimum: bool = False,
             short_circuits: bool = False,
             opt_kwargs: Optional[Dict] = None):
    obj_func = ObjectiveFunction()
    if minimum:
        obj_func.sign = -1
    expectation = get_expectation(graph, shots=shots, backend=backend, obj_func=obj_func.cut_size,
                                  retrieve_interval=retrieve_interval, retries=retries,
                                  cut_edges_at_the_end=short_circuits)
    if opt_kwargs is None:
        opt_kwargs = {}
    optimizer = get_optimizer(method, callback=callback, **opt_kwargs)
    result = optimizer.minimize(fun=expectation, x0=params)
    result._total_obj_evals = obj_func.get_number_of_unique_evaluations()
    logger.info(f'Number of different objective evaluations: {obj_func.get_number_of_unique_evaluations()}')
    return result


def run_qaoa_cut(graph: nx.Graph,
                 shots: int,
                 params: Union[List[float], np.ndarray],
                 partitions: List[int],
                 backend: Optional[qiskit.providers.Backend] = None,
                 method: str = 'COBYLA',
                 reduced: bool = False,
                 retrieve_interval: int = 0,
                 retries: int = 0,
                 callback: Callable[[List[float]], None] = None,
                 opt_kwargs: Optional[Dict] = None):
    obj_func = ObjectiveFunction()
    expectation_cut = get_expectation_cut(graph, partitions=partitions, shots=shots, backend=backend,
                                          obj_func=obj_func.cut_size, reduced=reduced,
                                          retrieve_interval=retrieve_interval, retries=retries)
    if opt_kwargs is None:
        opt_kwargs = {}
    optimizer = get_optimizer(method, callback=callback, **opt_kwargs)
    result = optimizer.minimize(expectation_cut, params)
    result._total_obj_evals = obj_func.get_number_of_unique_evaluations()
    logger.info(f'Number of different objective evaluations: {obj_func.get_number_of_unique_evaluations()}')
    return result


def get_max_result(graph: nx.Graph,
                   shots: int,
                   params: Union[List[float], np.ndarray],
                   backend: Optional[qiskit.providers.Backend] = None,
                   retrieve_interval: int = 0,
                   retries: int = 0,
                   seed_simulator: Optional[int] = None):
    cut, obj, _ = get_max_result_and_counts(graph, shots, params, backend, retrieve_interval, retries, seed_simulator)
    return cut, maxcut_obj(cut, graph)


def get_max_result_and_counts(graph: nx.Graph,
                              shots: int,
                              params: Union[List[float], np.ndarray],
                              backend: Optional[qiskit.providers.Backend] = None,
                              retrieve_interval: int = 0,
                              retries: int = 0,
                              seed_simulator: Optional[int] = None,
                              p: int = 1,
                              cut_edges_at_the_end: bool = False):
    if backend is None:
        backend = Aer.get_backend('aer_simulator')
    circuit = create_qaoa_circ_parameterized(graph, p, cut_edges_at_the_end)
    qc = circuit.bind_parameters(params)

    executor = Executor(backend)
    executor.add(qc, "circuit", shots=shots)
    executor.execute(seed_simulator=seed_simulator, retrieve_interval=retrieve_interval, retries=retries)
    counts = executor.get_counts_by_name("circuit")[0]
    cut = max(counts, key=counts.get)
    return cut, maxcut_obj(cut, graph), counts


def get_max_result_cut(graph: nx.Graph,
                       shots: int,
                       params: Union[List[float], np.ndarray],
                       partitions: List[int],
                       backend: Optional[qiskit.providers.Backend] = None,
                       reduced: bool = False,
                       retrieve_interval: int = 0,
                       retries: int = 0):
    cut, obj, _ = get_max_result_and_counts_cut(graph, shots, params, partitions, backend, reduced, retrieve_interval,
                                                retries)
    return cut, obj


def get_max_result_and_counts_cut(graph: nx.Graph,
                                  shots: int,
                                  params: Union[List[float], np.ndarray],
                                  partitions: List[int],
                                  backend: Optional[qiskit.providers.Backend] = None,
                                  reduced: bool = False,
                                  retrieve_interval: int = 0,
                                  retries: int = 0):
    counts_cut, sub_circuits_info = execute_cut_graph(graph, params, partitions, reduced, backend, shots,
                                                      retrieve_interval,
                                                      retries)
    postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
    result = postprocessor.postprocess()
    format_str = '{0:0' + str(graph.number_of_nodes()) + 'b}'
    cut = format_str.format(np.argmax(result))
    return cut, maxcut_obj(cut, graph), result


def get_expectation_cut_final(graph: nx.Graph,
                              counts_cut: Optional[dict] = None,
                              sub_circuits_info: Optional[dict] = None,
                              postprocessor: Optional[QaoaPostProcessor] = None,
                              obj_func: Optional[Callable[[str, nx.Graph], float]] = None,
                              reduced: bool = False):
    if postprocessor is None:
        num_cuts = sum(fragment_info["num_cuts"] for fragment_info in sub_circuits_info.values()) // 2
        postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
    else:
        num_cuts = int(postprocessor.num_cuts)

    def execute_circ(theta):
        print(f'Start {time.time()}')
        params = []
        for i in range(num_cuts):
            params.append(theta)

        result = postprocessor.postprocess(params=params)
        print(f'Finish {time.time()}')
        return compute_expectation_cut(result, graph, obj_func, True)

    return execute_circ
