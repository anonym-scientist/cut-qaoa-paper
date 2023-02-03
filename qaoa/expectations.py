import time
from typing import Optional, Callable, List, Tuple, Dict

import networkx as nx
import qiskit
from qiskit.visualization import plot_histogram

from qaoa.circuit_generation import create_qaoa_circ, create_qaoa_circ_parameterized
from circuit_cutting import preprocess, util
from circuit_cutting.execute import Executor
from circuit_cutting.postprocess import QaoaPostProcessor
from circuit_cutting.util import array_to_counts


def goemans_williamson(graph):
    import cvxgraphalgs as cvxgr
    approximation = cvxgr.algorithms.goemans_williamson_weighted(graph)
    approximation_list = []
    # cut in reverse order like ibmq measurement
    for n in range(len(approximation.vertices)):
        if n in approximation.left:
            approximation_list.insert(0, 0)
        else:
            approximation_list.insert(0, 1)

    return maxcut_obj(approximation_list, graph)


def maxcut_obj(x, graph):
    """
    Given a bitstring as a solution, this function returns
    the number of edges shared between the two partitions
    of the graph.

    Args:
        x: str
           solution bitstring

        graph: networkx graph

    Returns:
        obj: float
             Objective
    """
    obj = 0
    n = graph.number_of_nodes() - 1
    for i, j in graph.edges():
        if x[n - i] != x[n - j]:
            obj -= graph.edges[i, j]['weight']

    return obj


def compute_expectation(counts: dict,
                        graph: nx.Graph,
                        obj_func: Optional[Callable[[str, nx.Graph], float]] = None
                        ) -> float:
    """
    Computes expectation value based on measurement results
    """
    if obj_func is None:
        obj_func = maxcut_obj

    avg = 0
    sum_count = 0
    print(f"Obj evals:{len(counts)}")
    for bitstring, count in counts.items():
        obj = obj_func(bitstring, graph)
        avg += obj * count
        sum_count += count

    return avg / sum_count


def compute_expectation_cut(result, G, obj_func=None, dense=True, quiet=False):
    """
    Computes expectation value based on measurement results

    Args:
        result: ndarray

        G: networkx graph

    Returns:
        avg: float
             expectation value
    """
    if obj_func is None:
        obj_func = maxcut_obj

    avg = 0
    sum_count = 0
    format_str = '{0:0' + str(G.number_of_nodes()) + 'b}'
    eval_count = 0
    if dense:
        for state, prob in enumerate(result):
            if prob != 0:
                eval_count += 1
                obj = obj_func(format_str.format(state), G)
                avg += obj * prob
                sum_count += prob
        if not quiet:
            print(f"Obj evals cut:{eval_count}")
    else:
        result = result.todok()
        if not quiet:
            print(f"Obj evals cut:{result.count_nonzero()}")
        for state, prob in result.items():
            obj = obj_func(format_str.format(state[0]), G)
            avg += obj * prob
            sum_count += prob

    return avg / sum_count


def execute_cut_graph(graph: nx.Graph,
                      theta: List[float],
                      partitions: List[int],
                      reduced: bool,
                      backend: Optional[qiskit.providers.Backend],
                      shots: int,
                      retrieve_interval: int = 0,
                      retries: int = 1
                      ) -> Tuple[Dict, Dict]:
    """
    Cut the QAOA circuit and execute the sub-circuits

    :param graph: graph instance
    :param theta: QAOA parameters
    :param partitions: specify the cuts
    :param reduced: whether the cut is executed with a reduced amount of circuits or not
    :param backend: the backend to execute
    :param shots: number of shots
    :param retrieve_interval: time interval in seconds in which the executor checks termination of the jobs
    :param retries: how many times the executor tries to execute a failed job
    :return: sub-circuit results and information about the sub-circuits
    """
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')
    qc = create_qaoa_circ(graph, theta)
    sub_circuits, sub_circuits_info = preprocess.split(qc, partitions, reduced)
    counts_cut = execute_cut(sub_circuits, backend, shots, retrieve_interval, retries)
    return counts_cut, sub_circuits_info


def execute_cut(sub_circuits: Dict,
                backend: Optional[qiskit.providers.Backend],
                shots: int,
                retrieve_interval: int = 0,
                retries: int = 1,
                parameters: Optional[List[float]] = None) -> Dict:
    keys, sub_circuit_list = util.dict_to_lists(sub_circuits)
    if parameters is not None:
        sub_circuit_list = [circ.bind_parameters(parameters) for circ in sub_circuit_list]
    executor = Executor(backend)
    executor.add(sub_circuit_list, "cut", shots=shots)
    executor.execute(retrieve_interval=retrieve_interval, retries=retries)
    counts_cut = util.lists_to_dict(keys, executor.get_counts_by_name("cut"))
    return counts_cut


def get_expectation(graph: nx.Graph,
                    shots: int = 512,
                    backend: Optional[qiskit.providers.Backend] = None,
                    obj_func: Optional[Callable[[str, nx.Graph], float]] = None,
                    plot: bool = False,
                    sub_graph_nodes: Optional[List[int]] = None,
                    gamma_sign: Tuple[int, int] = (1, 1),
                    cut_edges_at_the_end=False,
                    p: int = 1,
                    retrieve_interval: int = 0,
                    retries: int = 1
                    ):
    """
    Runs parametrized circuit
    """
    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')

    circuit = create_qaoa_circ_parameterized(graph, p, cut_edges_at_the_end)

    def execute_circ(theta):
        print(f'Start {time.time()}')
        qc = circuit.bind_parameters(theta)
        executor = Executor(backend)
        executor.add(qc, "circuit", shots=shots)
        executor.execute(retrieve_interval=retrieve_interval, retries=retries)
        counts = executor.get_counts_by_name("circuit")[0]
        if plot:
            plot_histogram(counts, title=f"No_Cut: {theta}", bar_labels=False).show()
        print(f'Finish {time.time()}')
        if sub_graph_nodes is None:
            return compute_expectation(counts, graph, obj_func)
        else:
            sub_graph = graph.subgraph(sub_graph_nodes)
            sub_graph = nx.relabel_nodes(sub_graph, {old: new for new, old in enumerate(sub_graph_nodes)})
            return compute_expectation(counts, sub_graph, obj_func)

    return execute_circ


def get_expectation_cut(graph: nx.Graph,
                        partitions: List[int],
                        shots: int = 512,
                        backend: Optional[qiskit.providers.Backend] = None,
                        obj_func: Optional[Callable[[str, nx.Graph], float]] = None,
                        reduced: bool = False,
                        plot: bool = False,
                        retrieve_interval: int = 0,
                        retries: int = 1,
                        p: int = 1
                        ):
    """
    Runs parametrized circuit
    """

    if backend is None:
        backend = qiskit.Aer.get_backend('aer_simulator')

    circuit = create_qaoa_circ_parameterized(graph, p)
    sub_circuits, sub_circuits_info = preprocess.split(circuit, partitions, reduced)

    def execute_circ(theta, counts_cut=None):
        print(f'Start {time.time()}')
        if counts_cut is None:
            counts_cut = execute_cut(sub_circuits, backend, shots, retrieve_interval, retries, theta)

        postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
        params = _theta_to_postprocess_params(theta, postprocessor.num_cuts)
        result = postprocessor.postprocess(params)

        if plot:
            counts = array_to_counts(result)
            plot_histogram(counts, title=f"Cut: {theta}", bar_labels=False).show()

        print(f'Finish {time.time()}')
        return compute_expectation_cut(result, graph, obj_func, True)

    return execute_circ


def _theta_to_postprocess_params(theta: List[float], num_cuts: int):
    gamma = theta[len(theta) // 2:]
    return [g for _ in range(num_cuts) for g in gamma]
