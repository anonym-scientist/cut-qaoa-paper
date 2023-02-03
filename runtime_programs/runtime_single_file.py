import copy
import itertools
import json
import logging
import typing
from collections import defaultdict, Counter
import time
import datetime

import networkx as nx
import numpy as np
import qiskit
from qiskit import *
from qiskit.algorithms.optimizers import *
from qiskit.circuit.library import HGate, RZGate, ZGate, Measure
from qiskit.providers import JobError
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit.providers.ibmq import IBMQBackend, IBMQBackendJobLimitError
from qiskit.providers.ibmq.job import IBMQJobError
from qiskit.providers.ibmq.runtime import UserMessenger
from qiskit.result import Result
from qiskit.utils.measurement_error_mitigation import get_measured_qubits, build_measurement_error_mitigation_circuits
from qiskit.utils.mitigation import CompleteMeasFitter, TensoredMeasFitter
from qiskit.circuit import Parameter
from scipy.sparse import dok_matrix
from scipy import sparse

logger = logging.getLogger(__name__)

_OPTIMIZER_TO_CLASS_MAP = {
    "ADAM": ADAM,
    "AQGD": AQGD,
    "CG": CG,
    "COBYLA": COBYLA,
    "GSLS": GSLS,
    "GradientDescent": GradientDescent,
    "L_BFGS_B": L_BFGS_B,
    "NELDER_MEAD": NELDER_MEAD,
    "NFT": NFT,
    "P_BFGS": P_BFGS,
    "POWELL": POWELL,
    "SLSQP": SLSQP,
    "SPSA": SPSA,
    "QNSPSA": QNSPSA,
    "TNC": TNC,
    "CRS": CRS,
    "DIRECT_L": DIRECT_L,
    "DIRECT_L_RAND": DIRECT_L_RAND,
    "ESCH": ESCH,
    "ISRES": ISRES,
    "SNOBFIT": SNOBFIT,
    "BOBYQA": BOBYQA,
    "IMFIL": IMFIL
}

SUBSTITUTIONS = {"P_1-P_0": {"gates": ["P_1", "P_0"], "factors": {"P_1": 1, "P_0": -1}},
                 "RZ_minus-RZ_plus": {"gates": ["RZ_minus", "RZ_plus"], "factors": {"RZ_minus": 1, "RZ_plus": -1}}}

SUBSTITUTIONS_REDUCED = {"P_1-P_0": {"gates": ["P_1", "P_0"],
                                     "factors": {"P_1": 1, "P_0": -1}},
                         "RZ_minus-RZ_plus": {"gates": ["RZ_plus", "Z", "I"],
                                              "factors": {"RZ_plus": -2, "Z": 1, "I": 1}}}

GATE_COMBINATIONS_RZZ = [("I", "I"), ("Z", "Z"), ("P_1-P_0", "RZ_minus-RZ_plus"), ("RZ_minus-RZ_plus", "P_1-P_0")]

GATE_VERSIONS = ["I", "Z", "RZ_plus", "RZ_minus", "MEAS"]
GATE_VERSIONS_REDUCED = ["I", "Z", "RZ_plus", "MEAS"]


def create_qaoa_circ(graph: nx.Graph, theta: typing.List[float], sub_graph_nodes: typing.Optional[
    typing.List[int]] = None,
                     gamma_sign=(1, 1)):
    """
    Creates a parametrized qaoa circuit
    """
    if sub_graph_nodes is None:
        nodes = list(graph.nodes())
        nodes = sorted(nodes)
        n_qubits = len(nodes)
        node_qubit_mapping = {node: qubit for qubit, node in enumerate(nodes)}
    else:
        n_qubits = len(sub_graph_nodes)
        node_qubit_mapping = {node: qubit for qubit, node in enumerate(sub_graph_nodes)}

    p = len(theta) // 2  # number of alternating unitaries
    qc = QuantumCircuit(n_qubits)

    beta = theta[:p]
    gamma = theta[p:]

    # initial_state
    for i in range(0, n_qubits):
        qc.h(i)

    for irep in range(0, p):
        # problem unitary
        for u, v in list(graph.edges()):
            try:
                q_u = node_qubit_mapping[u]
                q_v = node_qubit_mapping[v]
                qc.rzz(gamma[irep], q_u, q_v)
            except KeyError:
                if u in sub_graph_nodes:
                    q_u = node_qubit_mapping[u]
                    qc.rz(gamma_sign[0] * gamma[irep], q_u)
                elif v in sub_graph_nodes:
                    q_v = node_qubit_mapping[v]
                    qc.rz(gamma_sign[1] * gamma[irep], q_v)

        # mixer unitary
        for i in range(0, n_qubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


def execute_cut_graph(graph: nx.Graph,
                      theta: typing.List[float],
                      partitions: typing.List[int],
                      reduced: bool,
                      backend: typing.Optional[qiskit.providers.Backend],
                      shots: int,
                      retrieve_interval: int = 0,
                      retries: int = 1
                      ) -> typing.Tuple[typing.Dict, typing.Dict]:
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
    sub_circuits, sub_circuits_info = split(qc, partitions, reduced)
    counts_cut = execute_cut(sub_circuits, backend, shots, retrieve_interval, retries)
    return counts_cut, sub_circuits_info


def get_max_result_and_counts_cut(graph: nx.Graph,
                                  shots: int,
                                  params: typing.Union[typing.List[float], np.ndarray],
                                  partitions: typing.List[int],
                                  backend: typing.Optional[qiskit.providers.Backend] = None,
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


def _append_gate(circuit: QuantumCircuit, gate_ver, qubit, c_reg_name="c_mid_meas"):
    if gate_ver == "RZ_minus":
        circuit.append(RZGate(-np.pi / 2), [qubit])

    elif gate_ver == "RZ_plus":
        circuit.append(RZGate(np.pi / 2), [qubit])

    elif gate_ver == "I":
        pass

    elif gate_ver == "Z":
        circuit.append(ZGate(), [qubit])

    elif gate_ver == "MEAS":
        top_meas_cr = ClassicalRegister(1, c_reg_name)
        circuit.add_register(top_meas_cr)
        circuit.append(Measure(), [qubit], [top_meas_cr])


def _get_partition_idx(qubit, partitions):
    for idx, part in enumerate(partitions):
        if qubit < part[0]:
            return idx - 1
    if qubit <= partitions[-1][1]:
        return len(partitions) - 1


def generate_sub_circuits(cut, reduced=False):
    sub_circuits = {}
    sub_circuits_info = {}
    gate_versions = GATE_VERSIONS_REDUCED if reduced else GATE_VERSIONS
    circuit_count = 0
    for fragment_idx, fragments in enumerate(cut):
        num_cuts = len(fragments["parts"]) - 1
        width = fragments["width"]
        sub_circuits[fragment_idx] = {}
        sub_circuits_info[fragment_idx] = {
            "num_cuts": num_cuts,
            "width": width,
            "connections": [fragment["link"]["idx"] for fragment in fragments["parts"][:-1]],
            "gates": [fragment["gate"] for fragment in fragments["parts"][:-1]],
            "params": [fragment["params"] for fragment in fragments["parts"][:-1]]
        }
        for gates in itertools.product(gate_versions, repeat=num_cuts):
            qc = QuantumCircuit(width, width)
            for i in range(num_cuts):
                fragment = fragments["parts"][i]
                circ = fragment["circuit"]
                qubit = fragment["qubit"]
                gate_ver = gates[i]
                qc.compose(circ, inplace=True)
                _append_gate(qc, gate_ver, qubit, f'c_mid_meas_{i}')
            last_circ = fragments["parts"][num_cuts]["circuit"]
            qc.compose(last_circ, inplace=True)
            sub_circuits[fragment_idx][gates] = qc
            circuit_count += 1
    logger.info(f'Generated {circuit_count} subcircuits')
    return sub_circuits, sub_circuits_info


def split(circuit: QuantumCircuit, fragment_widths: list[int], reduced: bool = False) -> typing.Tuple[dict, dict]:
    cut = _split(circuit, fragment_widths)
    return generate_sub_circuits(cut, reduced)


def _split(circuit: QuantumCircuit, fragment_widths: typing.List[int]) -> typing.List[
    typing.Dict[str, typing.Union[int, list]]]:
    if not circuit.num_qubits == sum(fragment_widths):
        raise Exception("Partitions do not match to the circuit")
    current_circuit_fragments = []
    cut = []
    partitions = []
    partition_start = 0
    for width in fragment_widths:
        current_circuit_fragments.append(QuantumCircuit(width, width))
        partitions.append((partition_start, partition_start + width - 1))
        partition_start = partition_start + width
        cut.append({"width": width, "parts": []})

    # qubit.index is deprecated, hence the index is accessed via the following map
    qbit_indices = {bit: index
                    for index, bit in enumerate(circuit.qubits)}

    clbit_indices = {bit: index
                     for index, bit in enumerate(circuit.clbits)}

    cuts_counter = 0

    for i, (gate, qubits, clbits) in enumerate(circuit.data):
        q_index = [qbit_indices[q] for q in qubits]
        cl_index = [clbit_indices[cl] for cl in clbits]
        if len(q_index) > 0:
            start_qubit = q_index[0]
            part_start_idx = _get_partition_idx(start_qubit, partitions)
            partition_start = partitions[part_start_idx]
            if len(q_index) == 1:
                circ_frag = current_circuit_fragments[part_start_idx]
                circ_frag.append(gate,
                                 qargs=[q - partition_start[0] for q in q_index],
                                 cargs=[cl - partition_start[0] for cl in cl_index])
            elif gate.name == "barrier":
                for circ in current_circuit_fragments:
                    circ.barrier()
            elif len(q_index) == 2:
                end_qubit = q_index[1]
                part_end_idx = _get_partition_idx(end_qubit, partitions)
                if part_start_idx == part_end_idx:
                    circ_frag = current_circuit_fragments[part_start_idx]
                    circ_frag.append(gate,
                                     qargs=[q - partition_start[0] for q in q_index],
                                     cargs=[cl - partition_start[0] for cl in cl_index])
                if part_start_idx != part_end_idx:
                    cuts_counter += 1
                    if gate.name == "cx":
                        target_qubit = end_qubit
                        part_target_idx = part_end_idx
                        circ_frag = current_circuit_fragments[part_target_idx]
                        circ_frag.append(HGate(), qargs=[target_qubit - partitions[part_target_idx][0]], cargs=[])
                    if gate.name == "cz" or gate.name == "cx" or gate.name == "rzz":
                        if part_end_idx < part_start_idx:
                            temp_part = part_end_idx
                            part_end_idx = part_start_idx
                            part_start_idx = temp_part
                            partition_start = partitions[part_start_idx]
                            temp_qubit = end_qubit
                            end_qubit = start_qubit
                            start_qubit = temp_qubit

                        partition_end = partitions[part_end_idx]

                        cut[part_start_idx]["parts"].append({
                            "circuit": current_circuit_fragments[part_start_idx],
                            "qubit": start_qubit - partition_start[0],
                            "link": {"idx": part_end_idx, "qubit": end_qubit - partition_end[0]},
                            "gate": gate.name,
                            "params": gate.params
                        })
                        width = fragment_widths[part_start_idx]
                        current_circuit_fragments[part_start_idx] = QuantumCircuit(width, width)
                        cut[part_end_idx]["parts"].append({
                            "circuit": current_circuit_fragments[part_end_idx],
                            "qubit": end_qubit - partition_end[0],
                            "link": {"idx": part_start_idx, "qubit": start_qubit - partition_start[0]},
                            "gate": gate.name,
                            "params": gate.params
                        })
                        width = fragment_widths[part_end_idx]
                        current_circuit_fragments[part_end_idx] = QuantumCircuit(width, width)
                        # TODO add connectivity
                    if gate.name == "cx":
                        circ_frag = current_circuit_fragments[part_target_idx]
                        circ_frag.append(HGate(), qargs=[target_qubit - partitions[part_target_idx][0]], cargs=[])
    logger.debug(f"Number of cuts: {cuts_counter}")

    for fragment_idx, circ in enumerate(current_circuit_fragments):
        cut[fragment_idx]["parts"].append({"circuit": circ, "qubit": None, "link": None})

    return cut


def dict_to_lists(dictionary):
    key_list = []
    value_list = []
    for key, value in dictionary.items():
        if isinstance(value, dict):
            rec_key_list, rec_value_list = dict_to_lists(value)
            key_list.extend([[key] + keys for keys in rec_key_list])
            value_list.extend(rec_value_list)
        else:
            key_list.append([key])
            value_list.append(value)

    return key_list, value_list


def lists_to_dict(key_list, value_list):
    dictionary = {}
    for keys, value in zip(key_list, value_list):
        temp_dict = dictionary
        for i in range(len(keys) - 1):
            key = keys[i]
            if not key in temp_dict:
                temp_dict[key] = {}
            temp_dict = temp_dict[key]
        temp_dict[keys[-1]] = value
    return dictionary


def dict_to_array(result_dict, n_qubits) -> np.ndarray:
    array = np.zeros(2 ** n_qubits)
    for key, value in result_dict.items():
        if isinstance(key, int):
            pass
        elif isinstance(key, str):
            if key.startswith("0x"):
                key = int(key, 16)
            elif key.startswith("0") or key.startswith("1"):
                key = int(key, 2)
            else:
                raise ValueError("String could not be decoded")
        else:
            raise TypeError("Type has to be either integer or string")
        array[key] = value
    return array


def _split_meas_counts(counts):
    zero_meas_counts = {}
    one_meas_counts = {}

    for key, value in counts.items():
        split_meas = key.split(" ", 1)
        mid_meas = split_meas[0]
        circ_meas = split_meas[1]
        if mid_meas == "0":
            zero_meas_counts[circ_meas] = value
        elif mid_meas == "1":
            one_meas_counts[circ_meas] = value
        else:
            raise Exception("Unknown Measurement")

    return zero_meas_counts, one_meas_counts


def _meas_key_to_projection_rec(key, counts):
    new_counts = {}
    shots = {}
    prob = {}
    if all(k != 'MEAS' for k in key):
        new_counts[key] = counts
        prob[key] = 1
        shots[key] = sum(counts.values())
    else:
        for i, op in enumerate(key):
            if op == 'MEAS':
                zero_counts, one_counts = _split_meas_counts(counts)
                zero_one_shots = sum(counts.values())
                zero_prob = sum(zero_counts.values()) / zero_one_shots if zero_one_shots != 0 else 0
                one_prob = sum(one_counts.values()) / zero_one_shots if zero_one_shots != 0 else 0
                key_zero = list(key)
                key_zero[i] = "P_0"
                c_0, p_0, s_0 = _meas_key_to_projection_rec(tuple(key_zero), zero_counts)
                new_counts.update(c_0)
                shots.update(s_0)
                prob.update({k: p * zero_prob for k, p in p_0.items()})
                key_one = list(key)
                key_one[i] = "P_1"
                c_1, p_1, s_1 = _meas_key_to_projection_rec(tuple(key_one), one_counts)
                new_counts.update(c_1)
                shots.update(s_1)
                prob.update({k: p * one_prob for k, p in p_1.items()})
    return new_counts, prob, shots


def _meas_to_projection(counts):
    new_counts = {}
    prob = {}
    shots = {}
    for fragment_idx, fragment_counts in counts.items():
        new_counts[fragment_idx] = {}
        prob[fragment_idx] = {}
        shots[fragment_idx] = {}
        for key, fragment_count in fragment_counts.items():
            c, p, s = _meas_key_to_projection_rec(key, fragment_count)
            new_counts[fragment_idx].update(c)
            prob[fragment_idx].update(p)
            shots[fragment_idx].update(s)

    return new_counts, prob, shots


def _tensor_product(list_of_arrays, dense=False):
    result = list_of_arrays[-1]
    kron = np.kron if dense else sparse.kron
    for i in range(len(list_of_arrays) - 2, -1, -1):
        result = kron(result, list_of_arrays[i])
    return result


def _get_all_combos(gate_combo_list, substitution_dict, gate_index):
    all_gate_lists = [[]]
    all_factors = [1]
    gates_to_append = []
    for idx, gate_combo in enumerate(gate_combo_list):
        gate = gate_combo[gate_index]
        if gate not in substitution_dict.keys():
            gates_to_append.append(gate)
            continue
        old_all_gate_lists = all_gate_lists
        old_factors = all_factors
        all_gate_lists = []
        all_factors = []
        for gate_list, factor in zip(old_all_gate_lists, old_factors):
            gate_list.extend(gates_to_append)
            for g in substitution_dict[gate]["gates"]:
                copy_list = gate_list.copy()
                copy_list.append(g)
                all_gate_lists.append(copy_list)
                all_factors.append(factor * substitution_dict[gate]["factors"][g])

        gates_to_append = []
    if len(gates_to_append) > 0:
        for gate_list in all_gate_lists:
            gate_list.extend(gates_to_append)

    return all_gate_lists, all_factors


def _get_factor(gate_combo_list, params):
    c = 1
    for i, gate_combo in enumerate(gate_combo_list):
        if gate_combo == ("I", "I"):
            c *= np.power(np.cos(params[i] / 2), 2)
        elif gate_combo == ("Z", "Z"):
            c *= np.power(np.sin(params[i] / 2), 2)
        else:
            c *= np.sin(params[i] / 2) * np.cos(params[i] / 2)
    return c


class PostProcessor:

    def __init__(self, sub_circuit_counts, sub_circuit_info):
        self._counts, self._probabilities, self._shots = _meas_to_projection(sub_circuit_counts)
        self._circuit_info = sub_circuit_info
        self._fragment_widths = [fragment_info["width"] for fragment_info in sub_circuit_info.values()]
        self._circuit_width = sum(self._fragment_widths)
        self._num_cuts = sum(fragment_info["num_cuts"] for fragment_info in sub_circuit_info.values()) // 2
        self._n_fragments = len(self._fragment_widths)
        # TODO support more than two fragments

    def _get_counts_as_array(self, fragment_index, key):
        data = self._counts[fragment_index][key]
        return dict_to_array(data, self._fragment_widths[fragment_index])

    def _get_counts_as_dok_matrix(self, fragment_index, key):
        data = self._counts[fragment_index][key]
        dok = dok_matrix((1, 2 ** self._fragment_widths[fragment_index]))
        for k, v in data.items():
            dok[0, k] = v
        return dok

    @property
    def num_cuts(self):
        return self._num_cuts


class QaoaPostProcessor(PostProcessor):

    def __init__(self, sub_circuit_counts, sub_circuit_info, reduced_substitution=True):
        super().__init__(sub_circuit_counts, sub_circuit_info)
        self._fragment_params = [[nested_list[0] for nested_list in fragment_info["params"] if len(nested_list) > 0] for
                                 fragment_info in sub_circuit_info.values()]
        if reduced_substitution:
            self._substitutions = SUBSTITUTIONS_REDUCED
        else:
            self._substitutions = SUBSTITUTIONS

    def postprocess(self, params=None, **kwargs):
        start_time = time.time()

        if params is None:
            params = self._fragment_params[0]

        logger.debug(params)
        result = self._postprocess_core(params)

        logger.info(f"Postprocessing duration {(time.time() - start_time)} s")
        return result

    def _postprocess_core(self, params):
        start_time = time.time()

        result = np.zeros(2 ** self._circuit_width)
        debug = {}
        for gate_combo_list in itertools.product(GATE_COMBINATIONS_RZZ, repeat=int(self._num_cuts)):
            c = _get_factor(gate_combo_list, params)
            sub_circuit_arrays = []
            for fragment_idx in range(self._n_fragments):
                gate_lists, factor_lists = _get_all_combos(gate_combo_list, self._substitutions, fragment_idx)
                array_sum = np.zeros(2 ** self._fragment_widths[fragment_idx])
                for gates, factor in zip(gate_lists, factor_lists):
                    key = tuple(gates)
                    arr = self._get_counts_as_array(fragment_idx, key)
                    if self._shots[fragment_idx][key] != 0:
                        array_sum += (factor * self._probabilities[fragment_idx][key] / self._shots[fragment_idx][
                            key]) * arr
                sub_circuit_arrays.append(array_sum)

            tensor = _tensor_product(sub_circuit_arrays, True)
            result += c * tensor
            debug[gate_combo_list] = {'c': c, 'tensor': tensor, 'value_sum_abs': np.sum(np.abs(tensor)),
                                      'value_sum': np.sum(tensor),
                                      's0_value_sum_abs': np.sum(np.abs(sub_circuit_arrays[0])),
                                      's0_value_sum': np.sum(sub_circuit_arrays[0]),
                                      's1_value_sum_abs': np.sum(np.abs(sub_circuit_arrays[1])),
                                      's1_value_sum': np.sum(sub_circuit_arrays[1])}
        logger.info(f"Postprocessing duration {(time.time() - start_time)} s")
        return result


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


def _theta_to_postprocess_params(theta: typing.List[float], num_cuts: int):
    gamma = theta[len(theta) // 2:]
    return [g for _ in range(num_cuts) for g in gamma]


def execute_cut(sub_circuits: typing.Dict,
                backend: typing.Optional[qiskit.providers.Backend],
                shots: int,
                retrieve_interval: int = 0,
                retries: int = 1,
                parameters: typing.Optional[typing.List[float]] = None) -> typing.Dict:
    keys, sub_circuit_list = dict_to_lists(sub_circuits)
    if parameters is not None:
        sub_circuit_list = [circ.bind_parameters(parameters) for circ in sub_circuit_list]
    executor = Executor(backend)
    executor.add(sub_circuit_list, "cut", shots=shots)
    executor.execute(retrieve_interval=retrieve_interval, retries=retries)
    counts_cut = lists_to_dict(keys, executor.get_counts_by_name("cut"))
    return counts_cut


def get_expectation_cut(graph: nx.Graph,
                        partitions: typing.List[int],
                        shots: int = 512,
                        backend: typing.Optional[qiskit.providers.Backend] = None,
                        obj_func: typing.Optional[typing.Callable[[str, nx.Graph], float]] = None,
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
    sub_circuits, sub_circuits_info = split(circuit, partitions, reduced)

    def execute_circ(theta, counts_cut=None):
        print(f'Start {time.time()}')
        if counts_cut is None:
            counts_cut = execute_cut(sub_circuits, backend, shots, retrieve_interval, retries, theta)

        postprocessor = QaoaPostProcessor(counts_cut, sub_circuits_info, reduced_substitution=reduced)
        params = _theta_to_postprocess_params(theta, postprocessor.num_cuts)
        result = postprocessor.postprocess(params)

        print(f'Finish {time.time()}')
        return compute_expectation_cut(result, graph, obj_func, True)

    return execute_circ


def run_qaoa_cut(graph: nx.Graph,
                 shots: int,
                 params: typing.Union[typing.List[float], np.ndarray],
                 partitions: typing.List[int],
                 backend: typing.Optional[qiskit.providers.Backend] = None,
                 method: str = 'COBYLA',
                 reduced: bool = False,
                 retrieve_interval: int = 0,
                 retries: int = 0,
                 callback: typing.Callable[[typing.List[float]], None] = None,
                 opt_kwargs: typing.Optional[typing.Dict] = None):
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


def get_max_result_and_counts(graph: nx.Graph,
                              shots: int,
                              params: typing.Union[typing.List[float], np.ndarray],
                              backend: typing.Optional[qiskit.providers.Backend] = None,
                              retrieve_interval: int = 0,
                              retries: int = 0,
                              seed_simulator: typing.Optional[int] = None,
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
                        obj_func: typing.Optional[typing.Callable[[str, nx.Graph], float]] = None
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


def get_expectation(graph: nx.Graph,
                    shots: int = 512,
                    backend: typing.Optional[qiskit.providers.Backend] = None,
                    obj_func: typing.Optional[typing.Callable[[str, nx.Graph], float]] = None,
                    plot: bool = False,
                    sub_graph_nodes: typing.Optional[typing.List[int]] = None,
                    gamma_sign: typing.Tuple[int, int] = (1, 1),
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
        print(f'Finish {time.time()}')
        if sub_graph_nodes is None:
            return compute_expectation(counts, graph, obj_func)
        else:
            sub_graph = graph.subgraph(sub_graph_nodes)
            sub_graph = nx.relabel_nodes(sub_graph, {old: new for new, old in enumerate(sub_graph_nodes)})
            return compute_expectation(counts, sub_graph, obj_func)

    return execute_circ


def create_qaoa_circ_parameterized(graph: nx.Graph, p: int = 1, cut_edges_at_the_end=False):
    """
    Creates a parametrized qaoa circuit
    """

    nodes = list(graph.nodes())
    nodes = sorted(nodes)
    n_qubits = len(nodes)
    node_qubit_mapping = {node: qubit for qubit, node in enumerate(nodes)}

    qc = QuantumCircuit(n_qubits)

    beta = [Parameter(f'beta_{i}') for i in range(p)]
    gamma = [Parameter(f'gamma_{i}') for i in range(p)]

    n_qubits_half = n_qubits // 2
    cut_edges = []

    # initial_state
    for i in range(0, n_qubits):
        qc.h(i)

    for irep in range(0, p):
        # problem unitary
        for u, v in list(graph.edges()):
            q_u = node_qubit_mapping[u]
            q_v = node_qubit_mapping[v]
            if cut_edges_at_the_end and ((q_u < n_qubits_half <= q_v) or (q_v < n_qubits_half <= q_u)):
                cut_edges.append((q_u, q_v))
                continue
            qc.rzz(gamma[irep], q_u, q_v)

        for q_u, q_v in cut_edges:
            qc.rzz(gamma[irep], q_u, q_v)

        # mixer unitary
        for i in range(0, n_qubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


class QiskitResultEncoder(json.JSONEncoder):
    """ Special json encoder for qiskit results """

    def default(self, obj):
        if isinstance(obj, qiskit.result.Result):
            result_dict = obj.to_dict()
            result_dict['__qiskit.result.Result__'] = True
            return result_dict
        elif isinstance(obj, datetime.datetime):
            return {'__datetime.datetime__': True, 'iso': obj.isoformat()}
        else:
            return json.JSONEncoder.default(self, obj)


def as_qiskit_result(dct):
    if '__qiskit.result.Result__' in dct:
        return qiskit.result.Result.from_dict(dct)
    elif '__datetime.datetime__' in dct:
        datetime.datetime.fromisoformat(dct['iso'])
    return dct


def get_fitter_cls(mitigator: str):
    if mitigator == 'CompleteMeasFitter':
        return CompleteMeasFitter
    elif mitigator == 'TensoredMeasFitter':
        return TensoredMeasFitter
    else:
        raise ValueError(f'Unknown mitigator: {mitigator}')


def _split_result(result: Result):
    result_list = []
    for res in result.results:
        result_list.append(
            Result(result.backend_name, result.backend_version, result.qobj_id, result.job_id, result.success,
                   [res], result.date, result.status, result.header)
        )
    return result_list


def _get_counts_from_result(result: Result):
    return sum(result.get_counts(0).values())


def _split_mid_circ_measurement_counts(counts):
    new_counts = defaultdict(lambda: defaultdict(dict))
    for key, count in counts.items():
        mid_meas_key, meas_key = key.rsplit(' ', 1)
        new_counts[mid_meas_key][meas_key] = count
    return new_counts


def _merge_mid_circ_measurement_counts(mid_counts):
    merged_counts = {}
    for mid_meas_key, counts in mid_counts.items():
        for key, count in counts.items():
            merged_counts[f'{mid_meas_key} {key}'] = count
    return merged_counts


def _mitigate_mid_circ_measurement_counts(counts, mitigation_fitter):
    split_counts = _split_mid_circ_measurement_counts(counts)
    mit_split_counts = {}
    for mid_key, mid_counts in split_counts.items():
        mit_split_counts[mid_key] = mitigation_fitter.filter.apply(mid_counts, method='pseudo_inverse')
    return _merge_mid_circ_measurement_counts(mit_split_counts)


def _get_transpile_info(circuit: QuantumCircuit):
    return {'width': circuit.width(), 'depth': circuit.depth(), 'size': circuit.size(),
            'num_nonlocal_gates': circuit.num_nonlocal_gates()}


class ExecutorResults:

    def __init__(self, memory=False):
        self._mitigation_fitters = {}
        self._circuit_labels = {}
        self._mitigator_index = defaultdict(list)
        self._state_labels = {}
        self._memory = memory
        self._results = {}
        self._circuit_index = []
        self._mitigation_index = []
        self._circuit_qubit_index = {}
        self._name_index = {}
        self._memory_cache = {}
        self._transpilation_info = {}

    def get_all_results(self):
        result_list = [result for shots, start_idx, end_idx in self._circuit_index for result in
                       self._results[shots][start_idx:end_idx]]
        return result_list

    def get_results_by_name(self, name, mitigator=None):
        shots, start_idx, end_idx = self._name_index[name]
        key = (shots, start_idx, end_idx)
        if mitigator is None:
            return self._results[shots][start_idx:end_idx]
        elif not str(key) in self._circuit_qubit_index:
            raise ValueError('No mitigation performed for circuits with name: ' + name)
        else:
            NotImplementedError('Mitigation is not implemented yet')
            # mitigated_results = []
            # qubit_index = self._circuit_qubit_index[str(key)]
            # for result in self._results[shots][start_idx:end_idx]:
            #     try:
            #         mitigated_results.append(self.get_mitigation_fitter(mitigator, qubit_index).filter.apply(result))
            #     except (ValueError, QiskitError):
            #         counts = _mitigate_mid_circ_measurement_counts(result.get_counts(),
            #                                                        self.get_mitigation_fitter(mitigator, qubit_index))
            #         mitigated_results.append()
            # return mitigated_results

    def get_transpilation_info(self, name):
        shots, start_idx, end_idx = self._name_index[name]
        return self._transpilation_info[shots][start_idx:end_idx]

    def get_all_counts(self):
        return list(map(lambda res: res.get_counts(), self.get_all_results()))

    def get_counts_by_name(self, name, mitigator=None):
        return list(map(lambda res: res.get_counts(), self.get_results_by_name(name, mitigator=mitigator)))

    def get_counts_by_name_from_memory(self, name, shots=None, mitigator=None):
        if not self._memory:
            raise ValueError('Memory is false')
        if shots is None:
            return self.get_results_by_name(name)
        results = self.get_results_by_name(name)
        max_shots = _get_counts_from_result(results[0])
        if shots < 0 or shots > max_shots:
            raise ValueError(f'Number of shots must be between 0 and {max_shots}')
        try:
            memory_results = self._memory_cache[name]
        except KeyError:
            memory_results = [res.get_memory() for res in self.get_results_by_name(name)]
            self._memory_cache[name] = memory_results
        results = list(map(lambda res: dict(Counter(res[:shots])), memory_results))
        if mitigator is None:
            return results
        # mitigate result
        shots, start_idx, end_idx = self._name_index[name]
        key = (shots, start_idx, end_idx)
        if not str(key) in self._circuit_qubit_index:
            raise ValueError('No mitigation performed for circuits with name: ' + name)
        mitigated_results = []
        qubit_index = self._circuit_qubit_index[str(key)]
        for result in results:
            try:
                mitigated_results.append(
                    self.get_mitigation_fitter(mitigator, qubit_index).filter.apply(result, method='pseudo_inverse'))
            except (ValueError, QiskitError) as e:
                mitigated_results.append(
                    _mitigate_mid_circ_measurement_counts(result, self.get_mitigation_fitter(mitigator, qubit_index)))
        return mitigated_results

    def _compute_mitigation_fitter(self, mitigator, qubit_index):
        name_key = f'__mitigation_{mitigator}_{qubit_index}__'
        if name_key not in self._name_index:
            raise ValueError('No mitigation performed')
        mit_results = self.get_results_by_name(name_key)

        if mitigator == 'CompleteMeasFitter':
            meas_error_mitigation_fitter = CompleteMeasFitter(mit_results, self._state_labels[name_key],
                                                              qubit_list=qubit_index.split('_'),
                                                              circlabel=self._circuit_labels[name_key])
        elif mitigator == 'TensoredMeasFitter':
            meas_error_mitigation_fitter = TensoredMeasFitter(
                mit_results, mit_pattern=self._state_labels[name_key], circlabel=self._circuit_labels[name_key]
            )
        else:
            raise ValueError(f'Unkown mitigator: {mitigator}')

        return meas_error_mitigation_fitter

    def get_mitigation_fitter(self, mitigator, qubit_index):
        name_key = f'__mitigation_{mitigator}_{qubit_index}__'
        if name_key in self._mitigation_fitters:
            return self._mitigation_fitters[name_key]
        fitter = self._compute_mitigation_fitter(mitigator, qubit_index)
        self._mitigation_fitters[name_key] = fitter
        return fitter

    def to_dict(self):
        return {'memory': self._memory,
                'circuit_index': self._circuit_index,
                'name_index': self._name_index,
                'results': self._results,
                'circuit_lables': self._circuit_labels,
                'mitigator_index': self._mitigator_index,
                'state_labels': self._state_labels,
                'mitigation_index': self._mitigation_index,
                'circuit_qubit_index': self._circuit_qubit_index,
                'transpilation_info': self._transpilation_info
                }

    @classmethod
    def from_dict(cls, data):
        in_data = copy.copy(data)
        obj = cls(memory=in_data['memory'])
        obj._circuit_index = in_data['circuit_index']
        obj._name_index = in_data['name_index']
        obj._results = {int(k): v for k, v in in_data['results'].items()}
        obj._circuit_labels = in_data['circuit_lables']
        obj._mitigator_index = in_data['mitigator_index']
        obj._state_labels = in_data['state_labels']
        obj._mitigation_index = in_data['mitigation_index']
        obj._circuit_qubit_index = in_data['circuit_qubit_index']
        obj._transpilation_info = in_data['transpilation_info']
        return obj

    def save_results(self, path, name=None):
        if name is None:
            name = 'executor_results'
        with open(f'{path}/{name}.json', 'w') as outfile:
            json.dump(self.to_dict(), outfile, indent=4, cls=QiskitResultEncoder)

    @classmethod
    def load_results(cls, path):
        with open(path, 'r') as file:
            data = json.load(file, object_hook=as_qiskit_result)
        return cls.from_dict(data)


class Executor(ExecutorResults):

    def __init__(self, backend, memory=False):
        super().__init__(memory)
        self.backend = backend
        self._runtime = False
        if isinstance(backend, IBMQBackend):
            # or programruntime.runtime_backend.RuntimeBackend (not available in local qiskit)
            self._max_experiments = self.backend.configuration().max_experiments
            self._max_shots = self.backend.configuration().max_shots
            self._job_limit = self.backend.job_limit().maximum_jobs
            self._job_limit = self._job_limit if self._job_limit is not None else 1
            self._default_shots = self._max_shots
        elif isinstance(backend, QasmSimulator) or isinstance(backend, AerSimulator):
            self._max_experiments = np.Infinity
            self._max_shots = np.Infinity
            self._job_limit = np.Infinity
            self._default_shots = 10000
        else:
            # assume RuntimeBackend
            self._max_experiments = self.backend.configuration().max_experiments
            self._max_shots = self.backend.configuration().max_shots
            self._job_limit = 1
            self._default_shots = self._max_shots
            self._runtime = True
        # else:
        #     raise Exception(f'Unknown backend type: {type(backend)} ')

        self.circuits = {}
        self._circuit_index = []
        self._mitigation_index = defaultdict(list)
        self._circuit_qubit_index = {}
        self._name_index = {}
        self._circuit_count = 0
        self._circuit_count_shots = {}
        self._results = defaultdict(list)
        self._mitigation_shots = 2048

    @property
    def shots(self):
        return self._default_shots

    @shots.setter
    def shots(self, value):
        if value < 1 or value > self._max_shots:
            raise ValueError(f"Shot number has to be between 1 and {self._max_shots}")
        self._default_shots = value

    @property
    def job_limit(self):
        return self._job_limit

    @job_limit.setter
    def job_limit(self, value):
        if value < 1:
            raise ValueError(f"Job limit has to be greater than 0")
        self._job_limit = value

    @property
    def max_experiments(self):
        return self._max_experiments

    @max_experiments.setter
    def max_experiments(self, value):
        if value < 1:
            raise ValueError(f"Maximal number of experiments per job has to be greater than 0")
        self._max_experiments = value

    def add(self, circuits, name=None, shots=None, mitigators=None):
        # ensure circuits is a list of QuantumCircuits
        if isinstance(circuits, QuantumCircuit):
            circuits = [circuits]
        elif not isinstance(circuits, list) and all(isinstance(elem, list) for elem in circuits):
            raise TypeError
        if shots is None:
            shots = self._default_shots
        if shots in self.circuits.keys():
            self.circuits[shots].extend(circuits)
        else:
            self.circuits[shots] = circuits
            self._circuit_count_shots[shots] = 0
        n_circuits = len(circuits)
        index_tuple = (shots, self._circuit_count_shots[shots], self._circuit_count_shots[shots] + n_circuits)
        if name is not None:
            self._name_index[name] = index_tuple
        self._circuit_index.append(index_tuple)
        self._circuit_count += n_circuits
        self._circuit_count_shots[shots] += n_circuits
        if mitigators is not None and isinstance(mitigators, typing.Iterable):
            for mitigator in mitigators:
                self._mitigation_index[mitigator].append(index_tuple)

    def execute(self, seed_simulator=None, retrieve_interval=0, retries=1):
        transpiled_circuits = {}
        for shots, circuits in self.circuits.items():
            logger.debug(f'Transpile {len(circuits)} circuits ({shots} shots)')
            t_circuits = transpile(circuits, self.backend)
            transpiled_circuits[shots] = t_circuits
            self._transpilation_info[shots] = [_get_transpile_info(c) for c in t_circuits]

        if len(self._mitigation_index) > 0:
            self._add_mitigation_circuits(transpiled_circuits)

        t = 0
        while t <= retries:
            t += 1
            start_time = time.time()
            num_circuits = 0
            num_circuits_for_shots = 0
            num_active_jobs = 0
            jobs = []
            result_counter = 0
            circuit_batch = []
            shots_iter = iter(transpiled_circuits.keys())
            shots = next(shots_iter)
            batch_shots = shots
            last_check = time.time()
            final_state = True
            job_errors = 0
            try:
                while num_circuits < self._circuit_count or len(jobs) > 0 or len(circuit_batch) > 0:
                    if num_circuits < self._circuit_count and len(circuit_batch) == 0:
                        start_circuit = num_circuits
                        batch_shots = shots
                        if len(transpiled_circuits[shots][num_circuits_for_shots:]) <= self._max_experiments:
                            circuit_batch.extend(transpiled_circuits[shots][num_circuits_for_shots:])
                            num_circuits += len(transpiled_circuits[shots][num_circuits_for_shots:])
                            num_circuits_for_shots = 0
                            try:
                                shots = next(shots_iter)
                            except StopIteration:
                                pass
                        else:
                            circuit_batch.extend(
                                transpiled_circuits[shots][
                                num_circuits_for_shots:num_circuits_for_shots + self._max_experiments])
                            num_circuits += self._max_experiments
                            num_circuits_for_shots += self._max_experiments
                        logger.debug(
                            f"Create batch of length {num_circuits - start_circuit} with {batch_shots} shots starting with circuit {start_circuit}")

                    if num_active_jobs < self._job_limit and len(circuit_batch) > 0:
                        logger.debug("Try to submit circuit batch")
                        try:
                            job = self.backend.run(circuit_batch, shots=batch_shots, seed_simulator=seed_simulator,
                                                   memory=self._memory)
                            logger.debug("Circuit batch submitted")
                            jobs.append(job)
                            num_active_jobs += 1
                            circuit_batch = []
                        except IBMQBackendJobLimitError:
                            logger.debug("Job limit reached")
                            time.sleep(max(retrieve_interval - time.time() + last_check, 0))
                    else:
                        if not final_state:
                            time.sleep(max(retrieve_interval - time.time() + last_check, 0))

                    if len(jobs) > 0 and (time.time() - last_check > retrieve_interval or final_state):
                        last_check = time.time()
                        logger.debug(f"Check final state")
                        final_state = jobs[0].in_final_state()
                        logger.debug(f"Job is in final state: {final_state}")
                        if final_state or self._runtime:  # job.final_state() does always return False in qiskit runtime
                            job = jobs.pop(0)
                            try:
                                result = job.result()
                                job_errors = 0
                            except JobError as e:
                                if job_errors >= retries:
                                    raise e
                                logger.info(f'An (IBMQ)JobError occurred: {e.message}')
                                logger.info(f'Retry: {t <= retries}')
                                job_circuits = job.circuits()
                                n_job_circuits = len(job_circuits)
                                logger.info(f'Failed job contains {n_job_circuits} circuits')
                                job_shots = job.backend_options()['shots']
                                circuit_start = len(self._results[job_shots])
                                logger.info(f'Transpile circuits again')
                                t_circuits = transpile(
                                    self.circuits[job_shots][circuit_start:circuit_start + n_job_circuits],
                                    self.backend)
                                if len(job_circuits) > 1:
                                    logger.info('Submit 2 jobs for circuits')
                                    new_job_1 = self.backend.run(t_circuits[:n_job_circuits // 2],
                                                                 shots=job_shots,
                                                                 seed_simulator=seed_simulator,
                                                                 memory=self._memory)
                                    new_job_2 = self.backend.run(t_circuits[n_job_circuits // 2:],
                                                                 shots=job_shots,
                                                                 seed_simulator=seed_simulator,
                                                                 memory=self._memory)
                                    jobs.insert(0, new_job_1)
                                    jobs.insert(1, new_job_2)
                                else:
                                    logger.info('Submit 1 job for circuits')
                                    new_job = self.backend.run(job.circuits(),
                                                               shots=job.backend_options()['shots'],
                                                               seed_simulator=seed_simulator,
                                                               memory=self._memory)
                                    jobs.insert(0, new_job)
                                    job_errors += 1
                                continue
                            result_shots = _get_counts_from_result(result)
                            logger.debug(
                                f"Got execution result with {result_shots} shots starting with circuit {result_counter}")
                            result_list = _split_result(result)
                            self._results[result_shots].extend(result_list)
                            result_counter += len(result_list)
                            num_active_jobs -= 1
                logger.info(f"Execution duration {(time.time() - start_time)} s")
                break
            except (IBMQJobError, JobError) as e:
                logger.info(f'An (IBMQ)JobError occurred: {e.message}')
                logger.info(f'Retry: {t <= retries}')

    def _add_mitigation_circuits(self, transpiled_circuits):
        for mitigator, mitigation_index in self._mitigation_index.items():
            fitter_cls = get_fitter_cls(mitigator)
            for shots, start_idx, end_idx in mitigation_index:
                key = str((shots, start_idx, end_idx))
                qubit_index, _ = get_measured_qubits(transpiled_circuits[shots][start_idx:end_idx])
                qubit_index_str = "_".join([str(x) for x in qubit_index])
                self._circuit_qubit_index[key] = qubit_index_str
                name_key = f'__mitigation_{fitter_cls.__name__}_{qubit_index_str}__'
                if name_key not in self._name_index:
                    self._mitigator_index[fitter_cls.__name__].append(qubit_index_str)
                    mit_pattern = [[i] for i in range(len(qubit_index))]
                    cal_circuits, self._state_labels[name_key], self._circuit_labels[
                        name_key] = build_measurement_error_mitigation_circuits(
                        qubit_index,
                        fitter_cls,
                        self.backend,
                        backend_config={},
                        compile_config={},
                        mit_pattern=mit_pattern
                    )
                    if self._mitigation_shots in transpiled_circuits.keys():
                        transpiled_circuits[self._mitigation_shots].extend(cal_circuits)
                    else:
                        transpiled_circuits[self._mitigation_shots] = cal_circuits
                        self._circuit_count_shots[self._mitigation_shots] = 0

                    n_mitigation_circuits = len(cal_circuits)
                    index_tuple = (self._mitigation_shots, self._circuit_count_shots[self._mitigation_shots],
                                   self._circuit_count_shots[self._mitigation_shots] + n_mitigation_circuits)
                    self._name_index[name_key] = index_tuple
                    self._circuit_index.append(index_tuple)
                    self._circuit_count += n_mitigation_circuits
                    self._circuit_count_shots[self._mitigation_shots] += n_mitigation_circuits

    @classmethod
    def create_ibmq(cls, backend_name='ibmq_qasm_simulator', hub='ibm-q'):
        IBMQ.load_account()
        provider = IBMQ.get_provider(hub=hub)
        backend = provider.get_backend(backend_name)
        return cls(backend)

    @classmethod
    def create_local_sim(cls, backend_name='aer_simulator'):
        backend = Aer.get_backend(backend_name)
        return cls(backend)

    @classmethod
    def create(cls, backend_name='aer_simulator'):
        if backend_name.startswith("ibmq"):
            return cls.create_ibmq(backend_name)
        else:
            return cls.create_local_sim(backend_name)

    @classmethod
    def load_results(cls, path) -> ExecutorResults:
        return ExecutorResults.load_results(path)


def get_expectation(graph: nx.Graph,
                    shots: int = 512,
                    backend: typing.Optional[qiskit.providers.Backend] = None,
                    obj_func: typing.Optional[typing.Callable[[str, nx.Graph], float]] = None,
                    plot: bool = False,
                    sub_graph_nodes: typing.Optional[typing.List[int]] = None,
                    gamma_sign: typing.Tuple[int, int] = (1, 1),
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
        print(f'Finish {time.time()}')
        if sub_graph_nodes is None:
            return compute_expectation(counts, graph, obj_func)
        else:
            sub_graph = graph.subgraph(sub_graph_nodes)
            sub_graph = nx.relabel_nodes(sub_graph, {old: new for new, old in enumerate(sub_graph_nodes)})
            return compute_expectation(counts, sub_graph, obj_func)

    return execute_circ


def get_optimizer(method: str, **kwargs) -> Optimizer:
    return _OPTIMIZER_TO_CLASS_MAP[method](**kwargs)


def run_qaoa(graph: nx.Graph,
             shots: int,
             params: typing.Union[typing.List[float], np.ndarray],
             backend: typing.Optional[qiskit.providers.Backend] = None,
             method: str = 'COBYLA',
             retrieve_interval: int = 0,
             retries: int = 0,
             callback: typing.Callable[[typing.List[float]], None] = None,
             minimum: bool = False,
             short_circuits: bool = False,
             opt_kwargs: typing.Optional[typing.Dict] = None):
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


class LogRecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, logging.LogRecord):
            obj_dict = obj.__dict__
            obj_dict['__log_record__'] = True
            return obj_dict
        return json.JSONEncoder.default(self, obj)


class UserMessengerHandler(logging.Handler):

    def __init__(self, user_messenger: UserMessenger):
        super().__init__()
        self._user_messenger = user_messenger

    def emit(self, record: logging.LogRecord) -> None:
        self._user_messenger.publish(json.dumps(record, cls=LogRecordEncoder))


def activate_logging(user_messenger, log_level, log_modules):
    if len(log_modules) == 0:
        return
    handler = UserMessengerHandler(user_messenger)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    for module in log_modules:
        logging.getLogger(module).setLevel(log_level)
        logging.getLogger(module).addHandler(handler)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_cobyla_callback(logger, optimization_path=None):
    callback_counter = 0
    callback_time = time.time()
    if optimization_path is None:
        optimization_path = []

    def _callback(x):
        nonlocal callback_counter
        nonlocal callback_time
        logger.info(json.dumps({'iteration': callback_counter, 'duration': time.time() - callback_time, 'params': x},
                               cls=NumpyEncoder))
        optimization_path.append(x)
        callback_time = time.time()
        callback_counter += 1

    return _callback


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
