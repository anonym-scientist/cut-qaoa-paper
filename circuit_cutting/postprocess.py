import itertools
import logging
import time

import numpy as np
from scipy import sparse
from scipy.sparse import dok_matrix

from .util import dict_to_array, dict_to_sparse_array, filter_counts, marginal_counts

logger = logging.getLogger(__name__)

GATE_COMBINATIONS_PROJECTIONS = [("RZ_minus", "RZ_minus"), ("RZ_plus", "RZ_plus"), ("P_0", "I"), ("P_1", "I"),
                                 ("P_0", "Z"), ("P_1", "Z"), ("I", "P_0"), ("I", "P_1"), ("Z", "P_0"), ("Z", "P_1")]

GATE_COMBINATIONS_REDUCED = [("RZ_minus", "RZ_minus"), ("RZ_plus", "RZ_plus"), ("P_0+P_1", "I+Z"), ("I+Z", "P_0+P_1")]

GATE_COMBINATIONS_SUMS = [("P_0+P_1", "I+Z"), ("I+Z", "P_0+P_1")]

GATE_COMBINATIONS_SINGLE_OPS_TOP = [("P_0", "I"), ("P_1", "I"), ("P_0", "Z"), ("P_1", "Z")]

GATE_COMBINATIONS_SINGLE_OPS_BOT = [("I", "P_0"), ("I", "P_1"), ("Z", "P_0"), ("Z", "P_1")]

GATE_COMBINATIONS_RZZ = [("I", "I"), ("Z", "Z"), ("P_1-P_0", "RZ_minus-RZ_plus"), ("RZ_minus-RZ_plus", "P_1-P_0")]

GATE_COMBINATIONS_RZZ_SUMS = [("P_1-P_0", "RZ_minus-RZ_plus"), ("RZ_minus-RZ_plus", "P_1-P_0")]

GATE_COMBINATIONS_RZZ_SINGLE_OPS_TOP = [("P_0", "RZ_plus"), ("P_1", "RZ_plus"), ("P_0", "RZ_minus"),
                                        ("P_1", "RZ_minus")]

GATE_COMBINATIONS_RZZ_SINGLE_OPS_BOT = [("RZ_plus", "P_0"), ("RZ_plus", "P_1"), ("RZ_minus", "P_0"),
                                        ("RZ_minus", "P_1")]

SUBSTITUTIONS = {"P_1-P_0": {"gates": ["P_1", "P_0"], "factors": {"P_1": 1, "P_0": -1}},
                 "RZ_minus-RZ_plus": {"gates": ["RZ_minus", "RZ_plus"], "factors": {"RZ_minus": 1, "RZ_plus": -1}}}

SUBSTITUTIONS_REDUCED = {"P_1-P_0": {"gates": ["P_1", "P_0"],
                                     "factors": {"P_1": 1, "P_0": -1}},
                         "RZ_minus-RZ_plus": {"gates": ["RZ_plus", "Z", "I"],
                                              "factors": {"RZ_plus": -2, "Z": 1, "I": 1}}}

GATE_COMBINATIONS_RZZ_SINGLE = [("I", "I"), ("Z", "Z"), ("P_0", "RZ_plus"), ("P_1", "RZ_plus"), ("P_0", "RZ_minus"),
                                ("P_1", "RZ_minus"), ("RZ_plus", "P_0"), ("RZ_plus", "P_1"), ("RZ_minus", "P_0"),
                                ("RZ_minus", "P_1")]


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


def _get_all_keys_from_reduced(gate_combo_list, rzz=False):
    all_gate_combo_lists = [[]]
    last_update = -1
    if rzz:
        gate_combo_sums = GATE_COMBINATIONS_RZZ_SUMS
        projection_top = [("P_0", "RZ_plus"), ("P_1", "RZ_plus"), ("P_0", "RZ_minus"), ("P_1", "RZ_minus")]
        projection_bot = [("RZ_plus", "P_0"), ("RZ_plus", "P_1"), ("RZ_minus", "P_0"), ("RZ_minus", "P_1")]
    else:
        gate_combo_sums = GATE_COMBINATIONS_SUMS
        projection_top = [("P_0", "I"), ("P_1", "I"), ("P_0", "Z"), ("P_1", "Z")]
        projection_bot = [("I", "P_0"), ("I", "P_1"), ("Z", "P_0"), ("Z", "P_1")]

    for current_idx, gate_combo in enumerate(gate_combo_list):
        if gate_combo in gate_combo_sums:
            old_all_gate_combo_lists = all_gate_combo_lists
            all_gate_combo_lists = []
            if last_update - current_idx + 1 < 0:
                append = gate_combo_list[last_update + 1:current_idx]
            else:
                append = []

            if gate_combo[0].startswith("P"):
                for gcl in old_all_gate_combo_lists:
                    gcl.extend(append)
                    for gate in projection_top:
                        copy_list = gcl.copy()
                        copy_list.append(gate)
                        all_gate_combo_lists.append(copy_list)
            else:
                for gcl in old_all_gate_combo_lists:
                    gcl.extend(append)
                    for gate in projection_bot:
                        copy_list = gcl.copy()
                        copy_list.append(gate)
                        all_gate_combo_lists.append(copy_list)
            last_update = current_idx
    if last_update < len(gate_combo_list) - 1:
        append = gate_combo_list[last_update + 1:]
        for gcl in all_gate_combo_lists:
            gcl.extend(append)

    return all_gate_combo_lists


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


def _get_factor_single(gate_combo_list, params, probabilities):
    c = 1
    gates_circ_1 = []
    gates_circ_2 = []
    for i, gate_combo in enumerate(gate_combo_list):
        op_1 = gate_combo[0]
        op_2 = gate_combo[1]
        gates_circ_1.append(op_1)
        gates_circ_2.append(op_2)
        if gate_combo == ("I", "I"):
            c *= np.power(np.cos(params[i] / 2), 2)
        elif gate_combo == ("Z", "Z"):
            c *= np.power(np.sin(params[i] / 2), 2)
        else:
            c *= np.sin(params[i] / 2) * np.cos(params[i] / 2)
            if gate_combo in [("P_1", "RZ_plus"), ("P_0", "RZ_minus"), ("RZ_plus", "P_1"), ("RZ_minus", "P_0")]:
                c *= -1
    c *= probabilities[0][tuple(gates_circ_1)]
    c *= probabilities[1][tuple(gates_circ_2)]
    return c


def _tensor_product(list_of_arrays, dense=False):
    result = list_of_arrays[-1]
    kron = np.kron if dense else sparse.kron
    for i in range(len(list_of_arrays) - 2, -1, -1):
        result = kron(result, list_of_arrays[i])
    return result


def _get_data_as_array(projection_counts, info, fragment_idx, key, marginal_indices=None, fixed_bits=None,
                       marginal_func=None, dense=False):
    data = projection_counts[fragment_idx][key]
    if fixed_bits is not None:
        data = filter_counts(data, fixed_bits)
    if marginal_indices is not None:
        if len(data) > 0:
            data = marginal_counts(data, marginal_indices, marginal_func=marginal_func)
        width = len(marginal_indices)
    else:
        width = info[fragment_idx]["width"]
    if isinstance(data, np.ndarray):
        return data
    if dense:
        return dict_to_array(data, width)
    else:
        return dict_to_sparse_array(data, width)


def _get_width_marginals_and_fixed_bits(info, marginal_indices, fixed_bits):
    fragment_widths = [fragment_info["width"] for fragment_info in info.values()]
    fragment_widths_rev = list(reversed(fragment_widths))
    fragment_widths_sum = [0]
    for i in range(1, len(fragment_widths_rev)):
        fragment_widths_sum.append(fragment_widths_sum[i - 1] + fragment_widths_rev[i])

    fragment_fixed_bits = [{} for _ in range(len(fragment_widths))]
    if fixed_bits is not None:
        sorted_fixed_bits = sorted(fixed_bits.items())
        f_idx = 0
        for idx, bit in sorted_fixed_bits:
            while f_idx < len(fragment_widths_sum) - 1 and idx >= fragment_widths_sum[f_idx + 1]:
                f_idx += 1
            fragment_bit_index = fragment_widths_rev[f_idx] - (idx - fragment_widths_sum[f_idx]) - 1
            fragment_fixed_bits[len(fragment_widths_sum) - f_idx - 1][fragment_bit_index] = bit

    if marginal_indices is None:
        width = sum(fragment_widths)
        fragment_marginal_indices = [None for _ in range(len(fragment_widths))]
        return width, fragment_widths, fragment_marginal_indices, fragment_fixed_bits
    else:
        fragment_marginal_indices = [[] for _ in range(len(fragment_widths))]
        marginal_indices.sort()
        f_idx = 0
        for idx in marginal_indices:
            while f_idx < len(fragment_widths_sum) - 1 and idx >= fragment_widths_sum[f_idx + 1]:
                f_idx += 1
            fragment_marginal_index = fragment_widths_rev[f_idx] - (idx - fragment_widths_sum[f_idx]) - 1
            fragment_marginal_indices[len(fragment_widths_sum) - f_idx - 1].append(fragment_marginal_index)

        marginal_fragment_widths = [len(m_i) for m_i in fragment_marginal_indices]
        width = sum(marginal_fragment_widths)
        return width, marginal_fragment_widths, fragment_marginal_indices, fragment_fixed_bits
