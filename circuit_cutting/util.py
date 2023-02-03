from collections import defaultdict

import numpy as np
import qiskit.result
from scipy.sparse import dok_matrix, spmatrix


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


def array_to_counts(array) -> dict:
    bit_length = int(np.log2(len(array)))
    format_str = '{0:0' + str(bit_length) + 'b}'
    return {format_str.format(i): v for i, v in enumerate(array)}


def dict_to_sparse_array(result_dict, n_qubits) -> spmatrix:
    array = dok_matrix((2 ** n_qubits, 1))
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
    return array.tocsc()


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


def remove_ws_dict_keys(d):
    return {
        key.replace(' ', '') if isinstance(key, str) else key:
            remove_ws_dict_keys(value) if isinstance(value, dict) else value for key, value in d.items()
    }


def tensorproduct_of_dicts(dicts_1, dicts_2, c=1):
    tensor_counts = {}
    if c != 1:
        for k1, v1 in dicts_1.items():
            for k2, v2 in dicts_2.items():
                tensor_counts[k2 + k1] = c * v1 * v2
    else:
        for k1, v1 in dicts_1.items():
            for k2, v2 in dicts_2.items():
                tensor_counts[k2 + k1] = v1 * v2
    return tensor_counts


def add_dicts(*args, default_factory=None):
    if default_factory is None:
        default_factory = int
    result = defaultdict(default_factory)
    for d in args:
        for k, v in d.items():
            result[k] += v
    return result


def multiply_const_to_dict(d, c):
    if c != 1:
        return {k: c * v for k, v in d.items()}
    else:
        return d.copy()


def filter_dict(d, func):
    return dict(filter(func, d.items()))


def filter_counts(counts, bitstring_filter):
    if bitstring_filter is None or len(bitstring_filter) == 0:
        return counts

    def filter_func(item):
        key, _ = item
        for i, bit in bitstring_filter.items():
            if key[i] != bit:
                return False
        return True

    return filter_dict(counts, filter_func)


def _remove_space_underscore(bitstring):
    """Removes all spaces and underscores from bitstring"""
    return bitstring.replace(" ", "").replace("_", "")


def _update_dict_with_func(dictionary, key, val, func):
    if key not in dictionary.keys():
        dictionary[key] = val
    else:
        dictionary[key] = func(dictionary[key], val)


def keys_to_ints(counts):
    return {int(key, 2): val for key, val in counts.items()}


def marginal_counts(counts, indices, marginal_func=None):
    if len(indices) > 0:
        if marginal_func is None or marginal_func == 'sum':
            return qiskit.result.marginal_counts(counts, indices)
        elif marginal_func == 'max':
            return _marginal_key_list_func(counts, indices, max)
        elif marginal_func == 'min':
            return _marginal_key_list_func(counts, indices, min)
        elif marginal_func == 'avg':
            return _marginal_key_list_func(counts, indices, np.average, zeros=True)
        elif marginal_func == 'avg-squared':
            return _marginal_key_list_func(counts, indices, lambda x_list: np.average([x * x for x in x_list]),
                                           zeros=True)
        elif marginal_func == 'var':
            return _marginal_key_list_func(counts, indices, np.var, zeros=True)
        elif marginal_func == 'sum-squared':
            return _marginal_key_list_func(counts, indices, lambda x_list: sum([x * x for x in x_list]))
        elif marginal_func == 'sum-exp':
            return _marginal_key_list_func(counts, indices, lambda x_list: sum([np.exp(x) for x in x_list]))
        elif marginal_func == 'sum-max':
            sum_counts = qiskit.result.marginal_counts(counts, indices)
            max_counts = _marginal_key_list_func(counts, indices, max)
            return {k: sum_counts[k] * max_counts[k] for k in sum_counts.keys()}
        else:
            raise Exception(f'Unkown marginal function: {marginal_func}')
    else:
        num_clbits = len(next(iter(counts)).replace(" ", ""))
        if marginal_func is None or marginal_func == 'sum':
            return np.array([sum(counts.values())])
        elif marginal_func == 'max':
            return np.array([max(counts.values())])
        elif marginal_func == 'min':
            return np.array([min(counts.values())])
        elif marginal_func == 'avg':
            avg_without_zeros = np.average(list(counts.values()))
            avg = len(counts) / (2 ** num_clbits) * avg_without_zeros
            return np.array([avg])
        elif marginal_func == 'avg-squared':
            avg_without_zeros = np.average(list([x * x for x in counts.values()]))
            avg = len(counts) / (2 ** num_clbits) * avg_without_zeros
            return np.array([avg])
        elif marginal_func == 'var':
            return np.array([np.var(list(counts.values()))])
        elif marginal_func == 'sum-squared':
            return np.array([sum([x * x for x in counts.values()])])
        elif marginal_func == 'sum-exp':
            return np.array([sum([np.exp(x) for x in counts.values()])])
        elif marginal_func == 'sum-max':
            return np.array([sum(counts.values()) * max(counts.values())])
        else:
            raise Exception(f'Unkown marginal function: {marginal_func}')


def _marginal_key_list_func(counts, indices, func, zeros=False):
    """Get the marginal max counts for the given set of indices"""
    num_clbits = len(next(iter(counts)).replace(" ", ""))

    # Check if we do not need to marginalize and if so, trim
    # whitespace and '_' and return
    if (indices is None) or set(range(num_clbits)) == set(indices):
        ret = {}
        for key, val in counts.items():
            key = _remove_space_underscore(key)
            ret[key] = val
        return ret

    if not indices or not set(indices).issubset(set(range(num_clbits))):
        raise Exception(f"indices must be in range [0, {num_clbits - 1}].")

    # Sort the indices to keep in descending order
    # Since bitstrings have qubit-0 as least significant bit
    indices = sorted(indices, reverse=True)

    # Build the return list
    new_counts_list = {}
    counter = 0
    for key, val in counts.items():
        new_key = "".join([_remove_space_underscore(key)[-idx - 1] for idx in indices])
        counter += 1
        if new_key not in new_counts_list.keys():
            new_counts_list[new_key] = [val]
        else:
            new_counts_list[new_key].append(val)
    new_counts = {}
    for key, val_list in new_counts_list.items():
        if zeros:
            val_list.extend([0] * (2 ** (num_clbits - len(indices)) - len(val_list)))
        new_counts[key] = func(val_list)
    return new_counts
