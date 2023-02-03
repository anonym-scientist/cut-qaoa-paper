from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from circuit_cutting.util import keys_to_ints


@dataclass
class CountsArray:
    __slots__ = ['data', 'n_bits']

    data: np.ndarray
    n_bits: int

    # qiskit runtime needs explicit constructor
    def __init__(self, data, n_bits):
        self.data = data
        self.n_bits = n_bits

    @classmethod
    def from_dict(cls, counts, n_bits):
        try:
            first_key = next(iter(counts))
        except StopIteration:
            return cls(np.array([]), n_bits)
        if isinstance(first_key, str):
            return cls(np.array(list(keys_to_ints(counts).items())), n_bits)
        elif isinstance(first_key, np.int):
            return cls(np.array(list(counts.items())), n_bits)
        else:
            raise Exception(f'Type of key {type(first_key)} is unsupported')

    def as_array(self):
        array = np.zeros(2 ** self.n_bits)
        for idx, val in self.data:
            array[idx] = val
        return array


def filter_and_marginalize(counts_array, fixed_bits, marg_indices, marg_func='sum'):
    filtered_data = _filter(counts_array.data, fixed_bits)
    if len(filtered_data) == 0:
        return CountsArray(filtered_data, len(marg_indices))
    return _marginalize(filtered_data, marg_indices, marg_func, to_array=False)


def filter_and_marginalize_return_array(counts_array, fixed_bits, marg_indices, marg_func='sum'):
    filtered_data = _filter(counts_array.data, fixed_bits)
    if len(filtered_data) == 0:
        return np.zeros(2 ** len(marg_indices))
    return _marginalize(filtered_data, marg_indices, marg_func, to_array=True)


def filter_counts_array(counts_array, fixed_bits):
    filtered_data = _filter(counts_array.data, fixed_bits)
    return CountsArray(filtered_data, counts_array.n_bits)


def marginalize(counts_array, indices, func='sum'):
    return _marginalize(counts_array.data, indices, func, to_array=False)


def _filter(data, fixed_bits):
    if fixed_bits is None or len(fixed_bits) == 0 or len(data) == 0:
        return data
    mask, mask_value = _generate_masks(fixed_bits)
    return data[data[:, 0] & mask == mask_value]


def _marginalize_bits(num, sorted_indices):
    res_num = 0
    for n, idx in enumerate(sorted_indices):
        bit = num >> idx & 1
        res_num += bit << n
    return res_num


def _marginalize_with_func(data, sorted_indices, func_callable, to_array):
    count_lists = defaultdict(list)
    for idx, val in data.tolist():
        new_idx = _marginalize_bits(idx, sorted_indices)
        count_lists[new_idx].append(val)
    if to_array:
        array = np.zeros(2 ** len(sorted_indices))
        for idx, val_list in count_lists.items():
            array[idx] = func_callable(val_list)
        return array
    else:
        array = np.zeros(len(count_lists), 2)
        for i, (idx, val_list) in enumerate(count_lists.items()):
            array[i][0] = idx
            array[i][1] = func_callable(val_list)
        return CountsArray(array, len(sorted_indices))


def _marginalize(data, sorted_indices, func, to_array):
    if len(sorted_indices) > 0:
        if func == 'sum':
            func_callable = sum
        elif func == 'max':
            func_callable = max
        elif func == 'min':
            func_callable = min
        else:
            raise Exception(f'Unkown marginal function: {func}')
        return _marginalize_with_func(data, sorted_indices, func_callable, to_array)
    else:
        if func == 'sum':
            func_callable = np.sum
        elif func == 'max':
            func_callable = np.max
        elif func == 'min':
            func_callable = np.min
        else:
            raise Exception(f'Unkown marginal function: {func}')
        if to_array:
            return np.array([func_callable(data[:, 1])])
        else:
            return CountsArray(np.array([[0, func_callable(data[:, 1])]]), 0)


def _generate_masks(bitstring_filter):
    mask = 0
    mask_value = 0
    for i, bit in bitstring_filter.items():
        mask += 1 << i
        mask_value += bit << i
    return mask, mask_value
