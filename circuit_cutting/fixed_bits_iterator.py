class FixedBitsIterator:

    def __init__(self, fixed_bits, n_bits):
        self._fixed_bits = fixed_bits
        self._n_bits = n_bits
        self._counter = 0
        self._n_iterations = 2 ** (n_bits - len(fixed_bits))
        self._fixed_offset = sum([2 ** (n_bits - idx - 1) for idx, bit in fixed_bits.items() if bit == '1'])
        self._format_str = '{0:0' + str(n_bits) + 'b}'

    def _calc_binary_value(self, value):
        binary_value = 0
        for bit_index in range(self._n_bits):
            if bit_index not in self._fixed_bits.keys():
                if value % 2 == 1:
                    binary_value += 2 ** (self._n_bits - bit_index - 1)
                value //= 2
        return binary_value

    def __iter__(self):
        return self

    def __next__(self):
        if self._counter < self._n_iterations:
            return_val = self._calc_binary_value(self._counter) + self._fixed_offset
            self._counter += 1
            return self._format_str.format(return_val)
        else:
            raise StopIteration
