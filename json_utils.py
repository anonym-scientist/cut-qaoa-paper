import json
import datetime

import numpy as np
import qiskit.result


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


def store_kwargs_as_json(path: str, name: str, **kwargs):
    with open(f'{path}/{name}.json', "w") as outfile:
        json.dump(kwargs, outfile, indent=4, cls=NumpyEncoder)


def store_as_json(path: str, name: str, data):
    with open(f'{path}/{name}.json', "w") as outfile:
        json.dump(data, outfile, indent=4, cls=NumpyEncoder)
