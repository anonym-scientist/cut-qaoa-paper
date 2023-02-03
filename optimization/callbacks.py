import json
import time

from json_utils import NumpyEncoder


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


def get_spsa_callback(logger, optimization_path=None):
    callback_counter = 0
    callback_time = time.time()
    if optimization_path is None:
        optimization_path = []

    def _callback(n_eval, parameters, value, stepsize, accepted):
        nonlocal callback_counter
        nonlocal callback_time
        logger.info(json.dumps(
            {'iteration': callback_counter, 'duration': time.time() - callback_time, 'params': parameters,
             'value': value, 'n_eval': n_eval, 'stepsize': stepsize, 'accepted': accepted},
            cls=NumpyEncoder))
        optimization_path.append(parameters)
        callback_time = time.time()
        callback_counter += 1

    return _callback


def get_callback(method, logger, optimization_path=None):
    pass