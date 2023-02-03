import copy
import json
import logging
import time
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
from qiskit import *
from qiskit.providers import JobError
from qiskit.providers.aer import QasmSimulator, AerSimulator
from qiskit.providers.ibmq import IBMQBackend, IBMQBackendJobLimitError
from qiskit.providers.ibmq.job import IBMQJobError
from qiskit.result import Result
from qiskit.utils.measurement_error_mitigation import get_measured_qubits, build_measurement_error_mitigation_circuits
from qiskit.utils.mitigation import CompleteMeasFitter, TensoredMeasFitter

from json_utils import QiskitResultEncoder, as_qiskit_result

logger = logging.getLogger(__name__)


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
        if mitigators is not None and isinstance(mitigators, Iterable):
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


def get_fitter_cls(mitigator: str):
    if mitigator == 'CompleteMeasFitter':
        return CompleteMeasFitter
    elif mitigator == 'TensoredMeasFitter':
        return TensoredMeasFitter
    else:
        raise ValueError(f'Unknown mitigator: {mitigator}')
