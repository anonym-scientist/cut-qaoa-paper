import itertools
import logging
from typing import Union, Tuple, List, Dict

import numpy as np
from qiskit import *
from qiskit.circuit.library import *

logger = logging.getLogger(__name__)

GATE_VERSIONS = ["I", "Z", "RZ_plus", "RZ_minus", "MEAS"]
GATE_VERSIONS_REDUCED = ["I", "Z", "RZ_plus", "MEAS"]


def _get_partition_idx(qubit, partitions):
    for idx, part in enumerate(partitions):
        if qubit < part[0]:
            return idx - 1
    if qubit <= partitions[-1][1]:
        return len(partitions) - 1


def split(circuit: QuantumCircuit, fragment_widths: list[int], reduced: bool = False) -> Tuple[dict, dict]:
    cut = _split(circuit, fragment_widths)
    return generate_sub_circuits(cut, reduced)


def _split(circuit: QuantumCircuit, fragment_widths: List[int]) -> List[Dict[str, Union[int, list]]]:
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
