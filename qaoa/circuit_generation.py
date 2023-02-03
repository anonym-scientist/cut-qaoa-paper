from typing import List, Optional

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


def create_qaoa_circ(graph: nx.Graph, theta: List[float], sub_graph_nodes: Optional[List[int]] = None,
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
