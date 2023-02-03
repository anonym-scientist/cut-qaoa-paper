import os
from typing import Optional

from qiskit import IBMQ
from qiskit.providers import QiskitBackendNotFoundError
from qiskit.providers.ibmq import IBMQProviderError
from qiskit.providers.ibmq.credentials import read_credentials_from_qiskitrc

DEFAULT_QISKITRC_FILE_EHNINGEN = os.path.join(os.path.expanduser("~"), '.qiskit', 'qiskitrc_ehningen')


def load_credentials(filename: Optional[str] = None):
    if filename is None:
        filename = DEFAULT_QISKITRC_FILE_EHNINGEN
    credentials, preferences = read_credentials_from_qiskitrc(filename)
    credentials_list = list(credentials.values())
    IBMQ._initialize_providers(credentials_list[0])


def get_provider(hub='fraunhofer-de', group='fhg-all', project='estu04'):
    return IBMQ.get_provider(hub=hub, group=group, project=project)


def get_backend_and_provider(backend_name):
    if backend_name == 'ibmq_ehningen':
        load_credentials()
        provider = IBMQ.get_provider(hub='fraunhofer-de', group='fhg-all', project='estu04')
        backend = provider.get_backend(backend_name)
    else:
        IBMQ.load_account()
        try:
            provider = IBMQ.get_provider(hub='ibm-q-fraunhofer', group='fhg-all', project='estu04')
            backend = provider.get_backend(backend_name)
        except (IBMQProviderError, QiskitBackendNotFoundError):
            provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
            backend = provider.get_backend(backend_name)
    return backend, provider


def get_backend(backend_name):
    return get_backend_and_provider(backend_name)[0]
