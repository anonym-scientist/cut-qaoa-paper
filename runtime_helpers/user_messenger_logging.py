import json
from logging import Handler, LogRecord

from qiskit.providers.ibmq.runtime import UserMessenger

from runtime_helpers.json_logs import LogRecordEncoder


class UserMessengerHandler(Handler):

    def __init__(self, user_messenger: UserMessenger):
        super().__init__()
        self._user_messenger = user_messenger

    def emit(self, record: LogRecord) -> None:
        self._user_messenger.publish(json.dumps(record, cls=LogRecordEncoder))
