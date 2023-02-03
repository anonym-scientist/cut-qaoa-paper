import json
from logging import LogRecord


class LogRecordEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, LogRecord):
            obj_dict = obj.__dict__
            obj_dict['__log_record__'] = True
            return obj_dict
        return json.JSONEncoder.default(self, obj)
