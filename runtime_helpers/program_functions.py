import logging

from runtime_helpers.user_messenger_logging import UserMessengerHandler


def activate_logging(user_messenger, log_level, log_modules):
    if len(log_modules) == 0:
        return
    handler = UserMessengerHandler(user_messenger)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    for module in log_modules:
        logging.getLogger(module).setLevel(log_level)
        logging.getLogger(module).addHandler(handler)
