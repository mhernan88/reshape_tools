from logging import Logger, getLogger, DEBUG, INFO, WARNING, ERROR
from typing import Optional

def fmt(msg, *args, **kwargs):
    args_str = None
    if args is not None:
        if len(args) > 0:
            args_str = ", ".join(args)
            args_str = f"args: {args_str}"
        
    
    kwargs_str = None
    if kwargs is not None:
        if len(list(kwargs.keys())) > 0:
            kwargs_list = []
            for k, v in kwargs.items():
                kwargs_list.append(f"{k}={v}")
            kwargs_str = ", ".join(kwargs_list)
            kwargs_str = f"kwawrgs: {kwargs_str}"

    if args_str is not None:
        msg += f"; {args_str}"
    if kwargs_str is not None:
        msg += f"; {kwargs_str}"
    return msg

class CustomLogger(Logger):
    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, fmt(msg, *args, **kwargs), tuple())

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(INFO):
            self._log(INFO, fmt(msg, *args, **kwargs), tuple())

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(WARNING):
            self._log(WARNING, fmt(msg, *args, **kwargs), tuple())

    def error(self, msg, *args, **kwargs):
        if self.isEnabledFor(ERROR):
            self._log(ERROR, fmt(msg, *args, **kwargs), tuple())
