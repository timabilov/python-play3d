import functools
import os
import time
from functools import wraps
import logging

log = logging.getLogger(__name__)


def logtime(fn, *args, **kwargs):
    message = kwargs.pop('message', None)
    if not message:
        message = str(fn)
    t = time.time()
    r = fn(*args, **kwargs)
    td = time.time() - t  # 0.004 ms
    if 1e6 * td > 0.1:
        log.debug(" %s.%s %s %0.6f seconds",
                  os.path.basename(fn.__code__.co_filename).split('.')[0],
                  fn.__name__, message, td)

    return r


def log_this(message=None, *targs):

    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            args_log = ' ('
            nonlocal targs
            if not targs:
                targs = [str] * len(args)

            for arg_fn, arg in zip(targs, args):
                args_log += str(arg_fn(arg)) + ','
            args_log += ')'

            return logtime(fn, *args, **kwargs, message=(message or '') + args_log)

        return wrapper
    return decorator


def _get_fps(last_frame_time):

    delta = time.time() - last_frame_time

    return round(1/delta, 2)


def capture_fps(fn):

    @functools.wraps(fn)
    def wrapper():
        start_time = time.time()
        fn()

        log.info(' %s.%s() %s FPS',
                 os.path.basename(fn.__code__.co_filename).split('.')[0],
                 wrapper.__name__, _get_fps(start_time))

    return wrapper