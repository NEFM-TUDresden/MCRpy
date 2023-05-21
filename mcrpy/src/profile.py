import functools

import tensorflow as tf

PROFILE = False

def maybe_trace(name: str):
    def decorate(f: callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if PROFILE:
                with tf.profiler.experimental.Trace(name):
                    result = f(*args, **kwargs)
                return result
            return f(*args, **kwargs)
        return wrapper
    return decorate

def maybe_profile(logdir: str):
    def decorate(f: callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if PROFILE:
                tf.profiler.experimental.start(logdir)
                result = f(*args, **kwargs)
                tf.profiler.experimental.stop()
                return result
            return f(*args, **kwargs)
        return wrapper
    return decorate