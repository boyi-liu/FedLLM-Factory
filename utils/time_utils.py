import time
from functools import wraps


def time_record(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        self.training_time = (end_time - start_time) * self.delay
        return result
    return wrapper