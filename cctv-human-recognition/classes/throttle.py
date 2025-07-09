import time

class Throttle:
    def __init__(self, interval_secs):
        self.interval_secs = interval_secs
        self.last_call_time = 0
        self.last_result = None

    def call(self, func, *args, **kwargs):
        current_time = time.time()
        if current_time - self.last_call_time > self.interval_secs:
            self.last_result = func(*args, **kwargs)
            self.last_call_time = current_time
        return self.last_result
