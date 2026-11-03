"""Signal handlers only request a stop; checkpointing happens at safe boundaries."""
import signal
from contextlib import AbstractContextManager

class StopRequested(AbstractContextManager):
    def __init__(self): self.requested = False; self.signal = None; self.previous = {}
    def _handler(self, number, frame): self.requested = True; self.signal = number
    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self._handler)
        return self
    def __exit__(self,*_):
        for number, handler in self.previous.items(): signal.signal(number, handler)
