import numpy as np

class Generator:
    def __init__(self, dur, dt):
        self.dur = dur
        self.dt = dt
        self.t = np.arange(0, self.dur, self.dt)
        
    def generate(self):
        raise NotImplementedError