class Generator:
    def __init__(self, dur, dt):
        self.dur = dur
        self.dt = dt
        
    def generate(self):
        raise NotImplementedError