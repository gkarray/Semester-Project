from enum import IntEnum

class Status(IntEnum):
    EMPTY = 0
    HAS_SIGNAL = 1
    ENCODED = 2
    DECODED = 3

class Sampler:
    def __init__(self):
        self.status = Status.EMPTY
        
    def setSignal(self):
        raise NotImplementedError
        
    def integrate(self):
        raise NotImplementedError
    
    def encode(self):
        raise NotImplementedError
        
    def decode(self):
        raise NotImplementedError
        
    def summary(self):
        raise NotImplementedError
    
    def plotSignal(self):
        raise NotImplementedError
        
    def plotSpikes(self):
        raise NotImplementedError
        
    def plotSignalAndSpikes(self):
        raise NotImplementedError
        
    def plotRecoveredSignal(self):
        raise NotImplementedError
        
    def plotRecoveredSignalAndSpikes(self):
        raise NotImplementedError
        
    def plotSignalAndRecoveredSignal(self):
        raise NotImplementedError
        
    def computeError(self):
        raise NotImplementedError
        
    def computeMeanSquaredError(self):
        raise NotImplementedError
        
    def plotError(self):
        raise NotImplementedError
        
    def getNumberOfSpikes(self):
        raise NotImplementedError
        