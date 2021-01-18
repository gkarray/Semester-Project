from Sampler import Status, Sampler
from helpers import getSpikesSignal
from plots import plotSignalAndFourier, plotAny, plotSignalAndSpikes, plotSignalAndRecoveredSignal, plotIntegralAndEncoderOutput
import numpy as np
from scipy import special

__pinv_rcond__ = 1e-8

class IAF_ASDM(Sampler):
    def __init__(self, bias, threshold, k_constant):
        super().__init__()
        
        if bias <= 0:
            raise ValueError('The bias must be strictly positive')
        
        self.bias = bias
        
        if threshold <= 0:
            raise ValueError('The threshold must be strictly positive')
            
        self.threshold = threshold    
            
        if k_constant <= 0:
            raise ValueError('The k_constant must be strictly positive')
            
        self.k_constant = k_constant
        
    def setSignal(self, t, u, dt, bw):
        self.t = t
        self.u = u
        self.dt = dt
        self.bw = bw
        self.status = Status.HAS_SIGNAL
        
    def integrate(self, y, sgn, i):
        return y + (self.dt * (sgn * self.bias + self.u[i]) / self.k_constant)
    
    def encode(self, initial_y = 0, initial_sgn = -1):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of encode()")
        
        interval_lengths = []
        y_signal = []
        z_signal = []
        spikes_idx = []
        
        interval = 0
        y = initial_y
        sgn = initial_sgn
        
        for i in range(len(self.u)):
            y = self.integrate(y, sgn, i)
            
            interval += self.dt
            
            z_signal.append(sgn * self.bias)
            
            if np.abs(y) >= self.threshold:
                interval_lengths.append(interval)
                interval = 0
                y = self.threshold * sgn
                sgn = - sgn
                spikes_idx.append(i)
                
            y_signal.append(y)

                
        self.interval_lengths = np.array(interval_lengths)
        self.y_signal = np.array(y_signal)
        self.z_signal = np.array(z_signal)
        self.spikes_idx = np.array(spikes_idx)
        self.spikes_timings = self.t[self.spikes_idx]
        self.spikes_signal = getSpikesSignal(self.spikes_idx, np.ones(self.spikes_idx.shape[0]), self.bias, self.t.shape[0])
        
        self.status = Status.ENCODED
        
        return self.spikes_signal

    def decode(self, initial_sgn = 1):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of decode()")
            
        if self.spikes_timings.shape[0] < 2:
            raise ValueError('Spikes signal contains less than 2 spikes')
            
        tsh = (self.spikes_timings[0:-1] + self.spikes_timings[1:]) / 2
        Nsh = len(tsh)

        bwpi = self.bw / np.pi

        # Compute G matrix:
        G = np.empty((Nsh, Nsh), np.float)
        
        for j in range(Nsh):
            # Compute the values for all of the sincs so that they do not
            # need to each be recomputed when determining the integrals
            # between spike times:
            temp = special.sici(self.bw * (self.spikes_timings - tsh[j]))[0] / np.pi
            G[:, j] = temp[1:]-temp[:-1]
            
        G_inv = np.linalg.pinv(G, __pinv_rcond__)

        # Compute quanta:
        if initial_sgn == -1:
            q = (-1) ** np.arange(1, Nsh + 1) * (2 * self.k_constant * self.threshold - self.bias * self.interval_lengths[1:])
        else:
            q = (-1) ** np.arange(0, Nsh) * (2 * self.k_constant * self.threshold - self.bias * self.interval_lengths[1:])

        # Reconstruct signal by adding up the weighted sinc functions. The
        # weighted sinc functions are computed on the fly here to save
        # memory:
        u_rec = np.zeros(len(self.t), np.float)
        
        c = np.dot(G_inv, q)
        
        for i in range(Nsh):
            u_rec += np.sinc(bwpi * (self.t - tsh[i])) * bwpi * c[i]
            
        self.u_rec = u_rec
        
        self.status = Status.DECODED
        
        return self.u_rec
    
    def summary(self):
        print(80 * f'=')
        print(80 * f'=')
        print(f'Integrate-and-Fire Sampler - ASDM')
        print(f'From "TIME ENCODING AND PERFECT RECOVERY OF BANDLIMITED SIGNALS"')
        print(f'Aurel A. Lazar, Laszlo T. Toth, 2004')
        print(f'Status: {self.status.name}')
        print(f'Parameters:')
        print(f"Bias 'b': {self.bias}")
        print(f"Threshold 'delta': {self.threshold}")
        print(f"Integretor constant 'k': {self.k_constant}")
        print(80 * f'=')
        print(80 * f'=')
        
    def plotSignal(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignal()")
        
        fig_title = 'Signal to be encoded with an Integrate-and-Fire - ASDM sampler'
        
        plotSignalAndFourier(self.t, self.u, self.dt, fig_title)
        
    def plotSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSpikes()")
        
        fig_title = 'Spikes resulting from an Integrate-and-Fire - ASDM sampler encoding'
        
        plotAny(self.t, self.spikes_signal, fig_title, spikes = True)
        
    def plotSignalAndSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndSpikes()")
        
        fig_title = 'Signal encoded with an Integrate-and-Fire - ASDM sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u, self.spikes_timings, fig_title)
        
    def plotRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignal()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - ASDM sampler'
        
        plotSignalAndFourier(self.t, self.u_rec, self.dt, fig_title)
        
    def plotRecoveredSignalAndSpikes(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignalAndSpikes()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - ASDM sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u_rec, self.spikes_timings, fig_title)
        
    def plotSignalAndRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndRecoveredSignal()")
        
        fig_title = 'Comparaison of original signal and recovered signal encoded then decoded with an Integrate-and-Fire - ASDM sampler (with spikes)'
        
        plotSignalAndRecoveredSignal(self.t, self.u, self.u_rec, fig_title)
        
    def computeError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of computeError()")
        
        return self.u - self.u_rec
    
    def computeMeanSquaredError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of computeMeanSquaredError()")
        
        return np.sum(self.computeError() ** 2)
    
    def plotError(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotError()")
        
        fig_title = 'Error between the reconstructed signal and the original signal'
        
        plotAny(self.t, self.computeError(), fig_title)
        
    def getNumberOfSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of getNumberOfSpikes()")
        
        return self.spikes_timings.shape[0]
    
    def plotIntegralAndEncoderOutput(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotIntegralAndEncoderOutput()")
            
        fig_title = 'Output of the integrator / Output of the encoder'    
        
        plotIntegralAndEncoderOutput(self.t, self.y_signal, self.z_signal, 'z(t)', fig_title)
        
    
        
        
        
        
        
        