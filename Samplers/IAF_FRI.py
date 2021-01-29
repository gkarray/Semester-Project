from Sampler import Status, Sampler
from helpers import getSpikesSignal, getESpline
from plots import plotSignalAndFourier, plotAny, plotSignalAndSpikes, plotSignalAndRecoveredSignal, plotIntegralAndEncoderOutput
import numpy as np
from scipy import special
import scipy

class IAF_FRI(Sampler):
    def __init__(self, threshold, w_0, k_pi, K_channels):
        super().__init__()
        
        self.threshold = threshold
        
        self.w_0 = w_0
            
        self.k_pi = k_pi
        
        if K_channels <= 0:
            raise ValueError('The number of channels must be strictly positive')
            
        self.K_channels = K_channels
        
        
    def setSignal(self, t, u, dt):
        self.t = t
        self.u = u
        self.dt = dt
        
        self.L_support = (2)*(1 + self.k_pi * np.pi / self.w_0)
        self.N_E_spline = int(self.L_support / self.dt)
        
        self.t_E_spline = np.arange(-self.t[-1]/2, self.t[-1]/2 + self.dt, self.dt)
        self.filter = getESpline(self.t_E_spline, self.w_0, self.N_E_spline)
        self.flipped_filter = np.flip(self.filter)
        
        self.status = Status.HAS_SIGNAL
        
    def integrate(self, y, i):
        return y + self.dt * self.filtered_signal[i]
    
    def encode(self, initial_y = 0):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of encode()")
        
        filtered_signal = scipy.convolve(self.u, self.flipped_filter)
        
        filtered_signal = filtered_signal[int(self.t.shape[0]/2): -int(self.t.shape[0]/2)+1]
        
        self.filtered_signal = filtered_signal
        
        y_signal = []
        spikes_idx = []
        q_signs = []
        
        y = initial_y
        
        for i in range(len(self.u)):
            y = self.integrate(y, i)
            y_signal.append(y)
            
            if np.abs(y) >= self.threshold:
                spikes_idx.append(i)
                y_signal.pop()
                y_signal.append(self.threshold * np.sign(y))
                q_signs.append(np.sign(y))
                y = 0
                
        self.y_signal = np.array(y_signal)
        self.spikes_idx = np.array(spikes_idx)
        self.q_signs = np.array(q_signs)
        self.spikes_timings = self.t[self.spikes_idx]
        self.spikes_signal = getSpikesSignal(self.spikes_idx, self.q_signs, self.threshold, self.t.shape[0])
        
        self.status = Status.ENCODED
        
        return self.spikes_signal

    def decode(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of decode()")
            
        if self.spikes_timings.shape[0] < 2:
            raise ValueError('Spikes signal contains less than 2 spikes')
            
        
        self.status = Status.DECODED
        
        raise NotImplementedError()
    
    def summary(self):
        print(80 * f'=')
        print(80 * f'=')
        print(f'Integrate-and-Fire Sampler - FRI')
        print(f'From "TIME ENCODING AND PERFECT RECOVERY OF NON-BANDLIMITED SIGNALS WITH AN INTEGRATE-AND-FIRE SYSTEM"')
        print(f'Roxana Alexandru, Pier Luigi Dragotti, 2019')
        print(f'Status: {self.status.name}')
        print(f'Parameters:')
        print(f"Frequency 'w_0': {self.w_0}")
        print(f"Number of pis 'k_pi': {self.k_pi}")
        print(f"Number of channels 'K_channels': {self.K_channels}")
        if self.status >= Status.HAS_SIGNAL:
            print(f"Support of E-spline filter 'L_support': {self.L_support}")
        print(80 * f'=')
        print(80 * f'=')
        
    def plotSignal(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignal()")
        
        fig_title = 'Signal to be encoded with an Integrate-and-Fire - FRI sampler'
        
        plotAny(self.t, self.u, fig_title, True)
        
    def plotSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSpikes()")
        
        fig_title = 'Spikes resulting from an Integrate-and-Fire - FRI sampler encoding'
        
        plotAny(self.t, self.spikes_signal, fig_title, spikes = True)
        
    def plotSignalAndSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndSpikes()")
        
        fig_title = 'Signal encoded with an Integrate-and-Fire - FRI sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u, self.spikes_timings, fig_title)
        
    def plotRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignal()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - FRI sampler'
        
        plotAny(self.t, self.u_rec, self.dt, fig_title, True)
        
    def plotRecoveredSignalAndSpikes(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignalAndSpikes()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - FRI sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u_rec, self.spikes_timings, fig_title)
        
    def plotSignalAndRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndRecoveredSignal()")
        
        fig_title = 'Comparaison of original signal and recovered signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
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
            
        fig_title = 'Output of the integrator / Filtered input'    
        
        plotIntegralAndEncoderOutput(self.t, self.y_signal, self.filtered_signal, 'f(t)', fig_title)
        
    def plotFilter(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotFilter()")
            
        fig_title = 'Flipped Phi filter'
        
        plotSignalAndFourier(self.t_E_spline, self.flipped_filter, self.dt, fig_title)
        