from Sampler import Status, Sampler
from helpers import getSpikesSignal, closedPhiFromRcos
from plots import plotSignalAndFourier, plotAny, plotSignalAndSpikes, plotSignalAndRecoveredSignal, plotIntegralAndEncoderOutput
import numpy as np
from scipy import special

class IAF_KernelBased(Sampler):
    def __init__(self, alpha, theta, gamma, Ts):
        super().__init__()
        
        if alpha <= 0:
            raise ValueError('The alpha parameter must be strictly positive')
        
        self.alpha = alpha
        
        if theta <= 0:
            raise ValueError('The theta parameter must be strictly positive')
            
        self.theta = theta
        
        if gamma < 0 or gamma > 1:
            raise ValueError('The gamma parameter for the psi_kernel must be strictly positive')
            
        self.gamma = gamma
        
        if Ts <= 0:
            raise ValueError('The Ts parameter must be strictly positive')
            
        self.Ts = Ts
        
    def setSignal(self, t, u, dt):
        self.t = t
        self.u = u
        self.dt = dt
        
        self.t_phi_kernel = np.arange(-self.t[-1]/2, self.t[-1]/2 + self.dt, self.dt)
        self.phi_kernel = closedPhiFromRcos(self.t_phi_kernel, self.gamma, self.Ts, self.alpha)
        
        self.status = Status.HAS_SIGNAL
        
    def integrate(self, y, i, tj):
        return y + self.dt * self.u[i] * np.exp((tj - self.t[i]) / self.alpha)
    
    def encode(self, initial_y = 0, initial_tj = 0):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of encode()")
        
        y_signal = []
        spikes_idx = []
        q_signs = []
        
        y = initial_y
        tj = initial_tj
        
        for i in range(len(self.u)):
            y = self.integrate(y, i, tj)
            y_signal.append(y)
            
            if np.abs(y) >= self.theta:
                spikes_idx.append(i)
                tj = self.t[i]
                q_signs.append(np.sign(y))
                y = 0
                
        self.y_signal = np.array(y_signal)
        self.spikes_idx = np.array(spikes_idx)
        self.q_signs = np.array(q_signs)
        self.spikes_timings = self.t[self.spikes_idx]
        self.spikes_signal = getSpikesSignal(self.spikes_idx, self.q_signs, self.theta, self.t.shape[0])
        
        self.status = Status.ENCODED
        
        return self.spikes_signal

    def decode(self, initial_sgn = 1):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of decode()")
            
        if self.spikes_timings.shape[0] < 2:
            raise ValueError('Spikes signal contains less than 2 spikes')
            
        scaled_qs = self.q_signs * self.theta

        # Wi
        # To be optimized
        ws = []
        ws.append(0)
        for idx, value in enumerate(scaled_qs):
            if(idx == 0):
                continue

            w = np.exp(self.alpha * (self.t[int(self.spikes_idx[idx-1])] - self.t[int(self.spikes_idx[idx])])) * ws[idx-1] + value
            ws.append(w)

        # Sk
        # To be optimized
        sk = []
        nb = 0
        for idx, value in enumerate(self.t):
            if(nb == 0):
                s = 0
            else:
                s = np.exp(self.alpha * (self.t[int(self.spikes_idx[nb])] - value)) * ws[nb]

            if(nb + 1 < self.spikes_idx.shape[0]):
                if(idx == int(self.spikes_idx[nb+1])):
                    nb = nb+1

            sk.append(s)

        # Applying the right filter
        # To be optimized
        convolved = np.convolve(self.phi_kernel, sk)
        filtered = convolved[int(self.t.shape[0] / 2): - int(self.t.shape[0] / 2) + 1]
            
        self.u_rec = filtered
        
        self.status = Status.DECODED
        
        return self.u_rec
    
    def summary(self):
        print(80 * f'=')
        print(80 * f'=')
        print(f'Integrate-and-Fire Sampler - KernelBased')
        print(f'From "APPROXIMATE RECONSTRUCTION OF BANDLIMITED FUNCTIONS FOR THE INTEGRATE AND FIRE SAMPLER"')
        print(f'Hans G. Feichtinger, 2009')
        print(f'Status: {self.status.name}')
        print(f'Parameters:')
        print(f"Firing parameter 'alpha': {self.alpha}")
        print(f"Threshold 'theta': {self.theta}")
        print(80 * f'=')
        print(80 * f'=')
        
    def plotSignal(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignal()")
        
        fig_title = 'Signal to be encoded with an Integrate-and-Fire - KernelBased sampler'
        
        plotSignalAndFourier(self.t, self.u, self.dt, fig_title)
        
    def plotSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSpikes()")
        
        fig_title = 'Spikes resulting from an Integrate-and-Fire - KernelBased sampler encoding'
        
        plotAny(self.t, self.spikes_signal, fig_title, spikes = True)
        
    def plotSignalAndSpikes(self):
        if self.status < Status.ENCODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotSignalAndSpikes()")
        
        fig_title = 'Signal encoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
        plotSignalAndSpikes(self.t, self.u, self.spikes_timings, fig_title)
        
    def plotRecoveredSignal(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignal()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler'
        
        plotSignalAndFourier(self.t, self.u_rec, self.dt, fig_title)
        
    def plotRecoveredSignalAndSpikes(self):
        if self.status < Status.DECODED:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotRecoveredSignalAndSpikes()")
        
        fig_title = 'Signal encoded then decoded with an Integrate-and-Fire - KernelBased sampler (with spikes)'
        
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
            
        fig_title = 'Output of the integrator / Output of the encoder'    
        
        plotIntegralAndEncoderOutput(self.t, self.y_signal, self.spikes_signal, 'q(t)', fig_title)
        
    def plotPhiKernel(self):
        if self.status < Status.HAS_SIGNAL:
            raise ValueError(f"The current status {self.status.name} doesn't allow the use of plotPhiKernel()")
            
        fig_title = 'Phi kernel'
        
        plotSignalAndFourier(self.t_phi_kernel, self.phi_kernel, self.dt, fig_title)
        