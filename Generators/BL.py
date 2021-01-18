from Generator import Generator
from numpy import ceil, zeros, array, exp, pi
from numpy.random import randint, rand
from numpy.fft import irfft


class BL(Generator):
    def __init__(self, dur, dt, f_max, N_c, min_diff = None):
        super().__init__(dur, dt)
        
        # The maximum frequency may not exceed the Nyquist frequency:
        fs = 1.0/dt
        
        if f_max > fs/2:
            raise ValueError("Maximum frequency may not exceed the Nyquist frequency")
            
        self.f_max = f_max
            
        N = int(ceil(dur/dt))
        
        f_max_i = int(N*f_max/fs)
        
        if f_max_i < N_c:
            raise ValueError(f"Maximum frequency {f_max} is too low to provide {N_c} frequency components")
            
        self.N_c = N_c
            
        if min_diff != None:
            min_diff_i = int(N*min_diff/fs)
            
            if (f_max_i/N_c) + (min_diff_i/2) >= (2*f_max_i/N_c) - (min_diff_i/2):
                raise ValueError(f"Maximum frequency {f_max} is too low to provide {N_c} frequency components with minimum difference equal to {min_diff}")
        
        self.min_diff = min_diff
        
    def generate(self):
        fs = 1.0/self.dt
        
        N = int(ceil(self.dur/self.dt))
        
        f = zeros(int(N/2)+1, complex)
        
        f_max_i = int(N*self.f_max/fs)
        
        ci = set()
        
        if self.min_diff != None:
            min_diff_i = int(N*self.min_diff/fs)
            
            f_wind_min = 1
            f_wind_max = (f_max_i/self.N_c) - (min_diff_i/2)
            
            while len(ci) < self.N_c:
                temp = randint(f_wind_min, f_wind_max)
                ci.add(temp)
                
                f_wind_min = f_wind_max + min_diff_i
                f_wind_max = f_wind_max + (f_max_i/self.N_c)
                if f_wind_max + min_diff_i > f_max_i:
                    f_wind_max = f_max_i
            
        else:
            while len(ci) < self.N_c:
                temp = randint(1, f_max_i+1)
                while temp in ci:
                    temp = randint(1, f_max_i+1)
                ci.add(temp)
        
        ci = array(list(ci))
        p = -2*pi*rand(self.N_c)
        f[ci] = (N/2)*exp(1j*p)
        
        u = irfft(f,N)
        
        return self.t, u