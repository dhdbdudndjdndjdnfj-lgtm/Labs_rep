import numpy as np
import math

class linear_regression():
    def __init__(self, delta_x = None, delta_y = None, x = None,y = None):
        self.x_array = np.asarray(x, dtype=float) if x is not None else None
        self.y_array = np.asarray(y, dtype=float) if y is not None else None
        self.delta_x = np.asarray(delta_x, dtype=float) if delta_x is not None else None
        self.delta_y = np.asarray(delta_y, dtype=float) if delta_y is not None else None
        
        if self.x_array is None or self.y_array is None:
            raise ValueError('x и y должны быть заданы')
        if np.size(self.x_array) != np.size(self.y_array):
            raise ValueError('x и y должны быть одной длины')
        if np.size(self.x_array) < 2:
            raise ValueError('нужно минимум 2 точки')
        if self.delta_x is not None:
            if np.size(self.delta_x) not in (1, np.size(self.x_array)):
                raise ValueError('delta_x: одно число или массив длины x')
            if np.size(self.delta_y) not in (1, np.size(self.y_array)):
                raise ValueError('delta_y: одно число или массив длины y')

        self.k = self.b = self.r2 = None
        self.delta_k = self.delta_b = None
        
    def error_rate_m_k (self, Nsim = 1000, seed = 42):
        rng = np.random.default_rng(seed)
        if not np.ndim(self.delta_x):
            delta_x = np.full(len(self.x_array), float(self.delta_x))
        else:
            delta_x = self.delta_x
        if not np.ndim(self.delta_y):
            delta_y = np.full(len(self.y_array), float(self.delta_y))
        else:
            delta_y = self.delta_y
            
        x_array, y_array = self.x_array, self.y_array
        N = len(x_array)
        Ks = np.empty(Nsim)
        Bs = np.empty(Nsim)
        
        for i in range(Nsim):
            xs = rng.normal(x_array, delta_x)
            ys = rng.normal(y_array, delta_y)
            Ks[i] = ((N*(xs*ys).sum() - xs.sum()*ys.sum())/(N*(xs**2).sum() - (xs).sum()**2))
            Bs[i] = ((ys.sum()*(xs**2).sum() - xs.sum()*(xs*ys).sum())/(N*(xs**2).sum() - (xs).sum()**2))
        
        self.delta_k = np.sqrt(((1/(Nsim - 1))*(Ks - Ks.mean())**2).sum())   
        self.delta_b = np.sqrt(((1/(Nsim - 1))*(Bs - Bs.mean())**2).sum())  
    
    def error_rate_s_c(self):
        x_array, y_array = self.x_array, self.y_array
        N = len(x_array)
        SS = (((y_array - self.k*x_array - self.b)**2).sum())/(N - 2)
        Sxx = ((x_array - x_array.mean())**2).sum()
        
        self.delta_k = np.sqrt(SS/Sxx)
        self.delta_b = np.sqrt(SS*(1/N + x_array.mean()**2/Sxx))
    
    def selector(self):
        if self.delta_x is not None and self.delta_y is not None:
            return self.error_rate_m_k()
        else:
            return self.error_rate_s_c()
        
    def format(self):
        if self.delta_k == 0:
            delta_k = 0
            k = self.k
            exp_k = -3
        else:
            exp_k = math.floor(math.log10(abs(self.delta_k)))
            delta_k = round(self.delta_k, -exp_k)
            k = round(self.k, -exp_k)
        if self.delta_b == 0:
            delta_b = 0
            b = self.b
            exp_b = -3
        else:
            exp_b = math.floor(math.log10(abs(self.delta_b)))
            delta_b = round(self.delta_b, -exp_b)
            b = round(self.b, -exp_b)
        return (max(-exp_k, 0),  max(-exp_b, 0),
                delta_k, delta_b, k, b)
        
    def analyzis(self):
        if len(self.x_array) < 2:
            raise ValueError('Нужно хотя бы два числа')
        x, y, N = self.x_array, self.y_array, len(self.x_array)
        
        k = (N*(x*y).sum() - x.sum()*y.sum())/(N*(x**2).sum() - (x).sum()**2)
        self.k = k
        b = (y.sum()*(x**2).sum() - x.sum()*(x*y).sum())/(N*(x**2).sum() - (x).sum()**2)
        self.b = b
        
        SS_tot = ((y - y.mean())**2).sum()
        SS_res = ((y - k*x - b)**2).sum()
        
        r2 = 1 - SS_res/SS_tot
        self.r2 = r2
        
        self.selector()
        exp_k, exp_b, delta_k, delta_b, k ,b = self.format()
        
        return {
            'k': k,
            'b': b,
            'r2': r2,
            'delta_k': delta_k,
            'delta_b': delta_b,
            'exp_k': exp_k,
            'exp_b': exp_b
        }
        
class approximation():
    def __init__(self, xarr = None, yarr = None, xerr = None, yerr = None):
        self.xarr = np.asarray(xarr, dtype = float) if xarr is not None else None
        self.yarr = np.asarray(yarr, dtype = float) if yarr is not None else None
        self.xerr = np.asarray(xerr, dtype = float) if xerr is not None else None
        self.yerr = np.asarray(yerr, dtype = float) if yerr is not None else None
        
        if self.xarr is None or self.yarr is None:
            raise ValueError('Введи значения')
        if np.size(self.xarr) != np.size(self.yarr):
            raise ValueError('Введи значения по-человечески')
        if np.size(self.xarr) < 2:
            raise ValueError('Введи больше одной точки')
        
        if self.xerr is None or self.yerr is None:
            raise ValueError('Введи погрешности')
        if np.size(self.xerr) not in (1,np.size(self.xarr)):
            raise ValueError('Одно число или массив из всех')
        if np.size(self.yerr) not in (1,np.size(self.yarr)):
            raise ValueError('Одно число или массив из всех')
        
        self.n = len(self.xarr)
        self.I = np.eye(self.n)
        self.K = None
        self.f = None

    def build_L(self):
        x = self.xarr
        n = self.n
        L = np.zeros((n-2,n))
        
        for i in range(self.n - 2):
            h1 = x[i + 1] - x[i]
            h2 = x[i + 2] - x[i + 1]
            denom = 1/(h1 + h2)
            
            L[i, i] = 2.0*denom*(1/h1)
            L[i, i + 1] = -2.0*denom*(1/h1 + 1/h2)
            L[i, i + 2] = 2.0*denom*(1/h2)
            
        return L
        
    def build_W(self):
        x = self.xarr
        n = self.n
        w = np.empty(n - 2)
        
        for i in range(n - 2):
            h1 = x[i + 1] - x[i]
            h2 = x[i + 2] - x[i + 1]
            w[i] = 1/2*(h1 + h2)
            
        return np.diag(w)
            
    def build_K(self):
        if self.K is not None:
            return self.K
        L = self.build_L()
        W = self.build_W()
        self.K = L.T@W@L 
        
        return self.K
    
    def smooth_approximation(self, lamda = 0.5):
        if self.K is None:
            self.build_K()
        
        w = 1/self.yerr**2
        W = np.diag(w)
        A = W + lamda*self.K
        f = np.linalg.solve(A,W@self.yarr)
        self.f = f
        return f
    
    def predict(self, x_new):
        from scipy.interpolate import CubicSpline
        if self.f is None:
            raise ValueError('Глупень')
        x_new = np.asarray(x_new, dtype = float)
        cs = CubicSpline(self.xarr, self.f)
        return cs(x_new)
    
    def __call__(self, x_new):
        return self.predict(x_new)
        
class formulas_labrary():
    def __init__(self, xarr = None):
        self.xarr = np.asarray(xarr, dtype = float) if xarr is not None else None
        if self.xarr is None:
            raise ValueError('Ошибкер')
        if np.size(self.xarr) < 2:
            raise ValueError('Хотя бы два числа')
        
        self.raw_delta = None 
        self.av_delta = None
        self.true_delta = None
        
    def format(self):
        import math
        av = self.xarr.mean()

        if self.true_delta is None:
            self.SEM()

        if self.true_delta == 0:
            exp = -3
            av = round(av, 3)
            true_delta = 0
            return 3, av, true_delta

        if self.true_delta >= 1:
            exp = 0
            true_delta = round(self.true_delta, 0)
            av = round(av, 0)
            return 0, av, true_delta

        exp = math.floor(math.log10(abs(self.true_delta)))
        true_delta = round(self.true_delta, -exp)
        av = round(av, -exp)
        return max(0, -exp), av, true_delta
        
        
    def SEM(self, con_int = 0.95):
        from scipy.stats import t
        
        xarr = self.xarr
        N =  np.size(self.xarr)
       
        raw_delta = np.sqrt((((xarr.mean() - xarr)**2).sum())*1/(N-1))
        av_delta = raw_delta*1/(np.sqrt(N))
        
        alpha = 1 - con_int
        t_coeff = t.ppf(1 - alpha/2, df = N - 1)
        true_delta = t_coeff*av_delta
        self.raw_delta, self.av_delta, self.true_delta = raw_delta, av_delta, true_delta
        
        return {
            'raw_delta':raw_delta,
            'av_delta':av_delta,
            'true_delta':true_delta,
            't_coeff':t_coeff,
            'av': xarr.mean()
        }
        
     
        
        
              
                
            
        
        
        
            

    
        

    
    
    

          
            
            
        
            
    
        
    
    
    

