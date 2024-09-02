'''
Vectorized functions for Duopoly and ToyDuopoly classes
'''
import numpy as np
from numba import vectorize, float64, int64

@vectorize([float64(int64, *([float64]*4))], nopython=True)
def C(i, x, μ, γ1, γ2):
    '''Cost function of firm i'''
    if i == 1:
        return (1/γ1)*μ*x**2
    else:
        return (1/γ2)*μ*x**2
    
@vectorize([float64(int64, *([float64]*4))], nopython=True)
def MC(i, x, μ, γ1, γ2):
    '''Marginal cost of firm i'''
    if i == 1:
        return 2*μ*x/γ1
    else:
        return 2*μ*x/γ2

@vectorize([float64(*([float64]*4))], nopython=True)
def b_m(x1, x2, t, μ):        
        '''Marginal b for toy'''
        b = (1/(2*t))*(μ*(x1-x2)+t)
        b = np.maximum(0, b)
        b = np.minimum(1, b)
        return b

@vectorize([float64(int64, *([float64]*4))], nopython=True)    
def Q(i, x1, x2, t, μ):
    '''Return the quantity demanded from firm i'''
    if i == 1:
        return b_m(x1, x2, t, μ)
    else:
        return 1-b_m(x1, x2, t, μ)

@vectorize([float64(float64, float64)], nopython=True)  
def U(x, μ):
    '''Utility of x for type μ'''
    ν = 0
    return μ*x + ν*(1-(1-x)**2)

@vectorize([float64(int64, float64, float64)], nopython=True)
def tc(i, b, t):
    '''Transport cost'''
    if i == 1:
        return t*b
    else:
        return t*(1-b)

@vectorize([int64(int64, *([float64]*5))], nopython=True)    
def D(i, x1, x2, b, μ, t):
    '''Demand indicator of firm i'''
    v1 = U(x1, μ) - tc(1, b, t)
    v2 = U(x2, μ) - tc(2, b, t)
    if i == 1:
        return v1 >= v2
    else:
        return v2 > v1

@vectorize([float64(int64, *([float64]*7))], nopython=True)
def Π(i, x1, x2, t, μ, p, γ1, γ2):
    '''Return the profit of firm i'''
    x = x1 if i == 1 else x2
    return Q(i, x1, x2, t, μ)*(p-C(i, x, μ, γ1, γ2))

@vectorize([float64(int64, *([float64]*7))], nopython=True)
def FOC(i, x1, x2, t, μ, p, γ1, γ2):
    '''Return the first order condition of firm i'''
    dQ = (1/(2*t))*μ # Same for both firms
    x = x1 if i == 1 else x2
    A1 = dQ*(p-C(i, x, μ, γ1, γ2))
    A2 = Q(i, x1, x2, t, μ)*MC(i, x, μ, γ1, γ2)
    return  A1 - A2

@vectorize([float64(int64, *([float64]*4))], nopython=True)
def x_max(i, γ1, γ2, μ, p):
    '''Upper bound to quality of firm i'''
    if i==1:
        return np.sqrt((γ1*p)/μ)
    else:
        return np.sqrt((γ2*p)/μ)

@vectorize([float64(*([float64]*4))], nopython=True)
def λ_tilde(x1, x2, b, t):
    '''Switching margin λ_tilde'''
    return (2*t*b-t)/(x1-x2)