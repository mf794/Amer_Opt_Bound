from scipy import optimize
from scipy.interpolate import interp1d
import scipy.integrate as integrate

from scipy.special import erf
from numpy import log, exp, sqrt

import numpy as np

def call_boundary(r, delta, sigma, K, T, iteration=3, list_out=False):
    def EC(S, t, K, sigma, r, delta):
        d1 = (log(S / K) + (r - delta) * t) / sigma / sqrt(t) + 0.5 * sigma * sqrt(t)
        d2 = (log(S / K) + (r - delta) * t) / sigma / sqrt(t) - 0.5 * sigma * sqrt(t)
        return 0.5 * (S * exp(-delta * t) * (1 + erf(d1 / sqrt(2))) - K * exp(-r * t) * (1 + erf(d2 / sqrt(2))))
    
    def F(epsilon, u, v, sigma, r, delta, t):
        return 1 + erf(((r - delta + epsilon * 0.5 * sigma ** 2) * (u - t) - log(v)) / (sigma * sqrt(2 * (u - t))))
    
    def F1C(z, t):
        return exp(-delta * (z - t)) * (delta / 2) * F(1, z, f(z) / f(t), sigma, r, delta, t)

    def F2C(z, t):
        return exp(-r * (z - t)) * (r * K / 2) * F(-1, z, f(z) / f(t), sigma, r, delta, t)    
    
    l1 = np.linspace(0, T * 0.9, num=10, endpoint=False)
    l2 = np.linspace(T * 0.9, T, num=15, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K * max(1, r / delta)] * len(absc), kind='cubic')
    
    for iter_count in range(iteration):
        F1_int = [integrate.fixed_quad(F1C, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        F2_int = [integrate.fixed_quad(F2C, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        loc = [optimize.brentq(lambda x: x - K - EC(x, T - absc[time_idx], K, sigma, r, delta) - F1_int[time_idx] * x + F2_int[time_idx], K, K + 100) for time_idx in range(len(absc) - 1)]
        loc.append(K * max(1, r / delta))
        f = interp1d(absc, loc, kind='cubic')
    
    if list_out:
        return  f(np.linspace(0, T, 200))
    else:
        np.savetxt("bound.csv", np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=",")
    
def put_boundary(r, delta, sigma, K, T, iteration=3, list_out=False):
    def EP(S, t, K, sigma, r, delta):
        d1 = (log(S / K) + (r - delta) * t) / sigma / sqrt(t) + 0.5 * sigma * sqrt(t)
        d2 = (log(S / K) + (r - delta) * t) / sigma / sqrt(t) - 0.5 * sigma * sqrt(t)
        return 0.5 * (K * exp(-r * t) * (1 - erf(d2 / sqrt(2))) - S * exp(-delta * t) * (1 - erf(d1 / sqrt(2))))
    
    def F(epsilon, u, v, sigma, r, delta, t):
        return 1 + erf(((r - delta + epsilon * 0.5 * sigma ** 2) * (u - t) - log(v)) / (sigma * sqrt(2 * (u - t))))
    
    def F1P(z, t):
        return exp(-delta * (z - t)) * (delta / 2) * (2 -  F(1, z, f(z) / f(t), sigma, r, delta, t))

    def F2P(z, t):
        return exp(-r * (z - t)) * (r * K / 2) * (2 - F(-1, z, f(z) / f(t), sigma, r, delta, t))
    
    l1 = np.linspace(0, T * 0.9, num=10, endpoint=False)
    l2 = np.linspace(T * 0.9, T, num=15, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K] * len(absc), kind='cubic')
    
    for iter_count in range(iteration):
        F1_int = [integrate.fixed_quad(F1P, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        F2_int = [integrate.fixed_quad(F2P, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        loc = [optimize.brentq(lambda x: K - x - EP(x, T - absc[time_idx], K, sigma, r, delta) + F1_int[time_idx] * x - F2_int[time_idx], 1, K) for time_idx in range(len(absc) - 1)]
        loc.append(K)
        f = interp1d(absc, loc, kind='cubic')
        
    if list_out:
        return  f(np.linspace(0, T, 200))
    else:
        np.savetxt("bound.csv", np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=",")

def call_boundary_list(r, delta, sigma, K, T):
    if isinstance(r, np.ndarray):
        result = [call_boundary(iter_r, delta, sigma, K, T, list_out=True) for iter_r in r]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(delta, np.ndarray):
        result = [call_boundary(r, iter_delta, sigma, K, T, list_out=True) for iter_delta in delta]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(sigma, np.ndarray):
        result = [call_boundary(r, delta, iter_sigma, K, T, list_out=True) for iter_sigma in sigma]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(K, np.ndarray):
        result = [call_boundary(r, delta, sigma, iter_K, T, list_out=True) for iter_K in K]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
def put_boundary_list(r, delta, sigma, K, T):
    if isinstance(r, np.ndarray):
        result = [put_boundary(iter_r, delta, sigma, K, T, list_out=True) for iter_r in r]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(delta, np.ndarray):
        result = [put_boundary(r, iter_delta, sigma, K, T, list_out=True) for iter_delta in delta]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(sigma, np.ndarray):
        result = [put_boundary(r, delta, iter_sigma, K, T, list_out=True) for iter_sigma in sigma]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(K, np.ndarray):
        result = [put_boundary(r, delta, sigma, iter_K, T, list_out=True) for iter_K in K]
        np.savetxt("bound.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return