from scipy import optimize
from scipy.interpolate import interp1d
import scipy.integrate as integrate

from scipy.special import erf
from numpy import log, exp, sqrt
from math import factorial

import numpy as np

def call_boundary_jump(r, delta, sigma, c, gamma, K, T, iteration=3, term=20, list_out=False):
    def LB(time, t0, ratio, i_jump):
        return (log(ratio) - i_jump * log(1 + gamma) - (r - delta - c * gamma - 0.5 * sigma ** 2) * (time - t0)) / sigma

    def Poisson(t_diff):
        return (c * t_diff) ** jump_list / fact * exp(-c * t_diff)

    def Aug_Poisson(t_diff):
        return (c * t_diff) ** jump_list / fact * exp(-c * t_diff) * (1 + gamma) ** jump_list

    def Fc1_list(time, t0):
        return 1 + erf((sigma * (time - t0) - LB(time, t0, f(time) / f(t0), jump_list)) / sqrt(2 * (time - t0)))   

    def Fc1(time, t0):
        return 0.5 * np.array([Aug_Poisson(iter_time - t0).dot(Fc1_list(iter_time, t0)) for iter_time in time])

    def Fc1_to_int(time, t0):
        return Fc1(time, t0) * delta * exp((-delta - c * gamma) * (time - t0))

    def Fc2_list(time, t0):
        return 1 + erf(- LB(time, t0, f(time) / f(t0), jump_list) / sqrt(2 * (time - t0)))

    def Fc2(time, t0):
        return 0.5 * np.array([Poisson(iter_time - t0).dot(Fc2_list(iter_time, t0)) for iter_time in time])

    def Fc2_to_int(time, t0):
        return Fc2(time, t0) * r * K * exp(-r * (time - t0))

    def ECJ_Fc1_list(s, t0):
        return 1 + erf((sigma * (T - t0) - LB(T, t0, K / s, jump_list)) / sqrt(2 * (T - t0)))

    def ECJ_Fc1(s, t0):
        return 0.5 * Aug_Poisson(T - t0).dot(ECJ_Fc1_list(s, t0))

    def ECJ_Fc2_list(s, t0):
        return 1 + erf(- LB(T, t0, K / s, jump_list) / sqrt(2 * (T - t0)))

    def ECJ_Fc2(s, t0):
        return 0.5 * Poisson(T - t0).dot(ECJ_Fc2_list(s, t0))

    def ECJ(s, t):
        return s * exp((-delta - c * gamma) * (T - t)) * ECJ_Fc1(s, t) - K * exp(-r * (T - t)) * ECJ_Fc2(s, t)
    
    jump_list = np.linspace(0, term, term + 1)
    fact = np.linspace(0, term, term + 1)
    fact[0] = 1
    fact = np.cumprod(fact)
    
    l1 = np.linspace(0, T * 0.9, num=10, endpoint=False)
    l2 = np.linspace(T * 0.9, T, num=15, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K * max(1, r / delta)] * len(absc), kind='cubic')

    for iter_count in range(iteration):

        # Use Gaussian Quadrature
        F1_int = [integrate.fixed_quad(Fc1_to_int, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        F2_int = [integrate.fixed_quad(Fc2_to_int, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        loc = [optimize.brentq(lambda x: x - K - ECJ(x, absc[time_idx]) - F1_int[time_idx] * x + F2_int[time_idx], K, K + 100) for time_idx in range(len(absc) - 1)]
        loc.append(K * max(1, r / delta))
        f = interp1d(absc, loc, kind='cubic')
    
    if list_out:
        return  f(np.linspace(0, T, 200))
    else:
        np.savetxt("bound.csv", np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=",")

def put_boundary_jump(r, delta, sigma, c, gamma, K, T, iteration=3, term=20, list_out=False):
    def LB(time, t0, ratio, i_jump):
        return (log(ratio) - i_jump * log(1 + gamma) - (r - delta - c * gamma - 0.5 * sigma ** 2) * (time - t0)) / sigma

    def Poisson(t_diff):
        return (c * t_diff) ** jump_list / fact * exp(-c * t_diff)

    def Aug_Poisson(t_diff):
        return (c * t_diff) ** jump_list / fact * exp(-c * t_diff) * (1 + gamma) ** jump_list

    def Fp1_list(time, t0):
        return 1 + erf((sigma * (time - t0) - LB(time, t0, f(time) / f(t0), jump_list)) / sqrt(2 * (time - t0)))   

    def Fp1(time, t0):
        return 0.5 * np.array([Aug_Poisson(iter_time - t0).dot(Fp1_list(iter_time, t0)) for iter_time in time])

    def Fp1_to_int(time, t0):
        return Fp1(time, t0) * delta * exp((-delta - c * gamma) * (time - t0))

    def Fp2_list(time, t0):
        return 1 + erf(- LB(time, t0, f(time) / f(t0), jump_list) / sqrt(2 * (time - t0)))

    def Fp2(time, t0):
        return 0.5 * np.array([Poisson(iter_time - t0).dot(Fp2_list(iter_time, t0)) for iter_time in time])

    def Fp2_to_int(time, t0):
        return Fp2(time, t0) * r * K * exp(-r * (time - t0))

    def EPJ_Fp1_list(s, t0):
        return 1 - erf((sigma * (T - t0) - LB(T, t0, K / s, jump_list)) / sqrt(2 * (T - t0)))

    def EPJ_Fp1(s, t0):
        return 0.5 * Aug_Poisson(T - t0).dot(EPJ_Fp1_list(s, t0))

    def EPJ_Fp2_list(s, t0):
        return 1 - erf(- LB(T, t0, K / s, jump_list) / sqrt(2 * (T - t0)))

    def EPJ_Fp2(s, t0):
        return 0.5 * Poisson(T - t0).dot(EPJ_Fp2_list(s, t0))

    def EPJ(s, t):
        return K * exp(-r * (T - t)) * EPJ_Fp2(s, t) - s * exp((-delta - c * gamma) * (T - t)) * EPJ_Fp1(s, t)
    
    jump_list = np.linspace(0, term, term + 1)
    fact = np.linspace(0, term, term + 1)
    fact[0] = 1
    fact = np.cumprod(fact)
    
    l1 = np.linspace(0, T * 0.9, num=10, endpoint=False)
    l2 = np.linspace(T * 0.9, T, num=15, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K] * len(absc), kind='cubic')

    for iter_count in range(3):
        F1_int = [integrate.fixed_quad(Fp1_to_int, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        F2_int = [integrate.fixed_quad(Fp2_to_int, absc[time_idx], T, args=(absc[time_idx], ), n=20)[0] for time_idx in range(len(absc) - 1)]
        loc = [optimize.brentq(lambda x: K - x - ECJ(x, absc[time_idx]) + F1_int[time_idx] * x - F2_int[time_idx], 1, K) for time_idx in range(len(absc) - 1)]
        loc.append(K)
        f = interp1d(absc, loc, kind='cubic')

    if list_out:
        return  f(np.linspace(0, T, 200))
    else:
        np.savetxt("bound.csv", np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=",")
    
def call_boundary_jump_list(r, delta, sigma, c, gamma, K, T):
    if isinstance(r, np.ndarray):
        result = [call_boundary_jump(iter_r, delta, c, gamma, sigma, K, T, list_out=True) for iter_r in r]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(delta, np.ndarray):
        result = [call_boundary_jump(r, iter_delta, c, gamma, sigma, K, T, list_out=True) for iter_delta in delta]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(c, np.ndarray):
        result = [call_boundary_jump(r, delta, iter_c, gamma, sigma, iter_K, T, list_out=True) for iter_c in c]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
    if isinstance(gamma, np.ndarray):
        result = [call_boundary_jump(r, delta, c, iter_gamma, sigma, iter_K, T, list_out=True) for iter_gamma in gamma]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
    if isinstance(sigma, np.ndarray):
        result = [call_boundary_jump(r, delta, c, gamma, iter_sigma, K, T, list_out=True) for iter_sigma in sigma]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(K, np.ndarray):
        result = [call_boundary_jump(r, delta, sigma, iter_K, T, list_out=True) for iter_K in K]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
def put_boundary_jump_list(r, delta, sigma, c, gamma, K, T):
    if isinstance(r, np.ndarray):
        result = [put_boundary_jump(iter_r, delta, c, gamma, sigma, K, T, list_out=True) for iter_r in r]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(delta, np.ndarray):
        result = [put_boundary_jump(r, iter_delta, c, gamma, sigma, K, T, list_out=True) for iter_delta in delta]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(c, np.ndarray):
        result = [put_boundary_jump(r, delta, iter_c, gamma, sigma, iter_K, T, list_out=True) for iter_c in c]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
    if isinstance(gamma, np.ndarray):
        result = [put_boundary_jump(r, delta, c, iter_gamma, sigma, iter_K, T, list_out=True) for iter_gamma in gamma]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return
    
    if isinstance(sigma, np.ndarray):
        result = [put_boundary_jump(r, delta, c, gamma, iter_sigma, K, T, list_out=True) for iter_sigma in sigma]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return

    if isinstance(K, np.ndarray):
        result = [put_boundary_jump(r, delta, sigma, iter_K, T, list_out=True) for iter_K in K]
        np.savetxt("bound_list.csv", np.vstack([np.linspace(0, T, 200), np.array(result)]).T, delimiter=",")
        return