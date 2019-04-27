from scipy.interpolate import interp1d
import numpy as np 
from numpy import *
from scipy import optimize
from scipy.special import erf
import scipy.integrate as integrate

def call_boundary(r, delta, sigma, K, T):
    EC = lambda S, t, K, sigma, r, delta: 1/2*sqrt(pi)*((erf(1/4*sqrt(2)*((sigma**2 - 2*delta + 2*r)*t
    - 2*log(K/S))/(sigma*sqrt(t)))*exp(-(delta - r)*t) + exp(-(delta - r)*t))*S
    + K*erf(1/4*sqrt(2)*((sigma**2 + 2*delta - 2*r)*t
    + 2*log(K/S))/(sigma*sqrt(t))) - K)*sqrt(t)*exp(-r*t)/sqrt(pi*t)

    F = lambda epsilon, t, u, v, r, delta, sigma: erf(
        epsilon/4*sqrt(2)*sigma*sqrt(u-t)
            -1/2*sqrt(2)*delta*sqrt(u-t)/sigma
            + 1/2*sqrt(2)*r*sqrt(u-t)/sigma
        - 1/2*sqrt(2)*log(v)/(sigma*sqrt(u-t))
        )+1
    
    l1 = np.linspace(0, T * 0.9, num=10, endpoint=False)
    l2 = np.linspace(T * 0.9, T, num=11, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K * max(1, r / delta) for x in absc], kind='cubic')

    ah = lambda t,z: (exp(-delta*(z-t))*(delta/2)*F(1,t,z,f(z)/f(t),r,delta,sigma))
    bh = lambda t,z: (exp(-r*(z-t))*(r*K/2)*F(-1,t,z,f(z)/f(t),r,delta,sigma))
    for iter_count in range(3):
        loc = [K * max(1, r / delta)]
        for ttt in absc[:-1][::-1]:
            aaa = integrate.quad(lambda z: ah(ttt,z),ttt,T, epsabs=1e-02, epsrel=1e-02)[0]
            bbb = integrate.quad(lambda z: bh(ttt,z),ttt,T, epsabs=1e-02, epsrel=1e-02)[0]
            LRT = optimize.brentq(lambda x: x - K - EC(x, T - ttt, K, sigma, r, delta) - aaa * x + bbb, K, K + 100)
            loc.append(LRT)

        loc.reverse()
        f = interp1d(absc, loc, kind='cubic')
        ah = lambda t,z: (exp(-delta*(z-t))*(delta/2)*F(1,t,z,f(z)/f(t),r,delta,sigma))
        bh = lambda t,z: (exp(-r*(z-t))*(r*K/2)*F(-1,t,z,f(z)/f(t),r,delta,sigma))
        
    np.savetxt("bound.csv", np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=",")