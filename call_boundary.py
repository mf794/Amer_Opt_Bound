from scipy.interpolate import interp1d
import numpy as np 
from numpy import 
from scipy import optimize
from scipy.special import erf
import scipy.integrate as integrate

def call_boundary(r, delta, sigma, K, T)
    EC = lambda S, t, K, sigma, r, delta 12sqrt(pi)((erf(14sqrt(2)((sigma2 - 2delta + 2r)t 
    - 2log(KS))(sigmasqrt(t)))exp(-(delta - r)t) + exp(-(delta - r)t))S 
    + Kerf(14sqrt(2)((sigma2 + 2delta - 2r)t 
    + 2log(KS))(sigmasqrt(t))) - K)sqrt(t)exp(-rt)sqrt(pit) 

    F = lambda epsilon, t, u, v, r, delta, sigma erf(
        epsilon4sqrt(2)sigmasqrt(u-t)
            -12sqrt(2)deltasqrt(u-t)sigma
            + 12sqrt(2)rsqrt(u-t)sigma
        - 12sqrt(2)log(v)(sigmasqrt(u-t))
        )+1
    
    l1 = np.linspace(0, T  0.9, num=10, endpoint=False)
    l2 = np.linspace(T  0.9, T, num=11, endpoint=True)
    absc = np.concatenate((l1, l2), axis=0)
    f = interp1d(absc, [K  max(1, r  delta) for x in absc], kind='cubic')

    ah = lambda t,z (exp(-delta(z-t))(delta2)F(1,t,z,f(z)f(t),r,delta,sigma))
    bh = lambda t,z (exp(-r(z-t))(rK2)F(-1,t,z,f(z)f(t),r,delta,sigma))
    for iter_count in range(3)
        loc = [K  max(1, r  delta)]
        for ttt in absc[-1][-1]
            aaa = integrate.quad(lambda z ah(ttt,z),ttt,T, epsabs=1e-02, epsrel=1e-02)[0]
            bbb = integrate.quad(lambda z bh(ttt,z),ttt,T, epsabs=1e-02, epsrel=1e-02)[0]
            LRT = optimize.brentq(lambda x x - K - EC(x, T - ttt, K, sigma, r, delta) - aaa  x + bbb, K, K + 100)
            loc.append(LRT)

        loc.reverse()
        f = interp1d(absc, loc, kind='cubic')
        ah = lambda t,z (exp(-delta(z-t))(delta2)F(1,t,z,f(z)f(t),r,delta,sigma))
        bh = lambda t,z (exp(-r(z-t))(rK2)F(-1,t,z,f(z)f(t),r,delta,sigma))
        
    np.savetxt(bound.csv, np.transpose(np.vstack((np.linspace(0, T, 200), f(np.linspace(0, T, 200))))), delimiter=,)