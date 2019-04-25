# Numerical Program for American-Style Call Option
from scipy.interpolate import interp1d
import numpy as np 
import pandas as pd
from numpy import *
from scipy import special, optimize
from scipy.special import *
import scipy.integrate as integrate
import scipy.special as special
import matplotlib
matplotlib.use("TkAgg")
matplotlib.interactive(True)
import matplotlib.pyplot as plt

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

K = 40
sigma = 3/10
delta = 7/100
r = 2/100
T = 1/2

l1 = np.linspace(0, 47/100, num=47, endpoint=False)
l2_0 = np.linspace(47/100, 1/2, num=30, endpoint=False)
l2 = np.linspace(47/100, 1/2, num=31, endpoint=True)
absc0 = np.concatenate((l1, l2_0), axis=0)
absc = np.concatenate((l1, l2), axis=0)
val = [max([K, K*(r/delta)]) for x in absc]

fig = plt.figure()
plt.xlabel('time')
plt.ylabel('spot price')

f = interp1d(absc, val, kind='cubic')
plt.plot(absc, f(absc), 'k')
plt.show()
plt.pause(0.001)
# np.savetxt("bound.csv", np.transpose(np.vstack((absc, f(absc)))), delimiter=",")
ah = lambda t,z: (exp(-delta*(z-t))*(delta/2)*F(1,t,z,f(z)/f(t),r,delta,sigma))
bh = lambda t,z: (exp(-r*(z-t))*(r*K/2)*F(-1,t,z,f(z)/f(t),r,delta,sigma))
for iter in range(10):
    print(iter) # 
    loc = [max([K,K*(r/delta)])]
    for ttt in absc0[::-1]:
        aaa = integrate.quad(lambda z: ah(ttt,z),ttt,T)[0]
        bbb = integrate.quad(lambda z: bh(ttt,z),ttt,T)[0]
        LRT = optimize.brentq(lambda x:
            x-K-EC(x,T-ttt,K,sigma,r,delta)-aaa*x+bbb,K-10,K+20)
        loc = [LRT]+loc
    
    val = loc
    ff = f
    f = interp1d(absc, val, kind='cubic')
    plt.plot(absc, f(absc), 'k')
    plt.show()
    plt.pause(0.001)
    # np.savetxt("bound.csv", np.transpose(np.vstack((absc, f(absc)))), delimiter=",")
    ah = lambda t,z: (exp(-delta*(z-t))*(delta/2)*F(1,t,z,f(z)/f(t),r,delta,sigma))
    bh = lambda t,z: (exp(-r*(z-t))*(r*K/2)*F(-1,t,z,f(z)/f(t),r,delta,sigma)) 

