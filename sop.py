"""
Simple Options Pricing
========================

S - Stock price
K - Strike price
T - Expiration days
rf - risk-free interest rate
sigma - volatility
N - periods
PC - put or call: 'P', 'C'
EuroAmer - European or American Options: 'Euro', 'Amer'

"""

import numpy as np

def BSM(S, K, T, rf, sigma, PC):
    d1 = (np.log(S/K) + (rf+sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if PC == 'C':
        return S*Ndp6(d1) - K*np.exp(-rf*T)*Ndp6(d2)
    elif PC == 'P':
        return K*np.exp(-rf*T)*Ndp6(-d2) - S*Ndp6(-d1)

def EuroBin(S, K, T, rf, sigma, N, PC):
    """
    European Binary Tree
    """
    import numpy as np
    from scipy.special import comb
    dt = T/N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(rf*dt) - d) / (u - d)
    EuroBinPrice = 0.0
    for i in range(N):
        if PC == 'C':
            EuroBinPrice = EuroBinPrice + comb(N, i, exact=False)* (p**i) * (1-p)**(N-i) * max(S*u**i*d**(N-i)-K, 0)
        elif PC == 'P':
            EuroBinPrice = EuroBinPrice + comb(N, i, exact=False)* (p**i) * (1-p)**(N-i) * max(K-S*u**i*d**(N-i), 0)
    EuroBinPrice *= np.exp(-rf*T)
    return EuroBinPrice

def Binomial(Spot, K, T, r, sigma, N, PC, EuroAmer = "Amer", Dividends = None):
    """
    Binomial Tree Model
    """
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r*dt) - d) / (u-d)
    print(u, d, p)
    S = np.zeros((N+1, N+1), 'd')
    S[0,0] = Spot
    for i in range(N+1):
        for j in range(i, N+1):
            S[i,j] = S[0,0] * u**(j-i) * d**i
    #print(S)
    Opt = np.zeros((N+1, N+1), 'd')
    for i in range(N+1):
        if PC == 'C':
            Opt[i,N] = max(S[i,N] - K, 0)
        elif PC == 'P':
            Opt[i,N] = max(K - S[i,N], 0)
    
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            if EuroAmer == "Amer":
                if PC == 'C':
                    Opt[i,j] = max(S[i,j] - K, np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
                elif PC == 'P':
                    Opt[i,j] = max(K - S[i,j], np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
            elif EuroAmer == "Euro":
                Opt[i,j] = np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1])
    # print(Opt)
    return Opt[0,0]


def LRBinomial(Spot, K, T, r, v, N, PC, Method=2, EuroAmer = 'Amer'):
    if N % 2 == 0: N += 1
    dt = T/N
    # exp_rt = np.exp(-r*dt)
    d1 = (np.log(Spot/K) + (r + v*v/2.0)*T)/ v / np.sqrt(T)
    d2 = (np.log(Spot/K) + (r - v*v/2.0)*T)/ v / np.sqrt(T)
    #u_n = np.exp(v*np.sqrt(dt))
    #d_n = np.exp(-v*np.sqrt(dt))
    r_n = np.exp(r*dt)
    x1 = (d1/(N+1.0/3-(1-Method)*0.1/(N+1)))**2 * (N + 1.0/6.0)
    pp = 0.5+np.sign(d1) * 0.5 * np.sqrt(1-np.exp(-x1))
    x1 = (d2/(N+1.0/3-(1-Method)*0.1/(N+1)))**2 * (N + 1.0/6)
    p = 0.5+np.sign(d2) * 0.5 * np.sqrt(1-np.exp(-x1))
    u = r_n * pp / p
    d = (r_n - p*u)/(1-p)
    
    # print(u, d, p)
    S = np.zeros((N+1, N+1), 'd')
    S[0,0] = Spot
    for i in range(N+1):
        for j in range(i, N+1):
            S[i,j] = S[0,0] * u**(j-i) * d**i
    #print(S)
    Opt = np.zeros((N+1, N+1), 'd')
    for i in range(N+1):
        if PC == 'C':
            Opt[i,N] = max(S[i,N] - K, 0)
        elif PC == 'P':
            Opt[i,N] = max(K - S[i,N], 0)
    
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            if EuroAmer == "Amer":
                if PC == 'C':
                    Opt[i,j] = max(S[i,j] - K, np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
                elif PC == 'P':
                    Opt[i,j] = max(K - S[i,j], np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
            elif EuroAmer == "Euro":
                Opt[i,j] = np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1])
    # print(Opt)
    return Opt[0,0]

    
def FlexBin(Spot, K, T, r, sigma, N, PC, EuroAmer = 'Amer', L = None):
    """
    Flexible Binary Tree
    """
    dt = T/N
    u0 = np.exp(sigma * np.sqrt(dt))
    d0 = 1.0 / u0
    if L is None:
        j0 = np.round((np.log(K/Spot) - N*np.log(d0)) / np.log(u0/d0))
        L = (np.log(K/Spot) - (2*j0-N)*sigma*np.sqrt(dt)) / N / sigma**2 / dt
    u = np.exp(sigma * np.sqrt(dt) + L*sigma**2 * dt)
    d = np.exp(-sigma * np.sqrt(dt) + L*sigma**2 * dt)
    p = (np.exp(r*dt) - d) / (u - d)
    
    print(u, d, p)
    S = np.zeros((N+1, N+1), 'd')
    S[0,0] = Spot
    for i in range(N+1):
        for j in range(i, N+1):
            S[i,j] = S[0,0] * u**(j-i) * d**i
    #print(S)
    Opt = np.zeros((N+1, N+1), 'd')
    for i in range(N+1):
        if PC == 'C':
            Opt[i,N] = max(S[i,N] - K, 0)
        elif PC == 'P':
            Opt[i,N] = max(K - S[i,N], 0)
    
    for j in range(N-1, -1, -1):
        for i in range(j+1):
            if EuroAmer == "Amer":
                if PC == 'C':
                    Opt[i,j] = max(S[i,j] - K, np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
                elif PC == 'P':
                    Opt[i,j] = max(K - S[i,j], np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1]))
            elif EuroAmer == "Euro":
                Opt[i,j] = np.exp(-r*dt)*(p*Opt[i,j+1] + (1-p)*Opt[i+1,j+1])
    # print(Opt)
    return Opt[0,0]

def Trinomial(Spot, K, T, r, sigma, N, PC, EuroAmer = 'Amer'):
    dt = T/N
    u = np.exp(sigma * np.sqrt(2*dt))
    d = 1.0 / u
    pu = (np.exp(r*dt/2.0) - np.exp(-sigma*np.sqrt(dt/2.0)))**2 / (np.exp(sigma*np.sqrt(dt/2.0)) - np.exp(-sigma*np.sqrt(dt/2.0)))**2
    pd = (np.exp(sigma*np.sqrt(dt/2.0)) - np.exp(r*dt/2.0))**2 / (np.exp(sigma*np.sqrt(dt/2.0)) - np.exp(-sigma*np.sqrt(dt/2.0)))**2
    pm = 1.0 - pu - pd
    
    S = np.zeros((2*N+1, N+1), 'd') #
    S[0,0] = Spot
    for j in range(1, N+1):
        for i in range(0, 2*j+1):
            S[i,j] = S[0,0] * u**j * d**i
    
    Opt = np.zeros((2*N+1, N+1), 'd')
    for i in range(2*N+1):
        if PC == "P": 
            Opt[i,N] = max(S[i,N] - K, 0.0)
        elif PC == "C":
            Opt[i,N] = max(K - S[i,N], 0.0)
    
    for j in range(N, -1, -1):
        for i in range(2*N + 1):
            if EuroAmer == "Amer":
                if PC == "C":
                    Opt[i,j] = max(S[i,j] - K, np.exp(-r*dt) * (pu * Opt[i,j+1] + pm*Opt[i+1,j+1] + pd * Opt[i+2,j+1]))
                elif PC == "P":
                    Opt[i,j] = max(K - S[i,j], np.exp(-r*dt) * (pu * Opt[i,j+1] + pm*Opt[i+1,j+1] + pd * Opt[i+2,j+1]))
            elif EuroAmer == "Euro":
                Opt[i,j] = np.exp(-r*dt) * (pu * Opt[i,j+1] + pm*Opt[i+1,j+1] + pd * Opt[i+2,j+1])
    return Opt[0,0]

# Polynomial Formulas for the Cumulative Normal Distribution Function
# Accuracy: 6 decimal places
def Ndp6(x):
  if x < 0: return 1 - Ndp6(-x)
  p = 0.2316419
  b1 =  0.319381530
  b2 = -0.356563782
  b3 =  1.781477937
  b4 = -1.821255978
  b5 =  1.330274429
  t = 1/(1+p*x)
  z = np.exp(-x*x/2.0)/np.sqrt(2.0*np.pi)
  return 1 - z*((b1+(b2+(b3+(b4+b5*t)*t)*t)*t)*t)

# Accuracy: 4 decimal places
def Ndp4(x):
  if x < 0: return 1.0-Ndp4(-x)
  a1 =  0.4361836
  a2 = -0.1201676
  a3 =  0.9372980
  p  =  0.33267
  t = 1/(1.0 + p*x)
  z = np.exp(-x*x/2.0)/np.sqrt(2.0*np.pi)
  return 1.0 - z*((a1+(a2+a3*t)*t)*t)

