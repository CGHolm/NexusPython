import numpy as np
from ..main import dataclass

# Calculate normalization constant
# Polytropic exponents
g1 = 1.0
g2 = 1.1
g3 = 1.4
g4 = 1.1
g5 = 5.0 / 3.0
# upper density limits in numerical units
r1 = 2.5e-16 / dataclass().d_cgs  # 7.88e4
r2 = 3.84e-13 / dataclass().d_cgs # 1.21e8
r3 = 3.84e-8 / dataclass().d_cgs  # 1.21e13
r4 = 3.84e-3 / dataclass().d_cgs  # 1.21e18
# normalization constants for each segment using P = k rho^gamma
# needed to make the pressure continous
sound_speed = 1
k1 = sound_speed**2
k2 = k1 * r1**(g1 - g2)
k3 = k2 * r2**(g2 - g3)
k4 = k3 * r3**(g3 - g4)
k5 = k4 * r4**(g4 - g5)

    # The function below is from: utilities/python/dispatch/EOS/polytrope.py
# 2. Equation in https://arxiv.org/pdf/2004.07523.pdf
def calc_gamma(rho):
    result = np.empty_like(rho)
    w1 = rho < r1
    w2 = np.logical_and(rho >= r1, rho < r2)
    w3 = np.logical_and(rho >= r2, rho < r3)
    w4 = np.logical_and(rho >= r3, rho < r4)
    w5 = rho >= r4
    result[w1] = g1
    result[w2] = g2
    result[w3] = g3
    result[w4] = g4
    result[w5] = g5
    return result

def calc_pressure(rho):
    P = np.empty_like(rho)
    w1 = rho < r1
    w2 = np.logical_and(rho >= r1, rho < r2)
    w3 = np.logical_and(rho >= r2, rho < r3)
    w4 = np.logical_and(rho >= r3, rho < r4)
    w5 = rho >= r4
    P[w1] = k1 * rho[w1]**g1
    P[w2] = k2 * rho[w2]**g2
    P[w3] = k3 * rho[w3]**g3
    P[w4] = k4 * rho[w4]**g4
    P[w5] = k5 * rho[w5]**g5
    return P