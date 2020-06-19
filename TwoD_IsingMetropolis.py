"""Code applies Monte Carlo Metropolis Algorithm in 2D Ising Model"""

### Sahil Islam ###
### 0.2/04/2020 ###

import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.random import rand

# Constants
J = 1
mu = 0.33
k = 1
B = 0.


def initial_state(N):
    initial = 2 * np.random.randint(2, size=(N, N)) - 1
    return initial


def Metropolis_loop(lattice_configuration, T):
    for i in range(N):
        for j in range(N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            lattice_point = lattice_configuration[a, b]
            nearest_neighbour = lattice_configuration[(a + 1) % N, b] + lattice_configuration[(a - 1) % N, b] + \
                                lattice_configuration[a, (b + 1) % N] + lattice_configuration[a, (b - 1) % N]
            prod = 2 * lattice_point * nearest_neighbour + mu * B * lattice_point
            if prod < 0:
                lattice_point *= -1
            elif rand() < math.exp(-prod / (k * T)):
                lattice_point *= -1

            lattice_configuration[a, b] = lattice_point

    return lattice_configuration


def Energy(lattice_configuration):
    energy1 = 0
    for i in range(len(lattice_configuration)):
        for j in range(len(lattice_configuration)):
            lattice_point = lattice_configuration[i, j]
            nearest_neighbour = lattice_configuration[(i + 1) % N, j] + lattice_configuration[(i - 1) % N, j] + \
                                lattice_configuration[i, (j + 1) % N] + lattice_configuration[
                                    i, (j - 1) % N]
            energy1 += - J * lattice_point * nearest_neighbour - mu * B * lattice_point
    return energy1 / 4.


def Magnetization(lattice_configuration):
    magnetization1 = np.sum(lattice_configuration)
    return magnetization1


def Analytic_magnetization(T):
    a = np.sinh(2 * J / (k * T))
    b = a ** (-0.25)
    c = 1 - b
    d = (c ** 0.125)
    return d


N = 15
T_min = 0.2
T_max = 4.0
step = 50
interval = (T_max - T_min) / float(step)
metropolis_step = 3000
calculation_step = 1000
n1 = 1 / (calculation_step * N * N)
n2 = 1 / (calculation_step * calculation_step * N * N)
T = np.linspace(T_min, T_max, step)
E = np.zeros(step)
M = np.zeros(step)
C = np.zeros(step)
X = np.zeros(step)
for tem in range(step):
    lattice_configuration = initial_state(N)

    En = 0
    Mag = 0
    En2 = 0
    Mag2 = 0

    for i in range(metropolis_step):
        Metropolis_loop(lattice_configuration, T[tem])

    for i in range(calculation_step):
        Metropolis_loop(lattice_configuration, T[tem])
        # Calculating from the loops
        energy = Energy(lattice_configuration)
        magnetization = Magnetization(lattice_configuration)

        # Summing
        En = En + energy
        Mag = Mag + magnetization
        En2 = En2 + (energy * energy)
        Mag2 = Mag2 + (magnetization * magnetization)

        E[tem] = n1 * En
        M[tem] = n1 * Mag
        C[tem] = (n1 * En2 - n2 * En * En) / (k * T[tem] * T[tem])
        X[tem] = (n1 * Mag2 - n2 * Mag * Mag) / (k * T[tem])

# Analytic Magnetization:


'''Ma = np.zeros(step)
for tem in range(step):
    Ma[tem] = Analytic_magnetization(T[tem])

plt.scatter(T, abs(M), s=50, marker='.', label='Computational')
plt.plot(T, Ma, label='Analytical')
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Simulation of 2D Ising Model by Metropolis Algorithm\n" + "Lattice Dimension:" + str(N) + "X" + str(
    N) + "\n" + "External Magnetic Field(B)=" + str(B) + "\n" + "Metropolis Step=" + str(metropolis_step))
plt.legend()

plt.show()'''

plt.figure(figsize=(14.17, 7))
plt.subplot(2, 2, 1)
plt.scatter(T, E, s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Energy")
plt.title('Energy Vs. Temperature')

plt.subplot(2, 2, 2)
plt.scatter(T, abs(M), s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Magnetization")
plt.title("Magnetization Vs Temperature")

plt.subplot(2, 2, 3)
plt.scatter(T, C, s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Specific Heat")
plt.title("Specific Heat vs Temperature")

plt.subplot(2, 2, 4)
plt.scatter(T, X, s=50, marker=".")
plt.xlabel("Temperature")
plt.ylabel("Susceptibility")
plt.title("Susceptibility vs Temperature")

plt.subplots_adjust(0.12, 0.11, 0.90, 0.81, 0.26, 0.56)
plt.suptitle("Simulation of 2D Ising Model by Metropolis Algorithm\n" + "Lattice Dimension:" + str(N) + "X" + str(
    N) + "\n" + "External Magnetic Field(B)=" + str(B) + "\n" + "Metropolis Step=" + str(metropolis_step))
