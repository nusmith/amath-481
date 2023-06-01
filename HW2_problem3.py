
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate


# Problem 3

# ODE
n = lambda x: x**2
def rhs_bvp(x, phi, gamma, epsilon):
    f1 = phi[1]
    f2 = (gamma*np.abs(phi[0])**2 + n(x) - epsilon)*phi[0]
    return np.array([f1, f2])



# Constants and aux functions
L = 3
tol = 1e-5
gamma1 = 0.05
gamma2 = -0.05
x_range = [-L, L]
x_eval = np.linspace(-L, L, 20 * L + 1)

# Initial conditions. A is the shooting method param we will change (for phi'')
A = 1e-4

# We will change epsilon to see when we have a solution
epsilon_start = 0

# To contain solutions for gamma=0.05
eigenvalues_1 = np.empty([1, 2])
eigenfunctions_1 = np.empty([20*L+1, 2])


# To contain solutions for gamma=-0.05
eigenvalues_2 = np.empty([1, 2])
eigenfunctions_2 = np.empty([20*L+1, 2])


# Loop over epsilon values to find more eigenvalue(epsilon)/function(phi) pairs
# Need to normalize the eigenvectors in the loop-- when checking for boundary conditions, also check for norm=1
# This is because this problem nonlinear so the eigenvectors don't scale
# First 2 modes (eigenfunctions) for gamma1 = 0.05
for modes in range(2):
    depsilon = 0.1
    epsilon = epsilon_start

    for j in range(1000):
        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma1, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=(phi_sol**2), x=x_eval)))
        print(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1])
        print(phi_sol.shape)

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            print("tolerance reached and eigenvector normalized!")
            print(epsilon, norm)
            eigenvalues_1[0, modes] = epsilon
            eigenfunctions_1[:, modes] = phi_sol
            plt.plot(sol.t, sol.y[0, :], label="Gamma = 0.05, mode: " + str(modes))
            break
        else:
            A = A/np.sqrt(norm)

        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma1, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=phi_sol**2, x=x_eval)))

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            print("tolerance reached and vector normalized!")
            print(epsilon, norm)
            eigenvalues_1[0, modes] = epsilon
            eigenfunctions_1[:, modes] = phi_sol
            plt.plot(sol.t, sol.y[0, :], label="Gamma = 0.05, mode: " + str(modes))
            break
        elif (-1)**modes*(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) > 0:  # if phi is > 0, we need to increase epsilon
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2
            depsilon = depsilon/2

    epsilon_start = epsilon + 0.1  # decrease beta once we find one eigenfunction to find another pair


epsilon_start = 0
for modes in range(2):
    depsilon = 0.1
    epsilon = epsilon_start

    for j in range(1000):
        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma2, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=(phi_sol**2), x=x_eval)))
        print(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1])
        print(phi_sol.shape)

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            print("tolerance reached and eigenvector normalized!")
            print(epsilon, norm)
            eigenvalues_2[0, modes] = epsilon
            eigenfunctions_2[:, modes] = phi_sol
            plt.plot(sol.t, sol.y[0, :], label="Gamma = -0.05, mode: " + str(modes))
            break
        else:
            A = A/np.sqrt(norm)

        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma2, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=phi_sol**2, x=x_eval)))

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            print("tolerance reached and vector normalized!")
            print(epsilon, norm)
            eigenvalues_2[0, modes] = epsilon
            eigenfunctions_2[:, modes] = phi_sol
            plt.plot(sol.t, sol.y[0, :], label="Gamma = -0.05, mode: " + str(modes))
            break
        elif (-1)**modes*(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) > 0:  # if phi is > 0, we need to increase epsilon
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2
            depsilon = depsilon/2

    epsilon_start = epsilon + 0.1  # decrease beta once we find one eigenfunction to find another pair

plt.legend()
plt.show()


# For gamma1
A13 = eigenfunctions_1[:, 0]
A14 = eigenfunctions_1[:, 1]
A15 = eigenvalues_1

# For gamma2
A16 = eigenfunctions_2[:, 0]
A17 = eigenfunctions_2[:, 1]
A18 = eigenvalues_2

plt.plot(x_eval, A13, label="Gamma = 0.05, mode: 0")

plt.plot(x_eval, A14, label="Gamma = 0.05, mode: 1")

plt.plot(x_eval, A16, label="Gamma = -0.05, mode: 0")

plt.plot(x_eval, A17, label="Gamma = -0.05, mode: 1")
