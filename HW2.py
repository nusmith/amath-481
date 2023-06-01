import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from mpl_toolkits import mplot3d
from matplotlib import cm


# Problem 1

# Define the ODE
def rhs_bvp(x, phi, epsilon):
    f1 = phi[1]
    f2 = (x**2 - epsilon)*phi[0]
    return np.array([f1, f2])

# Constants and range of evaluation
L = 4
tol = 1e-6
x_range = [-L, L]
x_eval = np.linspace(-L, L, 20*L+1)

# Guessing phi(-L)=1
# Let A denote phi(-L)
A = 1

# epsilon is the shooting method param we will change
# We will change epsilon to see when we have a solution
epsilon_start = 0

# Vectors to contain solutions for phi and epsilon
eigenvalues = np.empty([1, 5])
eigenfunctions = np.empty([20*L+1, 5])

# Loop over epsilon values to find more eigenvalue(epsilon)/function(phi) pairs
# First 5 modes (eigenfunctions)
for modes in range(5):
    depsilon = 1
    epsilon = epsilon_start

    for j in range(1000):
        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, epsilon), t_span=x_range,  t_eval=x_eval, y0=phi0)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]

        # If the BCs are met (within tolerance), we have found a solution!
        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol:
            eigenvalues[0, modes] = epsilon
            eigenfunctions[:, modes] = phi_sol
            break
        # If phi is > 0, we need to increase epsilon
        if (-1)**(modes)*(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) > 0:
            epsilon = epsilon + depsilon
        # If phi is > 0, we need to decrease epsilon
        else:
            epsilon = epsilon - depsilon/2
            depsilon = depsilon/2

    epsilon_start = epsilon + 0.1  # increase epsilon once we find one eigenfunction to find another pair

# eigenvalues and eigenfunctions vectors will contain first five modes for phi



# Normalize eigenpairs
# Finding the sqrt integral from -L, L of phi(x)^2
normalize_denominator = lambda y, x: np.sqrt(np.abs(scipy.integrate.trapz(y=y**2, x=x)))
sol_1 = np.abs(eigenfunctions[:, 0]/normalize_denominator(eigenfunctions[:, 0], x_eval))
sol_2 = np.abs(eigenfunctions[:, 1]/normalize_denominator(eigenfunctions[:, 1], x_eval))
sol_3 = np.abs(eigenfunctions[:, 2]/normalize_denominator(eigenfunctions[:, 2], x_eval))
sol_4 = np.abs(eigenfunctions[:, 3]/normalize_denominator(eigenfunctions[:, 3], x_eval))
sol_5 = np.abs(eigenfunctions[:, 4]/normalize_denominator(eigenfunctions[:, 4], x_eval))

for modes in range(5):
    plt.plot(x_eval, eigenfunctions[:, modes]/normalize_denominator(eigenfunctions[:, modes], x_eval)
             , label='Mode: ' + str(modes + 1) + ', ε = ' + str(np.round(eigenvalues[0, modes], 3)))
plt.legend(loc='upper right', fontsize=8)
plt.title('First 5 Modes of Probability Function in a 1-D Harmonic Trapping')
plt.xlabel('Position (x)')
plt.ylabel('PDF ($\phi_n$)')
plt.show()

A1 = sol_1.reshape([81, 1])
A2 = sol_2.reshape([81, 1])
A3 = sol_3.reshape([81, 1])
A4 = sol_4.reshape([81, 1])
A5 = sol_5.reshape([81, 1])
A6 = eigenvalues.reshape([1, 5])


# Plotting for presentation skills
# First find sol to PDE
# Gaussian IC
f = np.exp(-x_eval**2)
# a will do the integral shown in video
a = np.trapz(eigenfunctions[:, 1]*f, x=x_eval)
# Time array
t = np.linspace(0, 5, 100)
# Exponential
part = np.exp(-1j*(t * eigenvalues[0, 1]))/2
part = a * part
soln = eigenfunctions[:, 1].reshape([81, 1]) @ part.reshape([1, 100])

# Plot 3D Figure of Time Evolution of Psi
fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
X, T = np.meshgrid(x_eval, t)
surf = ax2.plot_surface(X, T, soln.T.real, cmap=cm.hsv, rstride=1, cstride=1)
fig2.colorbar(surf, orientation='horizontal', fraction=0.05)
ax2.set_xlabel('Position (x)', fontsize=7)
ax2.set_ylabel('Time (t)', fontsize=7)
ax2.set_zlabel('PDF (Ψ2)', fontsize=7)
plt.suptitle('Time Evolution of Probability Density Function $Ψ_2$', fontsize='12')
plt.title('In a 1-D Harmonic Trapping Potential', fontsize='10')
plt.show()

# Contour Plot for more information about Maxima, Minima
fig3, ax3 = plt.subplots()
X, T = np.meshgrid(x_eval, t)
surf = ax3.contourf(X, T, soln.T.real)
ax2.set_xlabel('Position (x)', fontsize=7)
ax2.set_ylabel('Time (t)', fontsize=7)
fig3.colorbar(surf)
plt.suptitle('Position vs. Time Contour Map of PDF $Ψ_2$', fontsize='12')
plt.title('In a 1-D Harmonic Trapping Potential', fontsize='10')
plt.xlabel('Position (x)')
plt.ylabel('Time (t)')
plt.show()




# Problem 2

# Constants and n function
L = 4
n = lambda x: x**2
x_range = [-L, L]
x_eval = np.linspace(-L, L, 20*L+1)
tol = 1e-6
dx = 0.1

A = np.zeros([20*L-1, 20*L-1])
# Fill in the middle of A using central difference methods
for m in range(1, 20*L-2):
    A[m, m-1] = -1/(dx**2)
    A[m, m] = 2/(dx**2) + n(x_eval[m+1])
    A[m, m+1] = -1/(dx**2)


# First and last rows using boundary conditions
# phi0 = -1/3 phi2 + 4/3 phi1
# phi-1 = 4 phi-2 - 3phi-3
# d/dx^2phi1 = -phi0/t^2 + 2phi1/t^2 + - phi2/t^2
    # phi0 = 4/3 phi1 - 1/3phi2
    # A[0] = (2-4/3)phi1/t^2 + x_eval[1]^2 | (-1+1/3)phi2/t^2 + x_eval[2]^2

A[0, 0] = (2/3)/(dx**2) + x_eval[1]**2  # coeffs of phi1, with fix for bcs
A[0, 1] = (-2/3)/(dx**2)   # coeffs of phi2, with fix for bcs
A[20*L-2, 20*L-3] = (-2/3)/(dx**2) # coeffs of phi n-2, with fix for bcs
A[20*L-2, 20*L-2] = (2/3)/(dx**2) + x_eval[len(x_eval) - 2]**2  # coeffs of phi n-1, with fix for bcs

w, v = np.linalg.eig(A)
# Sort by smallest to largest eigenvalues
idx = w.argsort()[::1]
w = w[idx]
v = v[:, idx]


# Append phi0 and phiN to five smallest eigenvectors, assume dx != 0
first_5_eigvals = np.array([w[0:5]])

eigvector_1 = v[:, 0].reshape([79, 1])
eigvector_1 = np.insert(eigvector_1, 0, (4*v[0, 0] - v[1, 0]) / (3 + 2*dx*np.sqrt((-L)**2 - first_5_eigvals[0, 0])))
eigvector_1 = np.append(eigvector_1, (4*v[78, 0] - v[77, 0]) / (3 + 2*dx*np.sqrt((L**2) - first_5_eigvals[0, 0])))

eigvector_2 = v[:, 1].reshape([79, 1])
eigvector_2 = np.insert(eigvector_2, 0, (4*v[0, 1] - v[1, 1]) / (3 + 2*dx*np.sqrt((-L)**2 - first_5_eigvals[0, 1])))
eigvector_2 = np.append(eigvector_2, (4*v[78, 1] - v[77, 1]) / (3 + 2*dx*np.sqrt((L**2) - first_5_eigvals[0, 1])))


eigvector_3 = v[:, 2].reshape([79, 1])
eigvector_3 = np.insert(eigvector_3, 0, (4*v[0, 2] - v[1, 2]) / (3 + 2*dx*np.sqrt((-L)**2 - first_5_eigvals[0, 2])))
eigvector_3 = np.append(eigvector_3, (4*v[78, 2] - v[77, 2]) / (3 + 2*dx*np.sqrt((L**2) - first_5_eigvals[0, 2])))


eigvector_4 = v[:, 3].reshape([79, 1])
eigvector_4 = np.insert(eigvector_4, 0, (4*v[0, 3] - v[1, 3]) / (3 + 2*dx*np.sqrt((-L)**2 - first_5_eigvals[0, 3])))
eigvector_4 = np.append(eigvector_4, (4*v[78, 3] - v[77, 3]) / (3 + 2*dx*np.sqrt((L**2) - first_5_eigvals[0, 3])))


eigvector_5 = v[:, 4].reshape([79, 1])
eigvector_5 = np.insert(eigvector_5, 0, (4*v[0, 4] - v[1, 4]) / (3 + 2*dx*np.sqrt((-L)**2 - first_5_eigvals[0, 4])))
eigvector_5 = np.append(eigvector_5, (4*v[78, 4] - v[77, 4]) / (3 + 2*dx*np.sqrt((L**2) - first_5_eigvals[0, 4])))

plt.plot(x_eval, eigvector_1, label='0')
plt.plot(x_eval, eigvector_2, label='1')
plt.plot(x_eval, eigvector_3, label='2')
plt.plot(x_eval, eigvector_4, label='3')
plt.plot(x_eval, eigvector_5, label='4')
plt.legend()
plt.show()
# Normalize the eigenvectors, take absolute value as well
normalize_denominator = lambda y, x: np.sqrt(np.abs(scipy.integrate.trapz(y=y**2, x=x)))

sol_1 = np.abs(eigvector_1)/normalize_denominator(eigvector_1, x_eval)
sol_2 = np.abs(eigvector_2)/normalize_denominator(eigvector_2, x_eval)
sol_3 = np.abs(eigvector_3)/normalize_denominator(eigvector_3, x_eval)
sol_4 = np.abs(eigvector_4)/normalize_denominator(eigvector_4, x_eval)
sol_5 = np.abs(eigvector_5)/normalize_denominator(eigvector_5, x_eval)
print(first_5_eigvals)

A7 = sol_1.reshape([81, 1])
A8 = sol_2.reshape([81, 1])
A9 = sol_3.reshape([81, 1])
A10 = sol_4.reshape([81, 1])
A11 = sol_5.reshape([81, 1])
A12 = first_5_eigvals.reshape([1, 5])


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

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            eigenvalues_1[0, modes] = epsilon
            eigenfunctions_1[:, modes] = phi_sol
            #plt.plot(sol.t, sol.y[0, :], label="Gamma = 0.05, mode: " + str(modes))
            break
        else:
            A = A/np.sqrt(norm)

        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma1, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=phi_sol**2, x=x_eval)))

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            eigenvalues_1[0, modes] = epsilon
            eigenfunctions_1[:, modes] = phi_sol
            #plt.plot(sol.t, sol.y[0, :], label="Gamma = 0.05, mode: " + str(modes))
            break
        elif (-1)**modes*(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) > 0:  # if phi is > 0, we need to increase epsilon
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2
            depsilon = depsilon/2

    epsilon_start = epsilon + 0.1  # decrease beta once we find one eigenfunction to find another pair

# Reset epsilon_start to find solution for gamma = -0.05
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

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            eigenvalues_2[0, modes] = epsilon
            eigenfunctions_2[:, modes] = phi_sol
            #plt.plot(sol.t, sol.y[0, :], label="Gamma = -0.05, mode: " + str(modes))
            break
        else:
            A = A/np.sqrt(norm)

        phi0 = np.array([A, A*np.sqrt(L**2 - epsilon)])
        sol = scipy.integrate.solve_ivp(lambda x, phi: rhs_bvp(x, phi, gamma2, epsilon), t_span=x_range, y0=phi0, t_eval=x_eval)
        phi_sol = sol.y[0, :]
        phi_prime_sol = sol.y[1, :]
        norm = np.sqrt(np.abs(scipy.integrate.trapz(y=phi_sol**2, x=x_eval)))

        if np.abs(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) < tol and np.abs(1 - norm) < tol:
            eigenvalues_2[0, modes] = epsilon
            eigenfunctions_2[:, modes] = phi_sol
            #plt.plot(sol.t, sol.y[0, :], label="Gamma = -0.05, mode: " + str(modes))
            break
        elif (-1)**modes*(phi_prime_sol[-1] + np.sqrt(L**2 - epsilon)*phi_sol[-1]) > 0:  # if phi is > 0, we need to increase epsilon
            epsilon = epsilon + depsilon
        else:
            epsilon = epsilon - depsilon/2
            depsilon = depsilon/2

    epsilon_start = epsilon + 0.1  # decrease beta once we find one eigenfunction to find another pair



# For gamma1
A13 = np.abs(eigenfunctions_1[:, 0]).reshape([61, 1])
A14 = np.abs(eigenfunctions_1[:, 1]).reshape([61, 1])
A15 = eigenvalues_1.reshape([1, 2])

# For gamma2
A16 = np.abs(eigenfunctions_2[:, 0]).reshape([61, 1])
A17 = np.abs(eigenfunctions_2[:, 1]).reshape([61, 1])
A18 = eigenvalues_2.reshape([1, 2])

for modes in range(2):
    plt.plot(x_eval, eigenfunctions_1[:, modes]/normalize_denominator(eigenfunctions_1[:, modes], x_eval)
             , label='Mode: ' + str(modes + 1) + ', ε = ' + str(np.round(eigenvalues_1[0, modes], 3)) + ', $\gamma$ = ' + str(gamma1))
plt.legend(loc='upper right', fontsize=8)
plt.suptitle('First 5 Modes of Probability Function in a 1-D Harmonic Trapping', fontsize=12)
plt.title('$\gamma$ = ' + str(gamma1), fontsize=10)
plt.xlabel('Position (x)')
plt.ylabel('PDF ($\phi_n$)')
plt.show()
for modes in range(2):
    plt.plot(x_eval, eigenfunctions_2[:, modes]/normalize_denominator(eigenfunctions_2[:, modes], x_eval)
             , label='Mode: ' + str(modes + 1) + ', ε = ' + str(np.round(eigenvalues_2[0, modes], 3)) + ', $\gamma$ = ' + str(gamma2))
plt.legend(loc='upper right', fontsize=8)
plt.suptitle('First 5 Modes of Probability Function in a 1-D Harmonic Trapping', fontsize=12)
plt.title('$\gamma$ = ' + str(gamma2), fontsize=10)
plt.xlabel('Position (x)')
plt.ylabel('PDF ($\phi_n$)')
plt.show()

print(A1[40, 0], A13[30, 0], A16[30, 0])
