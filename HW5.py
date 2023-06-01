import numpy as np
from numpy.fft import ifft2, fft2
import scipy
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm

# Constants for both problems
dt = 0.5
t_range = [0, 25]
t = np.arange(0, 25+dt, dt)
Beta = 1 # Beta
D_1 = 0.1
D_2 = 0.1


def cheb(N):
    # N is the number of points in the interior.
    if N==0:
        D = 0
        x = 1
        return D, x
    vals = np.linspace(0, N, N+1)
    x = np.cos(np.pi*vals/N)
    x = x.reshape(-1, 1)
    c = np.ones(N-1)
    c = np.pad(c, (1,), constant_values = 2)
    c *= (-1)**vals
    c = c.reshape(-1, 1)
    X = np.tile(x, (1, N+1))
    dX = X-X.T
    D = (c*(1/c.T))/(dX+(np.eye(N+1)))       #off-diagonal entries
    D = D - np.diag(sum(D.T))                #diagonal entries
    return D, x

# Problem 1
n = 64
L = 20
x1_total = np.linspace(-L/2, L/2, n+1)
x1 = x1_total[0:-1]
y1_total = np.linspace(-L/2, L/2, n+1)
y1 = y1_total[0:-1]
X1, Y1 = np.meshgrid(x1, y1)
m1 = 3
alpha1 = 0
# k values for FFT
r1 = np.arange(0, n/2, 1)
r2 = np.arange(-n/2, 0, 1)
kx = (2*np.pi/L) * np.concatenate((r1, r2))
ky = (2*np.pi/L) * np.concatenate((r1, r2))
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2  # Laplacian
K_vector = K.T.reshape(n**2)

# Initial conditions of Reaction Diffusion System
u1 = (np.tanh(np.sqrt(X1**2+Y1**2))-alpha1)*np.cos(m1*np.angle(X1+1j*Y1) - np.sqrt(X1**2+Y1**2))
v1 = (np.tanh(np.sqrt(X1**2+Y1**2))-alpha1)*np.sin(m1*np.angle(X1+1j*Y1) - np.sqrt(X1**2+Y1**2))
# Fourier transform of ICs
FU0 = fft2(u1)
FU0_vec = FU0.T.reshape(n**2)
FV0 = fft2(v1)
FV0_vec = FV0.T.reshape(n**2)
# Reshape initial conditions into an array (shape nxnx2)
FUV_initial = np.concatenate((FU0_vec, FV0_vec))
# Solve using RK45
# Solve the PDE
def fft_RHS (t, Fuv_arr):
    # Convert into matrix shape and back to space (inverse FFT)
    uhat_vec = Fuv_arr[0:n**2]
    vhat_vec = Fuv_arr[n**2:]
    uhat = Fuv_arr[0:n**2].reshape(n, n).T
    vhat = Fuv_arr[n**2:].reshape(n, n).T
    u = np.real(ifft2(uhat))
    v = np.real(ifft2(vhat))
    # Nonlinear terms in physical space
    A = u**2 + v**2
    lam = 1 - A
    omega = -Beta * A
    # RHS for U and V in Fourier domain
    ut = fft2(lam * u - omega * v).T.reshape(n**2) - D_1*K_vector*uhat_vec
    vt = fft2(omega * u - lam * v).T.reshape(n**2) - D_2*K_vector*vhat_vec
    return np.concatenate((ut, vt))

uv_vector_sol = scipy.integrate.solve_ivp(lambda t, Fuv: fft_RHS(t, Fuv), t_range, t_eval=t, y0=FUV_initial, method="RK45")
U_sol = np.real(ifft2(uv_vector_sol.y[0:n**2, 4].reshape(n, n).T))
V_sol = np.real(ifft2(uv_vector_sol.y[n**2:, 4].reshape(n, n).T))


A1 = X1.copy()
A2 = u1.copy()
A3 = np.real(FU0)
A4 = np.imag(FUV_initial.reshape(n**2*2, 1))
A5 = np.real(uv_vector_sol.y)
A6 = np.imag(uv_vector_sol.y)
A7 = np.real(uv_vector_sol.y[0:n**2, 4]).reshape(n**2, 1) # index 4 is where t=2
A8 = np.real(uv_vector_sol.y[0:n**2, 4]).reshape(n, n).T
A9 = np.real(ifft2(uv_vector_sol.y[0:n**2, 4].reshape(n, n).T))
# Plot Results
fig, ax = plt.subplots()
surf = ax.contourf(X1, Y1, A9, cmap=cm.plasma)
fig.colorbar(surf, orientation='horizontal')
ax.set_xlabel('Position (x)', fontsize=7)
ax.set_ylabel('Position(y)', fontsize=7)
ax.set_title("U(x,y) at time t = 2 using FFT")
plt.show()
fig, ax = plt.subplots()

V = np.real(ifft2(uv_vector_sol.y[n**2:, -1].reshape(n, n).T))
surf = ax.contourf(X1, Y1, V, cmap=cm.plasma)
fig.colorbar(surf, orientation='horizontal')
ax.set_xlabel('Position (x)', fontsize=7)
ax.set_ylabel('Position(y)', fontsize=7)
ax.set_title("V(x,y) at time t = 25 using FFT")
plt.show()


# Problem 2
n = 30
L = 20
m2 = 2
alpha2 = 1


# Chebyshev Differentiation Matrix
[D, x] = cheb(n)
# Second derivative
D2 = D@D
print(D.shape)

# Need to scale Chebyshev points and the Chebyshev derivative matrix so that the domain is [-1, 1]
# Means multiply x*L/2 and D2*4/L^2
x = x*L/2
D2 = 4/(L**2)*D2
# Delete last row, first row, last col, first col for BCs
D2 = D2[1:-1, 1:-1]
x2 = x[1:-1]
y2 = x2.copy()
X2, Y2 = np.meshgrid(x2, y2)


# Laplacian matrix.
I = np.eye(len(D2))
Lap = np.kron(D2,I)+np.kron(I,D2);


# Initial conditions
u2 = (np.tanh(np.sqrt(X2**2+Y2**2))-alpha2)*np.cos(m2*np.angle(X2+1j*Y2) - np.sqrt(X2**2+Y2**2))
v2 = (np.tanh(np.sqrt(X2**2+Y2**2))-alpha2)*np.sin(m2*np.angle(X2+1j*Y2) - np.sqrt(X2**2+Y2**2))
UV_initial2 = np.concatenate((u2.T.reshape((n-1)**2), v2.T.reshape((n-1)**2)))
# Solve using RK45
# Solve the PDE
def cheb_RHS (t, uv_arr):
    # Convert into matrix shape and back to space (inverse FFT)
    u = uv_arr[0:(n-1)**2]
    v = uv_arr[(n-1)**2:]
    u_matrix = u.reshape(n-1, n-1).T
    v_matrix = v.reshape(n-1, n-1).T
    # Nonlinear terms
    A = u_matrix**2 + v_matrix**2
    lam = 1 - A
    omega = -Beta * A
    # RHS
    ut = ((lam * u_matrix) - (omega * v_matrix)).T.reshape((n-1)**2) + D_1*Lap@u
    vt = ((omega * u_matrix) - (lam * v_matrix)).T.reshape((n-1)**2) + D_2*Lap@v
    return np.concatenate((ut, vt))

uv_vector_sol_2 = scipy.integrate.solve_ivp(lambda t, uv: cheb_RHS(t, uv), t_range, t_eval=t, y0=UV_initial2, method="RK45")
# Sol for V
V_sol_matrix = uv_vector_sol_2.y[(n-1)**2:, -1].reshape(n-1, n-1).T
# Add columns of zeroes to front and end
V_sol_matrix = np.append(V_sol_matrix, np.zeros((n-1, 1)), axis=1)
V_sol_matrix = np.concatenate((np.zeros((n-1, 1)), V_sol_matrix), axis=1)
# Add rows of zeroes to front and end
V_sol_matrix = np.append(V_sol_matrix, np.zeros((1, n+1)), axis=0)
V_sol_matrix = np.vstack((np.zeros((1, n+1)), V_sol_matrix))
# Sol for U
U_sol_matrix = uv_vector_sol_2.y[0:(n-1)**2, -1].reshape(n-1, n-1).T
# Add columns of zeroes to front and end
U_sol_matrix = np.append(U_sol_matrix, np.zeros((n-1, 1)), axis=1)
U_sol_matrix = np.concatenate((np.zeros((n-1, 1)), U_sol_matrix), axis=1)
# Add rows of zeroes to front and end
U_sol_matrix = np.append(U_sol_matrix, np.zeros((1, n+1)), axis=0)
U_sol_matrix = np.vstack((np.zeros((1, n+1)), U_sol_matrix))



A10 = Lap.copy()
A11 = Y2.copy()
A12 = v2.copy()
A13 = UV_initial2.copy().reshape((n-1)**2*2, 1)
A14 = uv_vector_sol_2.y.T
A15 = (uv_vector_sol_2.y[(n-1)**2:, 4]).reshape((n-1)**2, 1)
A16 = V_sol_matrix.copy()

# Plot Results
fig2, ax2 = plt.subplots()
X2_full, Y2_full = np.meshgrid(x, x)
surf = ax2.contourf(X2_full, Y2_full, A16, cmap=cm.winter)
fig2.colorbar(surf, orientation='horizontal')
ax2.set_xlabel('Position (x)', fontsize=7)
ax2.set_ylabel('Position(y)', fontsize=7)
ax2.set_title("V(x,y) at time t = 25 using Chebyshev")
plt.show()

fig2, ax2 = plt.subplots()
X2_full, Y2_full = np.meshgrid(x, x)
surf = ax2.contourf(X2_full, Y2_full, U_sol_matrix, cmap=cm.winter)
fig2.colorbar(surf, orientation='horizontal')
ax2.set_xlabel('Position (x)', fontsize=7)
ax2.set_ylabel('Position(y)', fontsize=7)
ax2.set_title("U(x,y) at time t = 25 using Chebyshev")
plt.show()