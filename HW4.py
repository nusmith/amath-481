import numpy as np
import scipy
from scipy import optimize
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu

# Constants for all problems
L = 10
a = 2 # alpha
t_eval = np.linspace(0, 2, 501)
dt = t_eval[1] - t_eval[0]
x = np.linspace(-L, L, 129)
x_eval = x[0:128]
dx = np.abs(x_eval[1] - x_eval[0])
u_x0 = lambda x: 10*np.cos(2*np.pi*x/L) + 30*np.cos(8*np.pi*x/L)
cfl = a*dt/dx**2 # CFL number aka lambda


# Problem 1
# Magnification factor derived in notes.
g_1 = lambda z : 1 + cfl/6 * (-np.cos(2*z) + 16*np.cos(z) - 15)
magnification_factor_1 = g_1(1)
t_max = np.abs(scipy.optimize.fminbound(lambda x: -1*np.abs(g_1(x)), -np.pi, np.pi))
max_abs_g = g_1(t_max)
# Discretization of spatial derivative function including periodic boundary conditions
e1 = np.ones(128)
D4 = (1/12) * scipy.sparse.spdiags([16*e1, -e1, -e1, 16*e1, -30*e1, 16*e1, -e1, -e1, 16*e1],
                         [-127, -126, -2, -1, 0, 1, 2, 126, 127], 128, 128, format='csc')

# Solve
u_sol_FE = np.zeros([len(x_eval), len(t_eval)])
u_sol_FE[:, 0] = u_x0(x_eval)
du_dt = (cfl*D4) # This is the time derivative discretization used to time step in forward euler (includes dt)
for t in range(1, len(t_eval)):
    u_sol_FE[:, t] = u_sol_FE[:, t-1] + cfl*D4*u_sol_FE[:, t-1]
# Submission
A1 = np.abs(magnification_factor_1)
A2 = max_abs_g
A3 = np.copy(D4.todense())
A4 = 0
A5 = u_sol_FE[:, -1].reshape([128, 1])


# Problem 2
# Discretization
e2 = np.ones(128)
B = scipy.sparse.spdiags([(-cfl/2)*e2, (-cfl/2)*e2, (1+cfl)*e2, (-cfl/2)*e2, (-cfl/2)*e2], [-127, -1, 0, 1, 127], 128, 128, format='csc')
C = scipy.sparse.spdiags([(cfl/2)*e2, (cfl/2)*e2, (1-cfl)*e2, (cfl/2)*e2, (cfl/2)*e2], [-127, -1, 0, 1, 127], 128, 128, format='csc')
# Solve
# Bu^m+1 = Cu^m
u_sol_CN = np.zeros([len(x_eval), len(t_eval)])
u_sol_CN[:, 0] = u_x0(x_eval)
for t in range(1, len(t_eval)):
    PLU_B = scipy.sparse.linalg.splu(B)
    Cum = C*u_sol_CN[:, t-1]
    u_next = PLU_B.solve(Cum)
    u_sol_CN[:, t] = u_next
u_sol_CN_BICGSTAB = np.zeros([len(x_eval), len(t_eval)])
u_sol_CN_BICGSTAB[:, 0] = u_x0(x_eval)
for t in range(1, len(t_eval)):
    Guess = C*u_sol_CN_BICGSTAB[:, t-1]
    u_next, success_0 = scipy.sparse.linalg.bicgstab(B, Guess)
    u_sol_CN_BICGSTAB[:, t] = u_next

# Submission
A6 = 1  # Based on derivation of magnification factor, g = 1 always for the trap derivative method
A7 = np.copy(B.todense())
A8 = np.copy(C.todense())
A9 = u_sol_CN[:, -1].reshape([128, 1])
A10 = u_sol_CN_BICGSTAB[:, -1].reshape([128, 1])


# Problem 3

exact_128 = np.genfromtxt('exact_128.csv')
exact_256 = np.genfromtxt('exact_256.csv')

L = 10
a = 2 # alpha
# Same cfl number
x_256 = np.linspace(-L, L, 257)
x_eval_256 = x_256[0:256]
dx_256 = np.abs(x_eval_256[1] - x_eval_256[0])
dt_256 = cfl*dx_256**2/a
t_eval = np.arange(0, 2+dt_256, dt_256)
# SAME CFL number aka lambda
u_x0 = lambda x: 10*np.cos(2*np.pi*x/L) + 30*np.cos(8*np.pi*x/L)

# FE
e1_256 = np.ones(256)
D4_256 = (1/12) * scipy.sparse.spdiags([16*e1_256, -e1_256, -e1_256, 16*e1_256, -30*e1_256, 16*e1_256, -e1_256, -e1_256, 16*e1_256],
                         [-255, -254, -2, -1, 0, 1, 2, 254, 255], 256, 256, format='csc')
u_sol_FE_256 = np.zeros([len(x_eval_256), len(t_eval)])
u_sol_FE_256[:, 0] = u_x0(x_eval_256)
du_dt = (cfl*D4_256) # This is the time derivative discretization used to time step in forward euler (includes dt)
for t in range(1, len(t_eval)):
    u_sol_FE_256[:, t] = u_sol_FE_256[:, t-1] + cfl*D4_256*u_sol_FE_256[:, t-1]

# CN
e2_256 = np.ones(256)
B_256 = scipy.sparse.spdiags([(-cfl/2)*e2_256, (-cfl/2)*e2_256, (1+cfl)*e2_256, (-cfl/2)*e2_256, (-cfl/2)*e2_256], [-255, -1, 0, 1, 255], 256, 256, format='csc')
C_256 = scipy.sparse.spdiags([(cfl/2)*e2_256, (cfl/2)*e2_256, (1-cfl)*e2_256, (cfl/2)*e2_256, (cfl/2)*e2_256], [-255, -1, 0, 1, 255], 256, 256, format='csc')
# Solve
# Bu^m+1 = Cu^m
u_sol_CN_256 = np.zeros([len(x_eval_256), len(t_eval)])
u_sol_CN_256[:, 0] = u_x0(x_eval_256)
for t in range(1, len(t_eval)):
    PLU_B = scipy.sparse.linalg.splu(B_256)
    Cum = C_256*u_sol_CN_256[:, t-1]
    u_next = PLU_B.solve(Cum)
    u_sol_CN_256[:, t] = u_next

# Submission
A11 = np.linalg.norm(u_sol_FE[:, -1] - exact_128)
A12 = np.linalg.norm(u_sol_CN[:, -1] - exact_128)
A13 = np.linalg.norm(u_sol_FE_256[:, -1] - exact_256)
A14 = np.linalg.norm(u_sol_CN_256[:, -1] - exact_256)