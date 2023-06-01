import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.fft import fft, ifft, fftshift
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
from matplotlib import animation, rc
from matplotlib.animation import FFMpegWriter
import sys
from IPython.display import HTML
import time


# Problem 1

L = 10
u_x0 = lambda x: np.e**(-(x-5)**2)
x_eval = np.arange(-L, L, 0.1)
dx = x_eval[1]-x_eval[0]
t_eval = np.arange(0, L+0.5, 0.5)

# a)
# Create Matrix A = u_x
n = len(x_eval)
e0 = np.negative(np.ones(n))
e1 = np.zeros(n)
e2 = np.ones(n)
Bin = np.array([(1/(2*dx)) * e0, e1, (1/(2*dx)) * e2])
d = np.array([-1, 0, 1])
A = spdiags(Bin, d, n, n, format='csc') # Use "format='csc'" to make solving faster
A[0, n-1] = -1/(2*dx)
A[n-1, 0] = 1/(2*dx)

# b)
c = -0.5
def advection_pde(t, u, A):
    u_t = (-c*A)@u
    return u_t

sol = scipy.integrate.solve_ivp(lambda t, y: advection_pde(t, y, A), t_span= [0, L], t_eval= t_eval, y0=u_x0(x_eval))
# X, T = np.meshgrid(x_eval, sol.t)
# fig, ax = plt.subplots(subplot_kw = {"projection" : "3d"}, figsize=(15, 8) )
# surf = ax.plot_surface(X, T, sol.y.T, cmap='magma')
# ax.plot3D(x_eval, 0*x_eval, u_x0(x_eval), '-r', linewidth=5)
# plt.xlabel=('x')
# plt.ylabel=('time')
# plt.show()

A1 = np.copy(A.todense())
A2 = sol.y
A3 = 0



# Problem 2

m = 64 # N value in x and y directions
n = m*m # total size of Laplacian matrix
v = 0.001
L = 10
x_eval_2 = np.arange(-L, L, 2*L/m)
y_eval_2 = np.arange(-L, L, 2*L/m)
t_range = np.array([0, 4])
dt = 0.5
# h is the step from x to x+1 (same as y to y+1) = 2L/m
h = np.abs(x_eval_2[1] - x_eval_2[0])
t_eval = np.arange(0, 4+dt, dt)
w_xy0 = lambda x, y: np.e**(-2*x**2 - ((y**2)/20))


# Creating a Laplacian Matrix A_2 (from example code)

e1 = np.ones(n) # vector of ones
Low1 = np.tile(np.concatenate((np.ones(m-1), [0])), (m,)) # Lower diagonal 1
Low2 = np.tile(np.concatenate(([1], np.zeros(m-1))), (m,)) #Lower diagonal 2
                                    # Low2 is NOT on the second lower diagonal,
                                    # it is just the next lower diagonal we see
                                    # in the matrix.

Up1 = np.roll(Low1, 1) # Shift the array for spdiags
Up2 = np.roll(Low2, m-1) # Shift the other array

A_2 = (1/(h**2)) * scipy.sparse.spdiags([e1, e1, Low2, Low1, np.append(np.array([2]), -4*e1[1:]), Up1, Up2, e1, e1],
                         [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)], n, n, format='csc')

# Create Matrix B for first x spatial derivative (dw/dx)
B = (1/(2*h)) * scipy.sparse.spdiags([e1, -1*e1, e1, -1*e1], [-(n-m), -m, m, n-m], n, n, format='csc')

# Create Matrix C for first y spatial derivative (dw/dy)
e0_up = np.tile(np.concatenate(((np.zeros(m-1)), [-1])), (m,))
e0_low = np.tile(np.concatenate(([1], (np.zeros(m-1)))), (m,))
e1_up = np.tile(np.concatenate(([0], (np.ones(m-1)))), (m,))
e1_low = np.tile(np.concatenate((np.ones(m-1), [0])), (m,))
C = (1/(2*h)) * scipy.sparse.spdiags([e0_low, -1*e1_low, e1_up, e0_up], [-(m-1), -1, 1, m-1], n, n, format='csc')


# IC for omega (w) at t=0
# Using loop
i = 0
w0 = np.zeros(n)
for x in x_eval_2:
    for y in y_eval_2:
        w0[i] = w_xy0(x, y)
        i += 1


# ODEs w/ GE and LU to solve for phi
# We will derive IC for phi(p) a time t1 using A and w at time t1, ivp solver will step w forward
def ode_GE(t, w):
    p = scipy.sparse.linalg.spsolve(A_2, w)
    w_t = v*(A_2*w) - ((B*p) * (C*w) - (C*p) * (B*w))
    return w_t

PLU = splu(A_2)
def ode_LU(t, w):
    p = PLU.solve(w)
    w_t = v*(A_2*w) - ((B*p) * (C*w) - (C*p) * (B*w))
    return w_t




# Using grid
# def grid_it(f, x, y):
#     xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#     for i in range(len(x)):
#         for j in range(len(y)):
#             xv[i, j] = f(x[i], y[j])
#     return(xv)
# w0 = grid_it(w_xy0, x_eval_2, y_eval_2)
# w0 = w0.reshape(-1, 1).flatten()

# Time step linear solve for ODES with Ap = w included to solve for p at that time

tGE0 = time.time()
sol_GE = scipy.integrate.solve_ivp(lambda t, w: ode_GE(t, w), t_range, t_eval=t_eval, y0=w0, method="RK45")
tGEf = time.time()
tLU0 = time.time()
sol_LU = scipy.integrate.solve_ivp(lambda t, w: ode_LU(t, w), t_range, t_eval=t_eval, y0=w0, method="RK45")
tLUf = time.time()
print('time for GE: ' + np.str(tGEf-tGE0))
print('time for LU: ' + np.str(tLUf-tLU0))
print(sol_LU.y.shape)
grid_sol = sol_LU.y.T.reshape((9, m, m))

A4 = np.copy(A_2.todense())
A5 = np.copy(B.todense())
A6 = np.copy(C.todense())
A7 = np.copy(sol_GE.y.T)
A8 = np.copy(sol_LU.y.T)
A9 = (np.copy(A8)).reshape((9, m, m))


# VORTICITY PLOTS (see Fig 14 in notes for comparison)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
fig4, ax4 = plt.subplots()
fig5, ax5 = plt.subplots()
fig6, ax6 = plt.subplots()
fig7, ax7 = plt.subplots()
fig8, ax8 = plt.subplots()
fig9, ax9 = plt.subplots()

X, Y = np.meshgrid(x_eval_2, y_eval_2)

surf1 = ax1.contourf(X, Y, A7[0].reshape([m, m]).T)
surf2 = ax2.contourf(X, Y, A7[1].reshape([m, m]).T)
surf3 = ax3.contourf(X, Y, A7[2].reshape([m, m]).T)
surf4 = ax4.contourf(X, Y, A7[3].reshape([m, m]).T)
surf5 = ax5.contourf(X, Y, A7[4].reshape([m, m]).T)
surf6 = ax6.contourf(X, Y, A7[5].reshape([m, m]).T)
surf7 = ax7.contourf(X, Y, A7[6].reshape([m, m]).T)
surf8 = ax8.contourf(X, Y, A7[7].reshape([m, m]).T)
surf9 = ax9.contourf(X, Y, A7[8].reshape([m, m]).T)

fig1.colorbar(surf1)
ax1.set_xlabel('x Position')
ax1.set_ylabel('y Position')
ax1.set_title('Vorticity at time t= ' + np.str(t_eval[0]), fontsize='12')

fig2.colorbar(surf2)
ax2.set_xlabel('x Position')
ax2.set_ylabel('y Position')
ax2.set_title('Vorticity at time t= ' + np.str(t_eval[1]), fontsize='12')

fig3.colorbar(surf3)
ax3.set_xlabel('x Position')
ax3.set_ylabel('y Position')
ax3.set_title('Vorticity at time t= ' + np.str(t_eval[2]), fontsize='12')

fig4.colorbar(surf4)
ax4.set_xlabel('x Position')
ax4.set_ylabel('y Position')
ax4.set_title('Vorticity at time t= ' + np.str(t_eval[3]), fontsize='12')

fig5.colorbar(surf5)
ax5.set_xlabel('x Position')
ax5.set_ylabel('y Position')
ax5.set_title('Vorticity at time t= ' + np.str(t_eval[4]), fontsize='12')

fig6.colorbar(surf6)
ax6.set_xlabel('x Position')
ax6.set_ylabel('y Position')
ax6.set_title('Vorticity at time t= ' + np.str(t_eval[5]), fontsize='12')

fig7.colorbar(surf7)
ax7.set_xlabel('x Position')
ax7.set_ylabel('y Position')
ax7.set_title('Vorticity at time t= ' + np.str(t_eval[6]), fontsize='12')

fig8.colorbar(surf8)
ax8.set_xlabel('x Position')
ax8.set_ylabel('y Position')
ax8.set_title('Vorticity at time t= ' + np.str(t_eval[7]), fontsize='12')

fig9.colorbar(surf9)
ax9.set_xlabel('x Position')
ax9.set_ylabel('y Position')
ax9.set_title('Vorticity at time t= ' + np.str(t_eval[8]), fontsize='12')

plt.show()


# # Animation
#
# dt = 0.5
# t = np.arange(0, 4+dt, dt)
# trange = np.array([0, len(t)])
# L = 10
# N = 64
# x = np.arange(-L, L, 2*L/N)
# y = np.arange(-L, L, 2*L/N)
# w0 = np.zeros(N**2)
# index = 0
# for i in x:
#     for j in y:
#         w0[int(i)] = w_xy0(i, j)
#         index += 1
#
# PLU = splu(A_2)
# def ode_LU(t, w):
#     p = PLU.solve(w)
#     w_t = v*(A_2*w) - ((B*p) * (C*w) - (C*p) * (B*w))
#     return w_t
#
#
# w = np.copy(A7)
#
# # Animate plot
# # Set up initial blank frame
# rc('animation', html='html5')
# fig, ax = plt.subplots()
# X, Y = np.meshgrid(x, y)
# # ax.set_xlim(-10, 10)
# # ax.set_ylim(-10, 10)
# surface = ax.contourf(X, Y, w[0].reshape([N, N]).T)
# title = ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad':0.5}, transform=ax.transAxes, ha='center')
# def init():
#     ax.set_xlim(x[0], x[-1])
#     ax.set_ylim(y[0], y[-1])
#     ax.set_xlabel('Position (x)')
#     ax.set_ylabel('Position (y)')
#     return surface
# # Update on each frame
# def update(frame):
#     ax.set_title("Vorticity at time t = %0.2f" %t[frame])
#     ax.contourf(X, Y, w[frame].reshape([N, N]).T)
# anim = animation.FuncAnimation(fig, update, init_func=init, frames = len(t), interval=500, blit=False)
# fig.colorbar(surface)
# plt.show()
# writermp4 = animation.FFMpegWriter(fps=10)
#
# anim.save('animationattempt.mp4', writer=writermp4)
#
# plt.show()








