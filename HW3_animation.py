import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
from matplotlib import animation, rc


m = 64 # N value in x and y directions
n = m*m # total size of Laplacian matrix
v = 0.001
L = 10
x_eval_2 = np.arange(-L, L, 2*L/m)
y_eval_2 = np.arange(-L, L, 2*L/m)
t_range = np.array([0, 4])
dt = 0.05
# h is the step from x to x+1 (same as y to y+1) = 2L/m
h = np.abs(x_eval_2[1] - x_eval_2[0])
t_eval = np.arange(0, 4+dt, dt)
w_xy0 = lambda x, y: np.e**(-2*x**2 - ((y**2)/20))


# Creating a Laplacian Matrix A (from example code)

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

# Create Matrix B (dw/dx)
B = (1/(2*h)) * scipy.sparse.spdiags([e1, -1*e1, e1, -1*e1], [-(n-m), -m, m, n-m], n, n, format='csc')

# Create Matrix C (dw/dy)
e0_up = np.tile(np.concatenate(((np.zeros(m-1)), [-1])), (m,))
e0_low = np.tile(np.concatenate(([1], (np.zeros(m-1)))), (m,))
e1_up = np.tile(np.concatenate(([0], (np.ones(m-1)))), (m,))
e1_low = np.tile(np.concatenate((np.ones(m-1), [0])), (m,))
C = (1/(2*h)) * scipy.sparse.spdiags([e0_low, -1*e1_low, e1_up, e0_up], [-(m-1), -1, 1, m-1], n, n, format='csc')

# ODEs w/ LU to solve for phi
# We will derive IC for phi(p) a time t1 using A and w at time t1, ivp solver will step w forward

PLU = splu(A_2)
def ode_LU(t, w):
    p = PLU.solve(w)
    w_t = v*(A_2*w) - ((B*p) * (C*w) - (C*p) * (B*w))
    return w_t

# IC for omega(w) at t=0
# Using loop
i = 0
w0 = np.zeros(n)
for x in x_eval_2:
    for y in y_eval_2:
        w0[i] = w_xy0(x, y)
        i += 1

sol_LU = scipy.integrate.solve_ivp(lambda t, w: ode_LU(t, w), t_range, t_eval=t_eval, y0=w0, method="RK45")
w = np.copy(sol_LU.y.T)


# Set up initial blank frame
rc('animation', html='html5')
fig, ax = plt.subplots()
X, Y = np.meshgrid(x_eval_2, y_eval_2)
surface = ax.contourf(X, Y, w[0].reshape([m, m]).T)
title = ax.text(0.5, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad':0.5}, transform=ax.transAxes, ha='center')
def init():
    ax.set_xlim(x_eval_2[0], x_eval_2[-1])
    ax.set_ylim(y_eval_2[0], y_eval_2[-1])
    ax.set_xlabel('Position (x)')
    ax.set_ylabel('Position (y)')
    return surface

# Update on each frame
def update(frame):
    ax.set_title("Vorticity $\omega$ (x, y t) at time t = %0.2f" %t_eval[frame])
    ax.contourf(X, Y, w[frame].reshape([m, m]).T)
anim = animation.FuncAnimation(fig, update, init_func=init, frames = len(t_eval), interval=20, blit=False)
fig.colorbar(surface)
writermp4 = animation.FFMpegWriter(fps=10)
plt.show()
anim.save('vorticity.mp4', writer=writermp4)

