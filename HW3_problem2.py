import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
from scipy.sparse import spdiags
from scipy.sparse.linalg import splu
import time

m = 64 # N value in x and y directions
n = m*m # total size of Laplacian matrix
v = 0.001
L = 10
x_eval_2 = np.arange(-L, L, 2*L/m)
print(x_eval_2)
y_eval_2 = np.arange(-L, L, 2*L/m)
t_range = np.array([0, 4])
dt = 0.5
# h is the step from x to x+1 (same as y to y+1)
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

# ODEs w/ GE and LU to solve for phi
# We will derive IC for phi(p) a time t1 using A and w at time t1, ivp solver will step w forward
def ode_1_GE(t, w):
    p = scipy.sparse.linalg.spsolve(A_2, w)
    w_t = v*(A_2*w) - ((B*p) * (C*w) - (C*p) * (B*w))
    return w_t

PLU = splu(A_2)
def ode_1_LU(t, w):
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
sol_GE = scipy.integrate.solve_ivp(lambda t, w: ode_1_GE(t, w), t_range, t_eval=t_eval, y0=w0, method="RK45")
tGEf = time.time()
tLU0 = time.time()
sol_LU = scipy.integrate.solve_ivp(lambda t, w: ode_1_LU(t, w), t_range, t_eval=t_eval, y0=w0, method="RK45")
tLUf = time.time()
print('time for GE: ' + np.str(tGEf-tGE0))
print('time for LU: ' + np.str(tLUf-tLU0))
GE_sol = np.zeros((9, n))
LU_sol = np.zeros((9, n))
GE_sol = sol_GE.y.T
LU_sol = sol_LU.y.T


A4 = np.copy(A_2.todense())
A5 = np.copy(B.todense())
A6 = np.copy(C.todense())
A7 = np.copy(GE_sol)
A8 = np.copy(LU_sol)
A9 = np.copy(LU_sol.reshape((9, 64, 64)))


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
fig10, ax10 = plt.subplots()
fig11, ax11 = plt.subplots()
fig12, ax12 = plt.subplots()
fig13, ax13 = plt.subplots()
fig14, ax14 = plt.subplots()
fig15, ax15 = plt.subplots()
fig16, ax16 = plt.subplots()
X, Y = np.meshgrid(x_eval_2, y_eval_2)
surf1 = ax1.contourf(X, Y, GE_sol[0].reshape([64, 64]).T)
surf2 = ax2.contourf(X, Y, GE_sol[1].reshape([64, 64]).T)
surf3 = ax3.contourf(X, Y, GE_sol[2].reshape([64, 64]).T)
surf4 = ax4.contourf(X, Y, GE_sol[3].reshape([64, 64]).T)
surf5 = ax5.contourf(X, Y, GE_sol[4].reshape([64, 64]).T)
surf6 = ax6.contourf(X, Y, GE_sol[5].reshape([64, 64]).T)
surf7 = ax7.contourf(X, Y, GE_sol[6].reshape([64, 64]).T)
surf8 = ax8.contourf(X, Y, GE_sol[7].reshape([64, 64]).T)
surf9 = ax9.contourf(X, Y, GE_sol[8].reshape([64, 64]).T)
surf10 = ax10.contourf(X, Y, GE_sol[9].reshape([64, 64]).T)
surf11 = ax11.contourf(X, Y, GE_sol[10].reshape([64, 64]).T)
surf12 = ax12.contourf(X, Y, GE_sol[11].reshape([64, 64]).T)
surf13 = ax13.contourf(X, Y, GE_sol[12].reshape([64, 64]).T)
surf14 = ax14.contourf(X, Y, GE_sol[13].reshape([64, 64]).T)
surf15 = ax15.contourf(X, Y, GE_sol[14].reshape([64, 64]).T)
surf16 = ax16.contourf(X, Y, GE_sol[15].reshape([64, 64]).T)


fig1.colorbar(surf1)
ax1.set_xlabel('t= ' + np.str(t_eval[0]), fontsize='8')
plt.show()
fig2.colorbar(surf2)
ax2.set_xlabel('t= ' + np.str(t_eval[1]), fontsize='8')
plt.show()
fig3.colorbar(surf3)
ax3.set_xlabel('t= ' + np.str(t_eval[2]), fontsize='8')
plt.show()
fig4.colorbar(surf4)
ax4.set_xlabel('t= ' + np.str(t_eval[3]), fontsize='8')
plt.show()
fig5.colorbar(surf5)
ax5.set_xlabel('t= ' + np.str(t_eval[4]), fontsize='8')
plt.show()
fig6.colorbar(surf6)
ax6.set_xlabel('t= ' + np.str(t_eval[5]), fontsize='8')
plt.show()
fig7.colorbar(surf7)
ax7.set_xlabel('t= ' + np.str(t_eval[6]), fontsize='8')
plt.show()
fig8.colorbar(surf8)
ax8.set_xlabel('t= ' + np.str(t_eval[7]), fontsize='8')
plt.show()
fig9.colorbar(surf9)
ax9.set_xlabel('t= ' + np.str(t_eval[8]), fontsize='8')
plt.show()
fig10.colorbar(surf10)
ax10.set_xlabel('t= ' + np.str(t_eval[9]), fontsize='8')
plt.show()
fig11.colorbar(surf11)
ax11.set_xlabel('t= ' + np.str(t_eval[10]), fontsize='8')
plt.show()
fig12.colorbar(surf12)
ax12.set_xlabel('t= ' + np.str(t_eval[11]), fontsize='8')
plt.show()
fig13.colorbar(surf13)
ax13.set_xlabel('t= ' + np.str(t_eval[12]), fontsize='8')
plt.show()
fig14.colorbar(surf14)
ax14.set_xlabel('t= ' + np.str(t_eval[13]), fontsize='8')
plt.show()
fig15.colorbar(surf15)
ax15.set_xlabel('t= ' + np.str(t_eval[14]), fontsize='8')
plt.show()
fig16.colorbar(surf16)
ax16.set_xlabel('t= ' + np.str(t_eval[15]), fontsize='8')
plt.show()


# Testing

# This should be the correct way to stack w
# z = lambda x, y: np.int(str(x) + str(y))
# x = np.array([1, 2, 3])
# y = ([4, 5, 6])
# xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
# for i in range(len(x)):
#     for j in range(len(y)):
#         xv[i, j] = z(x[i], y[j])
# print(xv)
# print(xv.reshape(-1, 1))
#
# def grid_it(f, x, y):
#     xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')
#     for i in range(len(x)):
#         for j in range(len(y)):
#             xv[i, j] = f(x[i], y[j])
#     return(xv)
#
# def test_f(x, y):
#     return np.sin(x) + np.cos(y)
#
# solnnn = grid_it(test_f, x_eval_2, y_eval_2)
#
# fig2 = plt.figure()
# ax2 = plt.axes(projection='3d')
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# X, Y = np.meshgrid(x_eval_2, y_eval_2)
# surf = ax2.plot_surface(X, Y, solnnn, cmap=cm.twilight, rstride=1, cstride=1)
# fig2.colorbar(surf, orientation='horizontal', fraction=0.05)
# ax2.set_xlabel('x', fontsize=7)
# ax2.set_ylabel('y', fontsize=7)
# ax2.set_zlabel('f(x, y)', fontsize=7)
# plt.show()
