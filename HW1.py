import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

# PROBLEM 1

y_true = lambda t: np.pi * np.e ** (3 * (np.cos(t) - 1)) / np.sqrt(2)
y0 = np.pi / np.sqrt(2)
y_prime = lambda t, y: -3 * y * np.sin(t)
# y1 for Adams predictor-corrector as a function of dt
y1 = lambda dt: y0 + dt * y_prime(dt / 2, y0 + (dt / 2) * y_prime(0, y0))

# delta t = 2^-2
t2 = np.arange(0, 5 + 2 ** (-2), 2 ** (-2))
dt2 = 2 ** (-2)
y2_fe = np.zeros(len(t2))
y2_fe[0] = y0
for i in range(1, len(t2)):
    y2_fe[i] = y2_fe[i - 1] + dt2 * y_prime(t2[i - 1], y2_fe[i - 1])
E2_fe = np.abs(y_true(5) - y2_fe[-1])
y2_h = np.zeros(len(t2))
y2_h[0] = y0
for i in range(1, len(t2)):
    y2_h[i] = y2_h[i - 1] + (dt2 / 2) * (y_prime(t2[i - 1], y2_h[i - 1]) +
                                         y_prime(t2[i], y2_h[i - 1] + dt2 * y_prime(t2[i - 1], y2_h[i - 1])))
E2_h = np.abs(y_true(5) - y2_h[-1])
y2_pc = np.zeros(len(t2))
y2_pc[0] = y0
y2_pc[1] = y1(dt2)
for i in range(2, len(t2)):
    p = y2_pc[i - 1] + (dt2 / 2) * (3 * y_prime(t2[i - 1], y2_pc[i - 1]) - y_prime(t2[i - 2], y2_pc[i - 2]))
    y2_pc[i] = y2_pc[i - 1] + (dt2 / 2) * (y_prime(t2[i], p) + y_prime(t2[i - 1], y2_pc[i - 1]))
E2_pc = np.abs(y_true(5) - y2_pc[-1])


# delta t = 2^-3
t3 = np.arange(0, 5 + 2 ** (-3), 2 ** (-3))
dt3 = 2 ** (-3)
y3_fe = np.zeros(len(t3))
y3_fe[0] = y0
for i in range(1, len(t3)):
    y3_fe[i] = y3_fe[i - 1] + dt3 * y_prime(t3[i - 1], y3_fe[i - 1])
E3_fe = np.abs(y_true(5) - y3_fe[-1])
y3_h = np.zeros(len(t3))
y3_h[0] = y0
for i in range(1, len(t3)):
    y3_h[i] = y3_h[i - 1] + (dt3 / 2) * (y_prime(t3[i - 1], y3_h[i - 1]) +
                                         y_prime(t3[i], y3_h[i - 1] + dt3 * y_prime(t3[i - 1], y3_h[i - 1])))
E3_h = np.abs(y_true(5) - y3_h[-1])
y3_pc = np.zeros(len(t3))
y3_pc[0] = y0
y3_pc[1] = y1(dt3)
for i in range(2, len(t3)):
    p = y3_pc[i - 1] + (dt3 / 2) * (3 * y_prime(t3[i - 1], y3_pc[i - 1]) - y_prime(t3[i - 2], y3_pc[i - 2]))
    y3_pc[i] = y3_pc[i - 1] + (dt3 / 2) * (y_prime(t3[i], p) + y_prime(t3[i - 1], y3_pc[i - 1]))
E3_pc = np.abs(y_true(5) - y3_pc[-1])


# delta t = 2^-4
t4 = np.arange(0, 5 + 2 ** (-4), 2 ** (-4))
dt4 = 2 ** (-4)
y4_fe = np.zeros(len(t4))
y4_fe[0] = y0
for i in range(1, len(t4)):
    y4_fe[i] = y4_fe[i - 1] + dt4 * y_prime(t4[i - 1], y4_fe[i - 1])
E4_fe = np.abs(y_true(5) - y4_fe[-1])
y4_h = np.zeros(len(t4))
y4_h[0] = y0
for i in range(1, len(t4)):
    y4_h[i] = y4_h[i - 1] + (dt4 / 2) * (y_prime(t4[i - 1], y4_h[i - 1]) +
                                         y_prime(t4[i], y4_h[i - 1] + dt4 * y_prime(t4[i - 1], y4_h[i - 1])))
E4_h = np.abs(y_true(5) - y4_h[-1])
y4_pc = np.zeros(len(t4))
y4_pc[0] = y0
y4_pc[1] = y1(dt4)
for i in range(2, len(t4)):
    p = y4_pc[i - 1] + (dt4 / 2) * (3 * y_prime(t4[i - 1], y4_pc[i - 1]) - y_prime(t4[i - 2], y4_pc[i - 2]))
    y4_pc[i] = y4_pc[i - 1] + (dt4 / 2) * (y_prime(t4[i], p) + y_prime(t4[i - 1], y4_pc[i - 1]))
E4_pc = np.abs(y_true(5) - y4_pc[-1])


# delta t = 2^-5
t5 = np.arange(0, 5 + 2 ** (-5), 2 ** (-5))
dt5 = 2 ** (-5)
y5_fe = np.zeros(len(t5))
y5_fe[0] = y0
for i in range(1, len(t5)):
    y5_fe[i] = y5_fe[i - 1] + dt5 * y_prime(t5[i - 1], y5_fe[i - 1])
E5_fe = np.abs(y_true(5) - y5_fe[-1])
y5_h = np.zeros(len(t5))
y5_h[0] = y0
for i in range(1, len(t5)):
    y5_h[i] = y5_h[i - 1] + (dt5 / 2) * (y_prime(t5[i - 1], y5_h[i - 1]) +
                                         y_prime(t5[i], y5_h[i - 1] + dt5 * y_prime(t5[i - 1], y5_h[i - 1])))
E5_h = np.abs(y_true(5) - y5_h[-1])
y5_pc = np.zeros(len(t5))
y5_pc[0] = y0
y5_pc[1] = y1(dt5)
for i in range(2, len(t5)):
    p = y5_pc[i - 1] + (dt5 / 2) * (3 * y_prime(t5[i - 1], y5_pc[i - 1]) - y_prime(t5[i - 2], y5_pc[i - 2]))
    y5_pc[i] = y5_pc[i - 1] + (dt5 / 2) * (y_prime(t5[i], p) + y_prime(t5[i - 1], y5_pc[i - 1]))
E5_pc = np.abs(y_true(5) - y5_pc[-1])


# delta t = 2^-6
t6 = np.arange(0, 5 + 2 ** (-6), 2 ** (-6))
dt6 = 2 ** (-6)
y6_fe = np.zeros(len(t6))
y6_fe[0] = y0
for i in range(1, len(t6)):
    y6_fe[i] = y6_fe[i - 1] + dt6 * y_prime(t6[i - 1], y6_fe[i - 1])
E6_fe = np.abs(y_true(5) - y6_fe[-1])
y6_h = np.zeros(len(t6))
y6_h[0] = y0
for i in range(1, len(t6)):
    y6_h[i] = y6_h[i - 1] + (dt6 / 2) * (y_prime(t6[i - 1], y6_h[i - 1]) +
                                         y_prime(t6[i], y6_h[i - 1] + dt6 * y_prime(t6[i - 1], y6_h[i - 1])))
E6_h = np.abs(y_true(5) - y6_h[-1])
y6_pc = np.zeros(len(t6))
y6_pc[0] = y0
y6_pc[1] = y1(dt6)
for i in range(2, len(t6)):
    p = y6_pc[i - 1] + (dt6 / 2) * (3 * y_prime(t6[i - 1], y6_pc[i - 1]) - y_prime(t6[i - 2], y6_pc[i - 2]))
    y6_pc[i] = y6_pc[i - 1] + (dt6 / 2) * (y_prime(t6[i], p) + y_prime(t6[i - 1], y6_pc[i - 1]))
E6_pc = np.abs(y_true(5) - y6_pc[-1])


# delta t = 2^-7
t7 = np.arange(0, 5 + 2 ** (-7), 2 ** (-7))
dt7 = 2 ** (-7)
y7_fe = np.zeros(len(t7))
y7_fe[0] = y0
for i in range(1, len(t7)):
    y7_fe[i] = y7_fe[i - 1] + dt7 * y_prime(t7[i - 1], y7_fe[i - 1])
E7_fe = np.abs(y_true(5) - y7_fe[-1])
y7_h = np.zeros(len(t7))
y7_h[0] = y0
for i in range(1, len(t7)):
    y7_h[i] = y7_h[i - 1] + (dt7 / 2) * (y_prime(t7[i - 1], y7_h[i - 1]) +
                                         y_prime(t7[i], y7_h[i - 1] + dt7 * y_prime(t7[i - 1], y7_h[i - 1])))
E7_h = np.abs(y_true(5) - y7_h[-1])
y7_pc = np.zeros(len(t7))
y7_pc[0] = y0
y7_pc[1] = y1(dt7)
for i in range(2, len(t7)):
    p = y7_pc[i - 1] + (dt7 / 2) * (3 * y_prime(t7[i - 1], y7_pc[i - 1]) - y_prime(t7[i - 2], y7_pc[i - 2]))
    y7_pc[i] = y7_pc[i - 1] + (dt7 / 2) * (y_prime(t7[i], p) + y_prime(t7[i - 1], y7_pc[i - 1]))
E7_pc = np.abs(y_true(5) - y7_pc[-1])


# delta t = 2^-8
t8 = np.arange(0, 5 + 2 ** (-8), 2 ** (-8))
dt8 = 2 ** (-8)
y8_fe = np.zeros(len(t8))
y8_fe[0] = y0
for i in range(1, len(t8)):
    y8_fe[i] = y8_fe[i - 1] + dt8 * y_prime(t8[i - 1], y8_fe[i - 1])
E8_fe = np.abs(y_true(5) - y8_fe[-1])
y8_h = np.zeros(len(t8))
y8_h[0] = y0
for i in range(1, len(t8)):
    y8_h[i] = y8_h[i - 1] + (dt8 / 2) * (y_prime(t8[i - 1], y8_h[i - 1]) +
                                         y_prime(t8[i], y8_h[i - 1] + dt8 * y_prime(t8[i - 1], y8_h[i - 1])))
E8_h = np.abs(y_true(5) - y8_h[-1])
y8_pc = np.zeros(len(t8))
y8_pc[0] = y0
y8_pc[1] = y1(dt8)
for i in range(2, len(t8)):
    p = y8_pc[i - 1] + (dt8 / 2) * (3 * y_prime(t8[i - 1], y8_pc[i - 1]) - y_prime(t8[i - 2], y8_pc[i - 2]))
    y8_pc[i] = y8_pc[i - 1] + (dt8 / 2) * (y_prime(t8[i], p) + y_prime(t8[i - 1], y8_pc[i - 1]))
E8_pc = np.abs(y_true(5) - y8_pc[-1])


# Find error for each dt
dt_series = np.array([dt2, dt3, dt4, dt5, dt6, dt7, dt8])
err_series_fe = np.array([E2_fe, E3_fe, E4_fe, E5_fe, E6_fe, E7_fe, E8_fe])
err_series_h = np.array([E2_h, E3_h, E4_h, E5_h, E6_h, E7_h, E8_h])
err_series_pc = np.array([E2_pc, E3_pc, E4_pc, E5_pc, E6_pc, E7_pc, E8_pc])

# Plot Log(Error) vs Log(dt)
fig, ax = plt.subplots()
ax.loglog(dt_series, err_series_fe, 'ko', color='#1E69F5', linewidth=3,
        label='Log(E) vs Log(dt) for Euler')
ax.loglog(dt_series, err_series_h, 'ko', color='#17A628', linewidth=3,
        label='Log(E) vs Log(dt) for Heun')
ax.loglog(dt_series, err_series_pc, 'ko', color='#5217A6', linewidth=3,
        label='Log(E) vs Log(dt) for Predictor-Corrector')
coeffs_fe = np.polyfit(np.log(dt_series), np.log(err_series_fe), 1);
coeffs_h = np.polyfit(np.log(dt_series), np.log(err_series_h), 1);
coeffs_pc = np.polyfit(np.log(dt_series), np.log(err_series_pc), 1);
slope_fe = coeffs_fe[0]
slope_h = coeffs_h[0]
slope_pc = coeffs_pc[0]
print(slope_h)
print(slope_pc)

#Plot Line of Best Fit
ax.loglog(dt_series, 2.5*dt_series, color='#B2CDFF', linewidth=3,
        label='Best Fit Line for FE Error (slope=1 for O(1)')
ax.loglog(dt_series, 0.6*dt_series**2, color='#6AD877', linewidth=3,
        label='Best Fit Line for Heun Error (slope=2 for O(2))')
ax.loglog(dt_series, slope_pc*3*dt_series**3, color='#C0A7E4', linewidth=3,
        label='Best Fit Line for Predictor Corrector Error (slope=3 for O(3))')
ax.legend(loc='lower right', fontsize=7)
plt.title('Log(Global Error) vs Log(dt) for Forward Euler, Heun, and Predictor Corrector Methods', fontsize=10)
plt.xlabel('Log(dt)')
plt.ylabel('Log(E)')
plt.show()

A1 = np.reshape(y8_fe, [len(y8_fe), 1])
A2 = np.reshape(err_series_fe, [1, 7])
A3 = slope_fe
A4 = np.reshape(y8_h, [len(y8_h), 1])
A5 = np.reshape(err_series_h, [1, 7])
A6 = slope_h
A7 = np.reshape(y8_pc, [len(y8_pc), 1])
A8 = np.reshape(err_series_pc, [1, 7])
A9 = slope_pc

# PROBLEM 2

ep_1 = 0.1
ep_2 = 1
ep_3 = 20
y0 = np.sqrt(3)
y_prime0 = 1
t = np.arange(0, 32.5, 0.5)


def solver1(t, z):
    x1, x2 = z
    return np.array([x2, -ep_1 * (x1 ** 2 - 1) * x2 - x1])


def solver2(t, z):
    x1, x2 = z
    return np.array([x2, -ep_2 * (x1 ** 2 - 1) * x2 - x1])


def solver3(t, z):
    x1, x2 = z
    return np.array([x2, -ep_3 * (x1 ** 2 - 1) * x2 - x1])


sol1 = scipy.integrate.solve_ivp(lambda x, y: solver1(x, y), t_span=[0, 32], t_eval=t, y0=np.array([y0, y_prime0]))
sol2 = scipy.integrate.solve_ivp(lambda x, y: solver2(x, y), t_span=[0, 32], t_eval=t, y0=np.array([y0, y_prime0]))
sol3 = scipy.integrate.solve_ivp(lambda x, y: solver3(x, y), t_span=[0, 32], t_eval=t, y0=np.array([y0, y_prime0]))
plt.plot(sol1.t, sol1.y[0], color='#E30B0B', linewidth=2, label='epsilon = 0.1')
plt.plot(sol2.t, sol2.y[0], color='#E3840B', linewidth=2, label='epsilon = 1')
plt.plot(sol3.t, sol3.y[0], color='#E3C60B', linewidth=2, label='epsilon = 20')
plt.legend(loc='upper left', fontsize=8)
plt.show()
A10 = np.transpose(np.array([sol1.y[0], sol2.y[0], sol3.y[0]]))

# Adaptive step size with ep=1
avg_dt = np.empty(7)
tol = np.array([10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10])
for i in range(0, 7):
    sol = scipy.integrate.solve_ivp(solver2, t_span=[0, 32], y0=[2, np.pi**2], atol=tol[i], rtol=tol[i])
    T = sol.t
    Y = sol.y
    avg_dt[i] = np.mean(np.diff(T))
coeffs = np.polyfit(np.log(avg_dt), np.log(tol), 1);
slopeRK45 = coeffs[0]
A11 = slopeRK45

avg_dt = np.empty(7)
tol = np.array([10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10])
for i in range(0, 7):
    sol = scipy.integrate.solve_ivp(solver2, t_span=[0, 32], y0=[2, np.pi**2], atol=tol[i], rtol=tol[i], method='RK23')
    T = sol.t
    Y = sol.y
    avg_dt[i] = np.mean(np.diff(T))
coeffs = np.polyfit(np.log(avg_dt), np.log(tol), 1);
slopeRK23 = coeffs[0]
A12 = slopeRK23

avg_dt = np.empty(7)
tol = np.array([10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10])
for i in range(0, 7):
    sol = scipy.integrate.solve_ivp(solver2, t_span=[0, 32], y0=[2, np.pi**2], atol=tol[i], rtol=tol[i], method='BDF')
    T = sol.t
    Y = sol.y
    avg_dt[i] = np.mean(np.diff(T))
coeffs = np.polyfit(np.log(avg_dt), np.log(tol), 1);
slopeBDF = coeffs[0]
A13 = slopeBDF


# Problem 3
a1 = 0.05
a2 = 0.25
b = 0.1
c = 0.1
I = 0.1
v1_0 = 0.1
v2_0 = 0.1
w1_0 = 0
w2_0 = 0
z0 = np.array([v1_0, v2_0, w1_0, w2_0])


def odes(t, z, d12, d21):
    v1 = z[0]
    v2 = z[1]
    w1 = z[2]
    w2 = z[3]

    v1_prime = -v1**3 + (1 + a1)*v1**2 - a1*v1 - w1 + I + d12*v2
    v2_prime = -v2**3 + (1 + a2)*v2**2 - a2*v2 - w2 + I + d21*v1
    w1_prime = b*v1 - c*w1
    w2_prime = b*v2 - c*w2

    return np.array([v1_prime, v2_prime, w1_prime, w2_prime])


t = np.arange(0, 100.5, 0.5)


sol1 = scipy.integrate.solve_ivp(lambda x, y: odes(x, y, 0, 0), t_span=[0, 100], t_eval=t, y0=z0, method='BDF')
A14 = np.transpose(sol1.y)

sol2 = scipy.integrate.solve_ivp(lambda x, y: odes(x, y, 0, 0.2), t_span=[0, 100], t_eval=t, y0=z0, method='BDF')
A15 = np.transpose(sol2.y)

sol3 = scipy.integrate.solve_ivp(lambda x, y: odes(x, y, -0.1, 0.2), t_span=[0, 100], t_eval=t, y0=z0, method='BDF')
A16 = np.transpose(sol3.y)

sol4 = scipy.integrate.solve_ivp(lambda x, y: odes(x, y, -0.3, 0.2), t_span=[0, 100], t_eval=t, y0=z0, method='BDF')
A17 = np.transpose(sol4.y)

sol5 = scipy.integrate.solve_ivp(lambda x, y: odes(x, y, -0.5, 0.2), t_span=[0, 100], t_eval=t, y0=z0, method='BDF')
A18 = np.transpose(sol5.y)


plt.plot(t, sol1.y[0], color='#3A46D6', linewidth=2, label='v')
plt.plot(t, sol1.y[2], color='#3AD659', linewidth=2, label='W')
plt.title('Fitzhugh Model: Neuron Voltage and Recovery for (d12, d21) = (0, 0)', fontsize=11)
plt.xlabel('Time (t)')
plt.ylabel('v, W')
plt.legend(loc='lower right', fontsize=7)
plt.show()

plt.plot(t, sol2.y[0], color='#3A46D6', linewidth=2, label='v')
plt.plot(t, sol2.y[2], color='#3AD659', linewidth=2, label='W')
plt.title('Fitzhugh Model: Neuron Voltage and Recovery for (d12, d21) = (0, 0.2)', fontsize=11)
plt.xlabel('Time (t)')
plt.ylabel('v, W')
plt.legend(loc='lower right', fontsize=7)
plt.show()


plt.plot(t, sol3.y[0], color='#3A46D6', linewidth=2, label='v')
plt.plot(t, sol3.y[2], color='#3AD659', linewidth=2, label='W ')
plt.title('Fitzhugh Model: Neuron Voltage and Recovery for  (d12, d21) = (-0.1, 0.2)', fontsize=11)
plt.xlabel('Time (t)')
plt.ylabel('v, W')
plt.legend(loc='lower right', fontsize=7)
plt.show()


plt.plot(t, sol4.y[0], color='#3A46D6', linewidth=2, label='v')
plt.plot(t, sol4.y[2], color='#3AD659', linewidth=2, label='W')
plt.title('Fitzhugh Model: Neuron Voltage and Recovery for  (d12, d21) = (-0.3, 0.2)', fontsize=11)
plt.xlabel('Time (t)')
plt.ylabel('v, W')
plt.legend(loc='lower right', fontsize=7)
plt.show()


plt.plot(t, sol5.y[0], color='#3A46D6', linewidth=2, label='v ')
plt.plot(t, sol5.y[2], color='#3AD659', linewidth=2, label='W')
plt.title('Fitzhugh Model: Neuron Voltage and Recovery for (d12, d21) = (-0.5, 0.2)', fontsize=11)
plt.xlabel('Time (t)')
plt.ylabel('v, W')
plt.legend(loc='lower right', fontsize=7)
plt.show()









