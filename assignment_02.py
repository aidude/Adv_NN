# author = amritansh
# Artificial Neural Networks
# CS 591


import numpy as np
import math
import matplotlib.pyplot as plt

# declare global variables
V_trace = []
time_trace =[]
frequency_trace =[]
External_V_trace=[]
v_reset = -0.065

v_thresh = -0.05

v_spike= 0.02

N = 500


class Leaky_Integrate_model:

    # some class variables
    def __init__(self):

        self.sim_timestep = 0.0001
        self.tau = 0.01
        self.leakage_potential = -0.065

        self.V_init = -0.065
        self.mem_resistance = 1
        self.input_current = 0.001

model = Leaky_Integrate_model()

def LIF_model(v, sim_timestep, tau,  leakage_potential, mem_resistance, input_current,  N):
    t_spike = 0

    for X in range(0, N):

        V_trace.append(v)

        # Using the Leaky integrate formulae
        v += (sim_timestep/tau) * ((-(v-leakage_potential)) + mem_resistance * input_current)

        # Q_trace.append(q)

        if v >= v_thresh:
            print(" Spike generated  ")
            # v = v_spike
            # V_trace.append(v)
            time_trace.append((X-t_spike)*sim_timestep)


            t_spike = X
            v = v_reset
        else:
            print ("No Spike")

    return V_trace, time_trace

# Firing rate approximation method
# def firing_rate(delta_abs, mem_time_const):
#
#     f1 = []
#     f2 = []
#
#     for X in range (1,100):
#
#         input_current=X*(-.001)
#
#         f1.append(1.0/(delta_abs+mem_time_const*math.log(((input_current)/(input_current-q_thresh)))))
#         f2.append(1.0/(0+mem_time_const*math.log(((input_current)/(input_current-q_thresh)))))
#
#     return f1,f2
for x in range(0, 400):
    External_V_trace.append(0.0001*x)
    a, t = LIF_model(model.V_init, model.sim_timestep, model.tau, model.leakage_potential, model.mem_resistance,
                     0.0001 * x, N)

    if (len(t) <= 0):
        frequency_trace.append(0)
    else :
        frequency_trace.append( float (1/(sum(t) / (len(t)))))

frequency_trace= np.asarray(frequency_trace)
print frequency_trace
External_V_trace = np.asarray(External_V_trace)

print frequency_trace.size, External_V_trace.size

# r1, r2 = firing_rate(model.delta_abs, model.mem_time_const)



# Plotting the graphs

# a,t = LIF_model(model.V_init, model.sim_timestep, model.tau, model.leakage_potential, model.mem_resistance, 0.001, N)
# print t
# potential_trace = np.asarray(a)
# print potential_trace
plt.xlabel('Potential due to External Current (volts)')
plt.ylabel('Frequency (Hz)')
# x = np.array(model.sim_timestep * i for i in range(0,507))
# print x.size

plt.title(' Leaky Integrate Fire model')
plt.grid(True)
plt.plot(External_V_trace,frequency_trace)
# plt.axis ([0,0.125, -0.07, 0.25])
plt.show()