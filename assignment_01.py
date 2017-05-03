# author = amritansh
# Artificial Neural Networks
# CS 591


import numpy as np
import math
import matplotlib.pyplot as plt

# declare global variables
Q_trace = []
time_trace =[]
frequency_trace =[]
q_reset = 1

q_thresh = 8

q_spike= 25

N = 500


class RC_model:

    # some class variables

    def __init__(self):
        self.q = 0
        self.sim_timestep = 0.1
        self.tau = 5
        # self.leakage_current_rev_potential = -.06

        self.V_ab = 10
        self.C = 1
        self.input_current = 0

        self.delta_abs = 5

model = RC_model()

def rc_model(q, sim_timestep, tau,  V_ab, C, input_current,  N):
    t_spike = 0

    for X in range(0, N):

        Q_trace.append(q)

        # Using the Leaky integrate formulae
        q += sim_timestep * ((-q/ tau ) +  ((C*V_ab)/tau) + input_current)

        # Q_trace.append(q)

        if q >= q_thresh:
            print(" Spike generated  ")
            # q = q_spike
            Q_trace.append(q)
            time_trace.append((X-t_spike)*sim_timestep)

            q = q_reset
            t_spike = X
        else:
            print ("No Spike")

    return Q_trace, time_trace

# Firing rate approximation method
def firing_rate(delta_abs, mem_time_const):

    f1 = []
    f2 = []

    for X in range (1,100):

        input_current=X*(-.001)

        f1.append(1.0/(delta_abs+mem_time_const*math.log(((input_current)/(input_current-q_thresh)))))
        f2.append(1.0/(0+mem_time_const*math.log(((input_current)/(input_current-q_thresh)))))

    return f1,f2
for x in range(0, 2000):

    a,t = rc_model(model.q, model.sim_timestep, model.tau, model.V_ab, model.C, x*0.001 , N)

    frequency_trace.append( float (1/(sum(t) / (len(t)))))

frequency_trace= np.asarray(frequency_trace)

# # r1, r2 = firing_rate(model.delta_abs, model.mem_time_const)
#

# a,t = rc_model(model.q, model.sim_timestep, model.tau, model.V_ab, model.C, model.input_current , N)
# charge_trace = np.asarray(a)
# Plotting the graphs
y = np.array(0.001 * x for x in range(0,2000))
plt.xlabel('Injection Current Value (I)')
plt.ylabel('Frequency (Hz)')

plt.title(' RC model firing rate with change in value of injection current')
plt.grid(True)
plt.plot(frequency_trace,y)
plt.show()