__author__ = 'amritansh'
# Let's build Artificial neural simulations with pre-synaptic synapses
# Start with some necessary imports


import numpy as np
import matplotlib.pyplot as plt
import math

# some key value intialization
V_thresh = -.05
V_reset = -.055
V_spike = 0.005
N = 150
n = 5


# class for variable initialization


class leaky_neural_with_synapses:
    def __init__(self):
        self.int_time_step = 0.0001
        self.mem_time_const = 0.025
        self.leakage_current_rev_potential = -.06
        self.p_max = 1
        self.E_sk = [0.0, -0.08, -0.08, -0.08, 0.0]
        self.post_synaptic_potential_time = .002
        self.time_arrival = .001
        self.effective_synaptic_weight = 0.25
        self.V_input_current = .01
        self.V_init = -0.065

neural = leaky_neural_with_synapses()


# function for euler approximation to calculate potential change with input sysnapses
def leaky_model_synapses(v, int_time_step, mem_time_const, leakage_current_rev_potential, V_input_current, effective_synaptic_weight, E_sk,
time_arrival, post_synaptic_potential_time, N):
    v_trace = []
    t_trace = []
    for X in range(1, N+1):
        v_trace.append(v)
        var = 0
    for Y in range(0, n):
        var += effective_synaptic_weight * (v-E_sk[Y])*math.exp(-(0.001-time_arrival)/post_synaptic_potential_time)
        print var
    # print var
    v += (int_time_step / mem_time_const) * ((leakage_current_rev_potential - v) - var + V_input_current)
    if v >= V_thresh:
        print("Action Potential Fired ")
        v = V_spike
        v_trace.append(v)
        t_trace.append(X)
        v = V_reset
    else:
        print ("No Activity")
    return v_trace, t_trace

# frequency calculation
def freq_calc(b):
    temp = 0
    var = 0
    for Z in b:
        var = Z-temp
        temp = Z
    print var
    if var == 0:
        freq = 0
    else:
        freq = float(1000.0/var)
    return freq
frequency = []
for X in range(10,34):
    a, b = leaky_model_synapses(neural.V_init, neural.int_time_step, neural.mem_time_const,neural.leakage_current_rev_potential,
    X*0.001, neural.effective_synaptic_weight, neural.E_sk,neural.time_arrival, neural.post_synaptic_potential_time, N)
potential_trace = np.asarray(a)
time_trace = np.asarray(b)
freq_inter = freq_calc(time_trace)
frequency.append(freq_inter)
print frequency
frequency_trace = np.asarray(frequency)

# print (potential_trace)
# plotting the graphs
plt.xlabel('Time (ms)')
# # plt.yticks([-.075,-.070,-.065,-.060,-.055,-.050,-.045,-.040])
plt.ylabel('Voltage (V)')
plt.title(' Leaky Integrate Fire Neural Model with synapses')
plt.grid ()
plt.plot(potential_trace)
plt.show()