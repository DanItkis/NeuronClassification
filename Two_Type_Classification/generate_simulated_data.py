from neuron import h, gui
import matplotlib.pyplot as plt
import numpy as np
import json

# Function to downsample data
def downsample(time_vec, data_vec, target_rate):
    dt = time_vec[1] - time_vec[0]  # time step of the original data
    original_rate = 1000 / dt  # original sampling rate in Hz
    downsample_factor = int(original_rate / target_rate)
    downsampled_time = np.mean(np.reshape(time_vec[:len(time_vec)//downsample_factor*downsample_factor], (-1, downsample_factor)), axis=1)
    downsampled_data = np.mean(np.reshape(data_vec[:len(data_vec)//downsample_factor*downsample_factor], (-1, downsample_factor)), axis=1)
    return downsampled_time, downsampled_data

# Function to normalize data between 0 and 1
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Simulation parameters
N = 10000  # Number of repetitions
sampling_rate = 1000  # Hz, adjustable parameter
prob_type1 = 0.25  # Probability of a neuron being type 1
noise_std = 5  # Standard deviation of the Gaussian noise
output_data = []

# Mean and standard deviations for parameter distributions (type 1 and type 2)
params_type1 = {
    'L_mean': 40, 'L_std': 4,
    'diam_mean': 20, 'diam_std': 2,
    'na3_gbar_mean': 0.038, 'na3_gbar_std': 0.0038/2,
    'kdr_gkdrbar_mean': 0.04, 'kdr_gkdrbar_std': 0.004/2,
    'kap_gkabar_mean': 0.1, 'kap_gkabar_std': 0.01/2,
    'leak': 0.000005
}
params_type2 = {
    'L_mean': 25, 'L_std': 2.5,
    'diam_mean': 25, 'diam_std': 2.5,
    'gNa_mean': 0.17, 'gNa_std': 0.017/2,
    'gK_mean': 0.04, 'gK_std': 0.004/2,
    'leak': 0.000005
}

for i in range(N):
    # Determine neuron type
    print(f"Running simulation {i+1}/{N}")
    random_label = 1 if np.random.rand() < prob_type1 else 2
    neuron_type = 1 if np.random.rand() < prob_type1 else 2
    params = params_type1 if neuron_type == 1 else params_type2

    # Draw parameters from normal distributions for the selected type
    L = abs(np.random.normal(params['L_mean'], params['L_std']))
    diam = abs(np.random.normal(params['diam_mean'], params['diam_std']))

    if neuron_type == 1:
        na3_gbar = abs(np.random.normal(params['na3_gbar_mean'], params['na3_gbar_std']))
        kdr_gkdrbar = abs(np.random.normal(params['kdr_gkdrbar_mean'], params['kdr_gkdrbar_std']))
        kap_gkabar = abs(np.random.normal(params['kap_gkabar_mean'], params['kap_gkabar_std']))
        leak_g = abs(params['leak'])
    else:
        gNa = abs(np.random.normal(params['gNa_mean'], params['gNa_std']))
        gK = abs(np.random.normal(params['gK_mean'], params['gK_std']))
        leak_g = abs(params['leak'])


    # Create a single compartment
    soma = h.Section(name='soma')
    soma.L = L  # length in microns, drawn from distribution
    soma.diam = diam  # diameter in microns, drawn from distribution

    if neuron_type == 1:
        soma.insert('na3')
        soma.insert('kdr')
        soma.insert('kap')
        soma.insert('pas')
        for seg in soma:
            seg.na3.gbar = na3_gbar
            seg.kdr.gkdrbar = kdr_gkdrbar
            seg.kap.gkabar = kap_gkabar
            seg.pas.g = leak_g
    else:
        soma.insert('hh')
        soma.insert('pas')
        for seg in soma:
            seg.hh.gnabar = gNa
            seg.hh.gkbar = gK
            seg.pas.g = leak_g


    # Add current plamp pulses
    stim1 = h.IClamp(soma(0.5))  # place at the middle of the soma
    stim1.delay = 500  # ms, time to start the stimulus
    stim1.dur = 500  # ms, duration of the stimulus
    stim1.amp = 0.01  # nA, amplitude of the stimulus

    stim2 = h.IClamp(soma(0.5))  # place at the middle of the soma
    stim2.delay = 1500  # ms, time to start the stimulus
    stim2.dur = 500  # ms, duration of the stimulus
    stim2.amp = 0.02  # nA, amplitude of the stimulus

    stim3 = h.IClamp(soma(0.5))  # place at the middle of the soma
    stim3.delay = 2500  # ms, time to start the stimulus
    stim3.dur = 500  # ms, duration of the stimulus
    stim3.amp = 0.03  # nA, amplitude of the stimulus

    stim4 = h.IClamp(soma(0.5))  # place at the middle of the soma
    stim4.delay = 3500  # ms, time to start the stimulus
    stim4.dur = 500  # ms, duration of the stimulus
    stim4.amp = 0.04  # nA, amplitude of the stimulus

    stim5 = h.IClamp(soma(0.5))  # place at the middle of the soma
    stim5.delay = 4500  # ms, time to start the stimulus
    stim5.dur = 500  # ms, duration of the stimulus
    stim5.amp = 0.05  # nA, amplitude of the stimulus

    # Record variables
    time = h.Vector().record(h._ref_t)  # record time
    v_soma = h.Vector().record(soma(0.5)._ref_v)  # record voltage at soma
    i_stim1 = h.Vector().record(stim1._ref_i)
    i_stim2 = h.Vector().record(stim2._ref_i)
    i_stim3 = h.Vector().record(stim3._ref_i)
    i_stim4 = h.Vector().record(stim4._ref_i)
    i_stim5 = h.Vector().record(stim5._ref_i)
    
    # Set up the simulation
    h.tstop = 5500  # simulation duration in ms

    # Run the simulation
    h.finitialize(-75)  # initialize the membrane potential
    h.run()

    # Extract currents
    i_data1   = np.array(i_stim1)
    i_data2   = np.array(i_stim2)
    i_data3   = np.array(i_stim3)
    i_data4   = np.array(i_stim4)
    i_data5   = np.array(i_stim5)
    t_data    = np.array(time)
    net_injected_current = i_data1 + i_data2 + i_data3 + i_data4 + i_data5

    # Downsample the data
    time_array = np.array(time)
    v_soma_array = np.array(v_soma)
    downsampled_time, downsampled_v_soma = downsample(time_array, v_soma_array, sampling_rate)
    downsampled_time, downsampled_i_soma = downsample(time_array, net_injected_current, sampling_rate)

    # Add Gaussian noise
    noisy_v_soma = downsampled_v_soma + np.random.normal(0, noise_std, downsampled_v_soma.shape)

    # Normalize voltage
    noisy_v_soma_normalized = normalize(noisy_v_soma)

    # Save run data to the library
    run_data = {
        'type': neuron_type,
        'random_label': random_label,
        'L': soma.L,
        'diam': soma.diam,
        'na3_gbar': na3_gbar if neuron_type == 1 else 0,
        'kdr_gkdrbar': kdr_gkdrbar if neuron_type == 1 else 0,
        'kap_gkabar': kap_gkabar if neuron_type == 1 else 0,
        'gNa': gNa if neuron_type == 2 else 0,
        'gK': gK if neuron_type == 2 else 0,
        'leak': leak_g,
        'noise_std': noise_std,
        'downsampled_voltage': downsampled_v_soma.tolist(),
        'downsampled_voltage_noisy': noisy_v_soma.tolist(),
        'downsampled_voltage_noisy_normalized': noisy_v_soma_normalized.tolist(),
        'downsampled_current': downsampled_i_soma.tolist(),
        'downsampled_time': downsampled_time.tolist()
    }
    output_data.append(run_data)

# Save the library to a file
with open('neuron_simulation_data.json', 'w') as f:
    json.dump(output_data, f, indent=4)
