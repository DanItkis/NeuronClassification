% Load JSON file
jsonFile = 'C:\Users\danie\Documents\GitHub\NeuronClassification\Two_Type_Classification\neuron_simulation_data.json';
fid = fopen(jsonFile, 'r');
rawData = fread(fid, inf);
fclose(fid);

% Decode JSON data
jsonData = jsondecode(char(rawData'));

% Display data for each run
figure;
%hold on;
for i = 1:length(jsonData)
    % Extract neuron type
    neuronType = jsonData(i).type;
    
    % Extract downsampled voltage and time
    downsampled_time = jsonData(i).downsampled_time;
    downsampled_voltage = jsonData(i).downsampled_voltage;
    downsampled_voltage_noisy = jsonData(i).downsampled_voltage_noisy;
    downsampled_voltage_noisy_normalized = jsonData(i).downsampled_voltage_noisy_normalized;

    plot_trace = downsampled_voltage;
    
    % Plot the downsampled voltage trace based on type
    if neuronType == 1
        plot(downsampled_time, plot_trace, '-o', 'Color', 'blue');
    else
        plot(downsampled_time, plot_trace, '-o', 'Color', 'red');
    end
    pause(0.5);
end

% Add labels and grid
title('Sample Voltage Traces by Neuron Type');
xlabel('Time (ms)');
ylabel('Normalized Voltage Traces');
grid on;
hold off;
