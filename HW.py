import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


num_ADC_bits = 15
number_of_channels = 8
voltage_resolution = 74.12 * 1.60217 * (10**(-7))  # you can change it to your specific resolution


class File_samples:
    def __init__(self, file_path):
        self.path = file_path

    # load the file and convert the values to voltage and return a matrix of the data
    def conversion_to_voltage(self):
        data = np.fromfile(self.path, dtype=np.uint16)
        data = np.reshape(data, (number_of_channels, -1), order='F')
        data = np.multiply(voltage_resolution, (data - np.float_power(2, num_ADC_bits - 1)))
        return data

    def file_to_pandas_df(self):
        tempdata = self.conversion_to_voltage()
        df = pd.DataFrame(data=tempdata, index=['CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8'])
        return df

    # I decided to create a methode to plot the signal from specific channel for convenience
    # Note there is an initial and final samples because there are over 10^6 samples and the graph is very dense
    def plot_file_channel(self, number_of_channel, initial_sample, final_sample):
        y1 = self.conversion_to_voltage()
        y = y1[number_of_channel-1, initial_sample:final_sample]
        plt.plot(y)
        plt.xlabel('Number Of Sample')
        plt.ylabel('Value')
        plt.title('CHANNEL' + '\n' + str(number_of_channel))
        plt.show()
        return y

    # In these methode you need to input the specific channel, the frequencies of the bandpass, sampling frequency and
    # the order of the bandpass.
    def zero_phase_bandpass(self, channel, lowest_frq_pass, highest_frq_pass, freq_semp, order):
        nyq = 0.5 * freq_semp
        low = lowest_frq_pass / nyq
        high = highest_frq_pass / nyq
        b, a = butter(order, [low, high], btype='band')
        temp = self.conversion_to_voltage()[channel-1]
        filtered_data = filtfilt(b, a, temp)
        plt.plot(filtered_data[:4000])
        plt.xlabel('Number Of Sample')
        plt.ylabel('Value')
        plt.title('CHANNEL' + '\n' + str(channel) + '\n' + 'filtered')
        plt.show()
        return filtered_data


# Load function
def load_file(file_path):
    file = File_samples(file_path)
    return file


