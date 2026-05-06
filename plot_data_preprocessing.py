import pandas as pd
from scipy.signal import butter, sosfreqz
import matplotlib.pyplot as plt

fs = 800.0
nyquist = 0.5 * fs

norm_low_freq = 75.0 / nyquist
norm_high_freq = 225.0 / nyquist
bandpass_order = 4
sos1 = butter(N=bandpass_order, Wn=[norm_low_freq, norm_high_freq],
              btype='bandpass', analog=False, output='sos')

w1, h1 = sosfreqz(sos1, worN=2000, fs=fs)
plt.plot(w1, abs(h1), 'b')
plt.title('Frequency Response of [75.0, 225.0] Hz Band-Pass Filter of Order 4')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude of Transfer Function')
plt.grid()
plt.show()

df = pd.read_csv('preprocessed_recording12.csv')

for i in range(3):  # Number of channels = 3

    plt.subplot(2, 1, 1)
    plt.plot(df.iloc[:, i+1], color='red', label=f'Channel {i} Raw Data')
    plt.title(f'Channel {i} Raw Data')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(df.iloc[:, i+4], color='red', label=f'Channel {i} Data after Band-Pass Filtering')
    plt.title(f'Channel {i} Data after Band-Pass Filtering')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude (mV)')
    plt.grid()

    plt.tight_layout()
    plt.show()
