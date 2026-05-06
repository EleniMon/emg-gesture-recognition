import glob
import pandas as pd
from scipy.signal import butter, sosfilt

fs = 800.0
nyquist = 0.5 * fs

norm_low_freq = 75.0 / nyquist
norm_high_freq = 225.0 / nyquist
bandpass_order = 4
sos1 = butter(N=bandpass_order, Wn=[norm_low_freq, norm_high_freq],
              btype='bandpass', analog=False, output='sos')
rec_files = glob.glob('recording*.csv')


for file in rec_files:

    df = pd.read_csv(file)
    for i in range(3):
        df[f'EMG_FIL_{i}'] = sosfilt(sos1, df.iloc[:, i+1])

    df.to_csv(file, index=False)

rec_files = glob.glob('recording*.csv')
trial_gestures = ["Closed Palm", "Two", "Open Palm", "One", "Three", "Four", "Thumbs Up", "Rest"]

i = 0
for file in rec_files:

    df = pd.read_csv(file)
    rows_to_drop = []

    for gesture in trial_gestures:
        rows_to_drop.append(df[df['Label'] == f'{gesture}'])

    # discard the first 1600 samples (2 sec) from each gesture
    for j in range(len(rows_to_drop)):
        df = df.drop(rows_to_drop[j].index[:1600])

    df.to_csv(f"preprocessed_recording{i}.csv", index=False)
    i = i+1
