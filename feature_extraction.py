import pandas as pd
import numpy as np
import math


def waveform_length(row, win_size):
    return row.iloc[1:win_size+1].diff().abs().sum()


def root_mean_square(row, win_size):
    return np.sqrt((row.iloc[1:win_size+1]*row.iloc[1:win_size+1]).mean())


def variance(row, win_size):
    num = 1/(win_size-1)
    mean = row[1:win_size+1].mean()
    return ((row.iloc[1:win_size+1]-mean)*(row.iloc[1:win_size+1]-mean)).sum()*num


def mean_absolute_value(row, win_size):
    return (row.iloc[1:win_size+1]).abs().mean()


def difference_absolute_standard_deviation_value(row, win_size):
    num = 1 / (win_size - 1)
    return np.sqrt(num*((row.iloc[1:win_size + 1].diff()*row.iloc[1:win_size + 1].diff()).sum()))


def slope_sign_changes(row, win_size):
    thresh = 0.1686
    window = row.iloc[1:win_size + 1].values

    diff1 = np.diff(window[:-1])
    diff2 = np.diff(window[1:])
    peaks_or_troughs = diff1 * diff2 < 0
    abs_diff1 = np.abs(diff1)
    abs_diff2 = np.abs(diff2)
    threshold_check = (abs_diff1 >= thresh) | (abs_diff2 >= thresh)

    return (peaks_or_troughs & threshold_check).sum()


def zero_crossing(row, win_size):
    thresh = 0.1686
    window = row.iloc[1:win_size + 1].values

    crossings1 = (window[:-1] > thresh) & (window[1:] < thresh)
    crossings2 = (window[:-1] < thresh) & (window[1:] > thresh)
    crossings = crossings1 | crossings2
    return crossings.sum()


def frequency_components(row, win_size, fsp):

    n = win_size
    next_pow2 = 2 ** math.ceil(math.log2(n))
    fft_vals = np.fft.fft(row.iloc[1:win_size+1], n=next_pow2)
    freq = np.fft.fftfreq(next_pow2, 1 / fsp)[:next_pow2 // 2]  # Positive frequencies only
    fft_vals = fft_vals / n
    spec = fft_vals[:next_pow2 // 2]
    power_spectrum = np.abs(spec) ** 2  # Magnitude squared

    mnf = np.sum(freq * power_spectrum) / np.sum(power_spectrum)

    cumulative_power = np.cumsum(power_spectrum)
    ttp = cumulative_power[-1]
    mdf_index = np.where(cumulative_power >= ttp / 2)[0][0]
    mdf = freq[mdf_index]

    return mnf, mdf, ttp


def create_new_entry(index, row_sig_values, win_size, fsp):

    new_ent = dict()
    new_ent[f'CH{index}_WL'] = waveform_length(row_sig_values, win_size)
    new_ent[f'CH{index}_RMS'] = root_mean_square(row_sig_values, win_size)
    new_ent[f'CH{index}_VAR'] = variance(row_sig_values, win_size)
    new_ent[f'CH{index}_MAV'] = mean_absolute_value(row_sig_values, win_size)
    new_ent[f'CH{index}_DASDV'] = difference_absolute_standard_deviation_value(row_sig_values, win_size)
    new_ent[f'CH{index}_SSC'] = slope_sign_changes(row_sig_values, win_size)
    new_ent[f'CH{index}_ZC'] = zero_crossing(row_sig_values, win_size)
    mnf, mdf, ttp = frequency_components(row_sig_values, win_size, fsp)
    new_ent[f'CH{index}_MNF'] = mnf
    new_ent[f'CH{index}_MDF'] = mdf
    new_ent[f'CH{index}_TTP'] = ttp

    return new_ent


df_sig = pd.read_csv('sig_segments.csv')
labels = df_sig.iloc[:, -1]
start_row_index = 0
end_row_index = int(df_sig.shape[0]/3)
window_size = 160
fs = 800.0
emg_features = []

for i in range(start_row_index, end_row_index):

    ch_row_sig_values = [df_sig.iloc[i], df_sig.iloc[i + end_row_index], df_sig.iloc[i + (2 * end_row_index)]]
    label = labels.iloc[i]

    new_entry = {}

    for j in range(3):
        current_entry = create_new_entry(j, ch_row_sig_values[j], window_size, fs)
        new_entry.update(current_entry)

    new_entry['Label'] = label
    emg_features.append(new_entry)

df = pd.DataFrame(emg_features)

label_mapping = {
    "Rest": 0,
    "Closed Palm": 1,
    "Two": 2,
    "Open Palm": 3,
    "One": 4,
    "Three": 5,
    "Four": 6,
    "Thumbs Up": 7
}

df['Label'] = df['Label'].map(label_mapping)
df.to_csv('feature_set.csv', index=False)
