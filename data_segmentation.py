import pandas as pd
import glob


def overlapping_windows(emg_signal, win_size, overlap_val):

    step_size = max(1, round(win_size * (1 - overlap_val)))
    windows = []

    for j in range(0, len(emg_signal) - win_size + 1, step_size):
        w = emg_signal[j:j+win_size]
        windows.append(w)

    return windows


emg_files = glob.glob('preprocessed_recording*.csv')
window_size = 160
overlap = 0.5
trial_gestures = ["Closed Palm", "Two", "Open Palm", "One", "Three", "Four", "Thumbs Up", "Rest"]
segments_0 = []
segments_1 = []
segments_2 = []

for file in emg_files:

    df = pd.read_csv(file)
    labels = df.iloc[:, 0]

    for i in range(3):

        for gesture in trial_gestures:
            gesture_rows = df[df['Label'] == f'{gesture}']

            emg_sig = gesture_rows[f'EMG_FIL_{i}'].values
            emg_sig_windows = overlapping_windows(emg_sig, window_size, overlap)

            df_sig_seg = pd.DataFrame(emg_sig_windows)
            df_sig_seg['Label'] = f'{gesture}'
            df_sig_seg['CH'] = f'EMG_CH_{i}_SEG'

            if i == 0:
                segments_0.append(df_sig_seg)

            elif i == 1:
                segments_1.append(df_sig_seg)

            elif i == 2:
                segments_2.append(df_sig_seg)

concat_df = pd.concat(segments_0 + segments_1 + segments_2, ignore_index=True)
concat_df = concat_df[['CH', *range(window_size), 'Label']]
concat_df.to_csv('sig_segments.csv', index=False)
