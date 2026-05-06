import serial
import time
import re
import numpy as np
from scipy.signal import butter, sosfilt
import math
import tensorflow as tf
import joblib
import statistics
import socket


def validate_serial_data(ser_data):
    return 0 if pattern.match(ser_data) is None else 1


def read_arduino_data(ser_data):

    line = ser_data.readline().decode('utf-8').strip()
    if validate_serial_data(line) == 1:
        line = line.split(',')
        emg_val_data = [float(val_data) for val_data in line]
        return emg_val_data
    else:
        return None


def waveform_length(x_last_overlap_val, segment):
    return x_last_overlap_val + np.sum(np.abs(np.diff(segment)))


def root_mean_square(x_last_overlap_val, segment, win_size):
    return np.sqrt((x_last_overlap_val + np.sum(segment*segment))/win_size)


def variance(mean_val, segment, win_size):
    return np.sum((segment - mean_val)*(segment - mean_val))*(1 / (win_size - 1))


def mean_absolute_value(x_last_overlap_val, segment, win_size):
    return (x_last_overlap_val + np.sum(np.abs(segment)))/win_size


def difference_absolute_standard_deviation_value(x_last_overlap_val, segment, win_size):
    return np.sqrt((x_last_overlap_val + np.sum(np.diff(segment) * np.diff(segment))) * (1/(win_size - 1)))


def slope_sign_change(segment):
    thres = 0.1686

    diff1 = np.diff(segment[:-1])
    diff2 = np.diff(segment[1:])
    peaks_or_troughs = diff1 * diff2 < 0
    abs_diff1 = np.abs(diff1)
    abs_diff2 = np.abs(diff2)
    threshold_check = (abs_diff1 >= thres) | (abs_diff2 >= thres)

    return (peaks_or_troughs & threshold_check).sum()


def zero_crossing(segment):
    thres = 0.1686

    crossings1 = (segment[:-1] > thres) & (segment[1:] < thres)
    crossings2 = (segment[:-1] < thres) & (segment[1:] > thres)
    crossings = crossings1 | crossings2
    return crossings.sum()


def frequency_components(segment, win_size, fsp):

    n = win_size
    nextpow2 = 2 ** math.ceil(math.log2(n))
    fft_vals = np.fft.fft(segment, n=nextpow2)
    freqs = np.fft.fftfreq(nextpow2, 1 / fsp)[:nextpow2 // 2]  # Positive frequencies only
    fft_vals = fft_vals / n
    spec = fft_vals[:nextpow2 // 2]
    power_spectrum = np.abs(spec) ** 2  # Magnitude squared

    mnf_val = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

    cumulative_power = np.cumsum(power_spectrum)
    ttp_val = cumulative_power[-1]
    mdf_index = np.where(cumulative_power >= ttp_val / 2)[0][0]
    mdf_val = freqs[mdf_index]

    return mnf_val, mdf_val, ttp_val


def compute_previous_features(transposed_emg_values, win_size, win_step):

    x_last_overlap_val = []

    for lt in transposed_emg_values:

        wl = np.sum(np.abs(np.diff(lt[-win_size + win_step:])))
        ssi = np.sum(lt[-win_size + win_step:] * lt[-win_size + win_step:])
        summ = np.sum(lt[-win_size + win_step:])
        abs_sum = np.sum(np.abs(lt[-win_size + win_step:]))
        wl_squared = np.sum(np.diff(lt[-win_size + win_step:])*np.diff(lt[-win_size + win_step:]))
        ssc = slope_sign_change(lt[-win_size + win_step:])
        zc = zero_crossing(lt[-win_size + win_step:])
        x_last_overlap_val += [wl, ssi, summ, abs_sum, wl_squared, ssc, zc]

    return x_last_overlap_val


pattern = re.compile(r'^-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+$')

fs = 800.0
nyquist = 0.5 * fs
norm_lowfreq = 75.0 / nyquist
norm_highfreq = 225.0 / nyquist
bandpass_order = 4
sos1 = np.array(butter(N=bandpass_order, Wn=[norm_lowfreq, norm_highfreq],
                       btype='bandpass', analog=False, output='sos'))
zi1 = np.zeros((sos1.shape[0], 3, 2))

window_size = 320
overlap = 0.70
window_step = max(1, round(window_size * (1 - overlap)))
emg_values_tot = []
transposed_emg_values_tot = []
filtered_transposed = []
x_last_overlap = []
cnt = 0

loaded_scaler = joblib.load('scaler.pkl')
interpreter = tf.lite.Interpreter(model_path="final_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
pred_list = []

rasp_ip = '   .   .   .   .'
rasp_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((rasp_ip, rasp_port))
print("Socket Connection Established")

sdata = serial.Serial('COM6', 230400, timeout=1.0)
time.sleep(2)
sdata.reset_input_buffer()
print("Arduino Connection Established")

try:
    while True:
        if sdata.in_waiting > 0:
            emg_data = read_arduino_data(sdata)
            if emg_data is not None:
                emg_values_tot.append(emg_data)

                if len(emg_values_tot) == window_size:

                    x = []
                    transposed_emg_values_tot = np.transpose(emg_values_tot)

                    if len(x_last_overlap) == 0:

                        filtered1, zi1 = sosfilt(sos1, transposed_emg_values_tot, zi=zi1)
                        filtered_transposed = filtered1
                        x_last_overlap = compute_previous_features(filtered_transposed, window_size, window_step)

                        i = 0
                        for lst in filtered_transposed:

                            x.append(waveform_length(x_last_overlap[i], lst[:window_step+1]))  # WL
                            x.append(root_mean_square(x_last_overlap[i+1], lst[:window_step], window_size))  # RMS
                            mean = (x_last_overlap[i+2] + np.sum(lst[:window_step]))/window_size
                            x.append(variance(mean, lst, window_size))  # VAR
                            x.append(mean_absolute_value(x_last_overlap[i+3], lst[:window_step], window_size))  # MAV
                            x.append(difference_absolute_standard_deviation_value(
                                x_last_overlap[i + 4], lst[:window_step+1], window_size))  # DASDV
                            x.append(x_last_overlap[i+5] + slope_sign_change(lst[:window_step + 2]))  # SSC
                            x.append(x_last_overlap[i+6] + zero_crossing(lst[:window_step+1]))  # ZC
                            mnf, mdf, ttp = frequency_components(lst, window_size, fs)
                            x.append(mnf)  # MNF
                            x.append(mdf)  # MDF
                            x.append(ttp)  # TTP
                            i = i + 7

                    else:

                        last_window_values = [lst[-window_size + window_step:].tolist() for lst in filtered_transposed]
                        recent_window_values = [lst[-window_step:] for lst in transposed_emg_values_tot]

                        filtered1, zi1 = sosfilt(sos1, recent_window_values, zi=zi1)
                        filtered_values = filtered1

                        for i in range(len(filtered_values)):
                            last_window_values[i].extend(filtered_values[i])

                        filtered_transposed = np.array(last_window_values)

                        i = 0
                        for lst in filtered_transposed:

                            x.append(waveform_length(x_last_overlap[i], lst[-window_step-1:]))  # WL
                            x.append(root_mean_square(x_last_overlap[i + 1], lst[-window_step:], window_size))  # RMS
                            mean = (x_last_overlap[i+2] + np.sum(lst[-window_step:]))/window_size
                            x.append(variance(mean, lst, window_size))  # VAR
                            x.append(mean_absolute_value(x_last_overlap[i + 3],
                                                         lst[-window_step:], window_size))  # MAV
                            x.append(difference_absolute_standard_deviation_value(
                                x_last_overlap[i + 4], lst[-window_step-1:], window_size))  # DASDV
                            x.append(x_last_overlap[i + 5] + slope_sign_change(lst[-window_step-2:]))  # SSC
                            x.append(x_last_overlap[i + 6] + zero_crossing(lst[-window_step-1:]))  # ZC
                            mnf, mdf, ttp = frequency_components(lst, window_size, fs)
                            x.append(mnf)  # MNF
                            x.append(mdf)  # MDF
                            x.append(ttp)  # TTP
                            i = i + 7

                        x_last_overlap = compute_previous_features(filtered_transposed, window_size, window_step)

                    emg_values_tot = emg_values_tot[-window_size + window_step:]

                    if cnt < 5:
                        cnt = cnt + 1

                    elif cnt >= 5:

                        x_scaled = loaded_scaler.transform(np.array(x).reshape(1, -1))
                        ipt = x_scaled.astype(np.float32)

                        interpreter.set_tensor(input_details[0]['index'], ipt)
                        interpreter.invoke()
                        output = interpreter.get_tensor(output_details[0]['index'])
                        prediction = np.argmax(output, axis=1)
                        pred_list.append(prediction.item())

                        if (len(pred_list)) == 15:

                            predicted_class = statistics.mode(pred_list)
                            client_socket.send(str(predicted_class).encode('utf-8'))
                            pred_list = []

except (KeyboardInterrupt, ConnectionAbortedError):
    print("Serial and Websocket Communication Closed")
    sdata.close()
    client_socket.close()
