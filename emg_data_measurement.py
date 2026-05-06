import re
import serial
import time
import pandas as pd
import random


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


pattern = re.compile(r'^-?\d+\.\d+,-?\d+\.\d+,-?\d+\.\d+$')
non_rest_gestures = ["Closed Palm", "Two", "Open Palm", "One", "Three", "Four", "Thumbs Up"]
rest_gesture = "Rest"
data = []
labels = []

sdata = serial.Serial('COM6', 230400, timeout=1.0)
time.sleep(2)
sdata.reset_input_buffer()
print("Arduino Connection Established, Trial Starting \n\n\n\n\n")

try:
    trial_gestures = random.sample(non_rest_gestures, len(non_rest_gestures)) + [rest_gesture]
    for gesture in trial_gestures:
        print(f"Perform: .....{gesture}..... \n\n\n\n\n")

        # Record data for this gesture or rest period
        gesture_data = []
        samples = 1
        while samples <= 4800:  # Record for 6 seconds (6 sec * 800 samples/sec = 4800 samples)
            if sdata.in_waiting > 0:
                emg_data = read_arduino_data(sdata)
                if emg_data is not None:
                    gesture_data.append(emg_data)
                    samples = samples + 1

        # Label the collected data
        for emg_values in gesture_data:
            data.append(emg_values)
            labels.append(gesture)

        # Pause before the next gesture
        if gesture != rest_gesture:
            print("Rest (6 seconds) \n\n\n\n\n")
            time.sleep(6)

    df = pd.DataFrame(data, columns=["EMG_CH_0", "EMG_CH_1", "EMG_CH_2"])
    df["Label"] = labels
    df = df[["Label", "EMG_CH_0", "EMG_CH_1", "EMG_CH_2"]]  # Reorder the columns
    df.to_csv('recording74.csv', index=False)
    print("Trial Completed")

except KeyboardInterrupt:
    print("Serial Communication Closed, Trial Interrupted")
    sdata.close()
