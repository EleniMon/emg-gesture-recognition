import pandas as pd
import glob

rec_files = glob.glob('preprocessed_recording*.csv')
rec_files_list = []

for file in rec_files:
    df = pd.read_csv(file)
    filtered_data = df[df['Label'] == 'Rest'].tail(10)
    rec_files_list.append(filtered_data)

concat_df = pd.concat(rec_files_list, ignore_index=True)
last_three_columns = concat_df.iloc[:, -3:]
thresh = abs(4 * (last_three_columns.values.flatten().mean()))

print(f"SSC and ZC Threshold Value: {thresh}")
