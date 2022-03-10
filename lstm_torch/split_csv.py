import mmap
import os
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

import data
np.set_printoptions(suppress=True, linewidth=np.inf) 

# https://blog.nelsonliu.me/2016/07/30/progress-bars-for-python-file-reading-with-tqdm/
# This is used for the progress bar so we can keep track of the progress
def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    fp.close()
    return lines

input_path = '/home/jperez/data/sled/350.csv'
output_dir = '/home/jperez/data/sled350/'
columns_to_save = data.COLUMNS_RAW

# Overall dataset statistics
row_counts = []
header_counts = []
plane_timesteps = []

with open(input_path, 'r') as file:
    reader = csv.reader(file)
    p_bar = tqdm(desc='Splitting big CSV file into individual timesteps', postfix='Timestep 0', total=get_num_lines(input_path))

    data_header = []
    data_rows = []

    prev_timestep = 0

    for row in reader:
        # Update progress bar
        p_bar.update(1)
        # Skip empty rows
        if len(row) == 0:
            continue
        # Look for the start of a new file
        elif row[0] == '[Name]':
            # Check if we have data to save
            if len(data_rows) > 0:
                # Save some dataset statistics to analyze later
                header_counts.append(len(data_header))
                row_counts.append(len(data_rows))

                # Create dataframe from the data and save it
                df = pd.DataFrame(data=data_rows, columns=data_header, dtype=np.float64)
                df = df.rename(columns=lambda col: col.strip())
                df['Timestep'] = current_timestep

                # Save CSV if you want
                # df.to_csv(os.path.join(output_dir, f'{prev_timestep}.csv'), index=False, columns=columns_to_save)
                
                # Saving to NP file format is faster
                np.save(os.path.join(output_dir, f'{current_timestep}'), df[columns_to_save].to_numpy().astype(np.float64))

                # Empty data lists for next CSV file
                data_header = []
                data_rows = []
        # Get the data header for this file
        elif row[0] == '[Data]':
            data_header = next(reader)
        # Check what timestep we are on
        # sled350 ['Plane 1 in Case data 0001'] / ['Plane 1 in Case data 0044 1']
        # sled300 ['Plane 1 in Case data.300 0001.1']
        # sled255 ['Plane 1 in Case 2d_data 0019']
        # sled250 ['Plane 1 in Case 2d_data 0001.250']
        elif row[0][:5] == 'Plane':
            plane_split = row[0].split(' ')
            case = plane_split[4]

            if len(plane_split) == 6 or len(plane_split) == 7:
                ts = plane_split[5]
            else:
                print(f'Could not process timestep in row={row}')
                break
            
            # Check if the timesteps are after the period
            if '.' in ts:
                current_timestep = int(ts.split('.')[0])
            else:
                current_timestep = int(ts)

            # Compare against previous timestep to see if we jumped more than 1
            if current_timestep - prev_timestep > 1:
                print(f'Jumped from timestep {prev_timestep} to timestep {current_timestep}')

            # Update variables and progress bar
            prev_timestep = current_timestep
            plane_timesteps.append(current_timestep)
            p_bar.postfix = f'Timestep {current_timestep}'
        # Otherwise just collect data rows
        else:
            data_rows.append(row)

print('Check that all files have the same number of rows', np.unique(row_counts))
print('Check that all files have the same header length', np.unique(header_counts))
print(f'From {plane_timesteps[0]} to {plane_timesteps[-1]}')
print(f'All timesteps: {plane_timesteps}')
print(f'Last DF: {df.head()}')

print('Testing data generator')
gen = data.SledDataGenerator(output_dir, sequence_length=1, inputs=data.COLUMNS_H, outputs=[], scaler=None, start=plane_timesteps[0], end=plane_timesteps[-1])

for idx, input_name in enumerate(data.COLUMNS_H):
    d = gen.x_data[:, :, idx].numpy()
    print(f'For input [{input_name}] | min={np.min(d):.5f}, max={np.max(d):.5f}, mean={np.mean(d):.5f}, median={np.median(d):.5f}, std={np.std(d):.5f}, var={np.var(d):.5f}')

###### sled250-part1 [1-371]
# For input [X] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35897
# For input [Y] | min=-0.00000, max=2.20000, mean=1.10243, median=1.10000, std=0.64424, var=0.41505
# For input [T] | min=1.00000, max=371.00000, mean=186.00018, median=186.00000, std=107.09808, var=11469.99805
# For input [P] | min=-196.57170, max=856.76923, mean=0.65986, median=0.02098, std=8.14695, var=66.37283
# For input [Vu] | min=-342.63785, max=574.69769, mean=99.78954, median=38.92635, std=112.41581, var=12637.31445
# For input [Vv] | min=-285.01266, max=725.59052, mean=62.08943, median=16.83602, std=98.96703, var=9794.47266
# For input [W.VF] | min=0.00000, max=1.00000, mean=0.02287, median=0.00000, std=0.14637, var=0.02142

###### sled250-part2 [342-742]
# For input [X] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35896
# For input [Y] | min=-0.00000, max=2.20000, mean=1.10243, median=1.10000, std=0.64424, var=0.41505
# For input [T] | min=342.00000, max=741.00000, mean=541.50018, median=541.50000, std=115.46955, var=13333.21680
# For input [P] | min=-403.07742, max=384.71158, mean=2.56864, median=-0.01262, std=16.81758, var=282.83115
# For input [Vu] | min=-277.06909, max=633.02643, mean=63.34733, median=14.09456, std=156.54611, var=24506.68750
# For input [Vv] | min=-410.29788, max=845.46094, mean=71.21968, median=25.40969, std=139.65906, var=19504.65039
# For input [W.VF] | min=0.00000, max=1.00000, mean=0.06658, median=0.00000, std=0.24037, var=0.05778

###### sled255 [19-760]
# For input [X] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35896
# For input [Y] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35896
# For input [T] | min=-0.00000, max=2.20000, mean=1.10243, median=1.10000, std=0.64424, var=0.41505
# For input [P] | min=-0.00000, max=0.00002, mean=0.00001, median=0.00001, std=0.00001, var=0.00000
# For input [Vu] | min=19.00000, max=759.00000, mean=389.00018, median=389.00000, std=213.90796, var=45756.61719
# For input [Vv] | min=-42,412,744.00000, max=91,767,672.00000, mean=177466.42188, median=1124.49976, std=1411693.50000, var=1992878587904.00000
# For input [W.VF] | min=-351.97946, max=672.08752, mean=79.29243, median=33.99055, std=142.58383, var=20330.15039

###### sled300 [1-742]
# For input [X] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35896
# For input [Y] | min=-0.00000, max=2.20000, mean=1.10243, median=1.10000, std=0.64424, var=0.41505
# For input [T] | min=1.00000, max=741.00000, mean=371.00024, median=371.00000, std=213.90796, var=45756.61719
# For input [P] | min=-560.46924, max=1223.64648, mean=2.81341, median=0.01786, std=20.42302, var=417.09970
# For input [Vu] | min=-423.32080, max=753.38202, mean=83.41341, median=37.21552, std=165.57777, var=27415.99805
# For input [Vv] | min=-461.07971, max=1042.05566, mean=79.37818, median=26.00456, std=157.45697, var=24792.69727
# For input [W.VF] | min=0.00000, max=1.00000, mean=0.05334, median=0.00000, std=0.21708, var=0.04712

###### sled350 [1-742]
# For input [X] | min=-0.02500, max=3.97500, mean=1.98092, median=2.00000, std=1.16575, var=1.35896
# For input [Y] | min=-0.00000, max=2.20000, mean=1.10243, median=1.10000, std=0.64424, var=0.41505
# For input [T] | min=1.00000, max=741.00000, mean=371.00024, median=371.00000, std=213.90796, var=45756.61719
# For input [P] | min=-801.75287, max=3161.28638, mean=4.27008, median=0.03000, std=29.35718, var=861.84406
# For input [Vu] | min=-504.64990, max=817.10913, mean=88.62949, median=39.18100, std=184.65282, var=34096.66406
# For input [Vv] | min=-591.90942, max=1243.00281, mean=87.22538, median=27.97395, std=195.59991, var=38259.32422
# For input [W.VF] | min=0.00000, max=1.00000, mean=0.05868, median=0.00000, std=0.22692, var=0.05149

