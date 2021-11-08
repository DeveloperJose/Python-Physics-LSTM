import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from xfoil import XFoil
from xfoil.test import naca0012
from xfoil.model import Airfoil

from sklearn.preprocessing import LabelEncoder

np.set_printoptions(suppress=True)

df = pd.read_csv('all_airfoil_data.csv', compression='gzip')
label_encoder = LabelEncoder()
df['Airfoil_Type_Encoded'] = label_encoder.fit_transform(df['Airfoil_Type'])

dat_filename = 'n0012-selig.dat'
with open(dat_filename, 'r') as file:
    # Parse the lengths separately
    dat_header = file.readline()
    lengths = file.readline().replace(' ', '').split('.')
    top_length = int(lengths[0])
    bot_length = int(lengths[1])

    #print(f"Dat Header: ", dat_header)
    #print(f"Top/Bot = {top_length}/{bot_length}")

    # Read all the coordinates together
    arr = np.loadtxt(dat_filename, skiprows=1)
    #top_arr = arr[0:top_length]
    #bot_arr = arr[top_length:]

    #print(f"Arr Shape={arr.shape}")
    #print(f"Shape Top/Bot = {top_arr.shape}/{bot_arr.shape}")

    # Rearrange to the proper XFoil format
    #arr2 = np.vstack((top_arr[::-1], bot_arr))

    # Create the airfoil
    test_foil = Airfoil(x=arr[:, :-1], y=arr[:, -1])

    # Visualize
    # plt.figure()
    #plt.suptitle("NACA0012 (example)")
    #plt.scatter(naca0012.x, naca0012.y)

    plt.figure()
    plt.suptitle(f'{dat_filename}')
    plt.scatter(test_foil.x, test_foil.y)

    # print(naca0012.x)
    # print(test_foil.x)

xf = XFoil()
xf.print = False
xf.airfoil = test_foil
xf.Re = 0.050e6
xf.n_crit = 9
xf.max_iter = 70
xf.M = 0
xf.xtr = (1.0, 1.0)
print("Calling XFoil")
from timeit import default_timer as timer

start_time = timer()
a, cl, cd, cm, cp, conv, xstrip1, xstrip2, xoctr1, xoctr2, yoctr1, yoctr2 = xf.aseq(-10, 10, 0.25)
end_time = timer()

print(f"Took {end_time-start_time:.04f}sec")

print("Preparing DF")
data = pd.DataFrame()
data['Alpha'] = a
data['Cl'] = cl
data['Cd'] = cd
data['CDp'] = cp
data['Cm'] = cm
data['XStrip(1)'] = xstrip1
data['XStrip(2)'] = xstrip2
data['XOCtr(1)'] = xoctr1
data['XOCtr(2)'] = xoctr2
data['YOCtr(1)'] = yoctr1
data['YOCtr(2)'] = yoctr2

print(data)
data.to_csv(f'{dat_filename}-polar.csv', index=False)