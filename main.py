import os
import pdb
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt #para plotar os gráficos
import csv

# open csv


lon = [] # x
lat = [] # y
with open('./municipios_mg.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    data = pd.read_csv("./municipios_mg.csv", delimiter=',')
    #skip index
    next(plots)
    for row in plots:
        lat.append(float(row[4]))
        lon.append(float(row[3]))

# #get min/max values to plot
# minlon= float(min(lon))
# maxlon= float(max(lon))
#
# minlat= float(min(lat))
# maxlat= float(max(lat))

# plt.scatter(lon, lat)
# plt.scatter(lon,lat, s=5) # esse!!

# get columns with, long and lat
# datanew = data.iloc[:, [ 3, 4]]
# sys.exit(1)
# import pdb; pdb.set_trace();

#concatenate arrays
lonnp = np.array(lon)

latnp = np.array(lat)
combined = np.transpose((lonnp, latnp))

print(combined)
# sys.exit(1)
#kmeans

kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(combined)

centroidk2 = kmeans.cluster_centers_


plt.scatter(centroidk2[0][0],centroidk2[0][1], color='red')
plt.scatter(centroidk2[1][0],centroidk2[1][1], color='red')
plt.scatter(centroidk2[2][0],centroidk2[2][1], color='red')
plt.scatter(centroidk2[3][0],centroidk2[3][1], color='red')

plt.show()
# import pdb; pdb.set_trace();