import os
import sys

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import sklearn.cluster
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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

lonnp = np.array(lon)

latnp = np.array(lat)
combined = np.transpose((lonnp, latnp))

print(combined)
# sys.exit(1)
#kmeans
results = []
# list with clusters qtd
clustersQtd = list(range(2, 11))
for clusterqt in clustersQtd:
    print(clusterqt)
    cluster = KMeans(n_clusters=clusterqt, random_state=0, n_init="auto")
    cluster.fit(combined)
    #
    plt.scatter(lon, lat, s=5)  # esse!!
    plt.scatter(cluster.cluster_centers_[:, 0], cluster.cluster_centers_[:, 1], c='red')
    plt.show()
    # inserir o valor de métrica do resultado no results junto com a quantidade de centroides clusterqt
    results.append(cluster.inertia_)
#usando a técnica da "elbow method" foi possivel determinar o ponto onde a quantidade de clusters deixa de ser interessante
#referencias https://vitalflux.com/k-means-elbow-point-method-sse-inertia-plot-python/
plt.plot(range(2, 11), results)
plt.xticks(range(2, 11))
plt.xlabel("# centroids")
plt.ylabel("Soma do erro médio Quadrático")
plt.show()
# Encontrando o ponto de inflexão, a quantidade de clusters a partir do qual qual a curva grafico começa a decrementar linearmente
kl = KneeLocator(range(2, 11), results, curve="convex", direction="decreasing")

print("valor ideal de centroides: ", kl.elbow)