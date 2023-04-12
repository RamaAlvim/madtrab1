import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# open csv


def read_csv():
    data = pd.read_csv("./municipios_mg.csv", delimiter=',')
    return(data)


data = read_csv()
datanew = data.iloc[:, [2, 3, 4]]
datanew2 = {'X,Y': datanew.iloc[:, [1, 2]]}

# df = pd.DataFrame(datanew2)
# kmeans = KMeans(n_clusters=10).fit(df)
print(datanew2)

