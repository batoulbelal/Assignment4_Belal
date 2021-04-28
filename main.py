import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv('C:/Users/HP/PycharmProjects/Assignment4/input.csv',
                 delimiter=";",
                 names=["x", "y"])

df = df.apply(lambda x: x.str.replace(',', '.'))

clusters_num = df.iloc[0,0]
clusters_num = int(clusters_num)
# print(clusters_num)

# print(df[2:])  # x and y values only

x_data = df["x"].tolist() # exporting x values to array
x_data = [float(x) for x in x_data[2:]] # converting str to float

y_data = df["y"].tolist() # exporting y values to array
y_data = [float(x) for x in y_data[2:]] # converting str to float

# print(x_data)
# print(y_data)

plt.scatter(x_data, y_data, s=20, c='r')
plt.title('Input Data Scatter Plot')
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.show()

plt.title("Output Dendrogram")
plt.xlabel('Clusters')
plt.ylabel('Euclidean Distance')
dendrogram = shc.dendrogram(shc.linkage(df[2:], method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=clusters_num, affinity='euclidean', linkage='ward')
print(cluster.fit_predict(df[2:])) # number of cluster allocated for each point
plt.scatter(x_data, y_data, c=cluster.labels_, cmap='rainbow')
plt.title("Input Data Scatter Plot - Clustered")
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.show()

