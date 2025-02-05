import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,silhouette_samples
import pandas as pd

#0A. Data Cleaning
df = pd.read_csv("dataIN/AC/Y_DNA_Tari.csv")

df = df.dropna()
labels = df["Country"]
df = df.drop(columns=["Country","Code"]).astype(float)

#0B Standardizare(daca se cere)

mean = df.mean()
std = df.std()
df = (df - mean) / std

#1. Calcul Matrice
linkage_data = linkage(df.to_numpy(), method='ward', metric='euclidean')
matrice = pd.DataFrame(linkage_data,columns=["Cluster1","Cluster2","Distanta","NumarInstante"])
print(matrice)

#2. Calcul partitiei optime
# Calculam diferentele dintre distantele dintre clusteri
# Alegem distanta jonctiunii cu diferenta cea mai mare ca prag
# Numar Optim de Clustere: Nr Jonctiuni - Indice Jonctiune Aleasa

#A Calcul
diff = []
for i in range(len(linkage_data[:,2])-1):
    diff.append(linkage_data[:,2][i+1]-linkage_data[:,2][i])

ind = np.argmax(diff)
th = (linkage_data[:,2][ind+1]+linkage_data[:,2][ind])/2
k = len(linkage_data) - ind
print("Prag: "+str(th))
print("Numar Clustere: "+str(k))

#B Grafic
plt.scatter([x for x in range(len(linkage_data))],linkage_data[:,2])
plt.plot([x for x in range(len(linkage_data))],linkage_data[:,2])
plt.plot([0,len(linkage_data)-1],[linkage_data[:,2][ind],linkage_data[:,2][ind]])
plt.show()

#3 Partitie oarecare sau partitie optima
# In cazul de fata k e partitia optima,
# dar putem alege si alt k, k<=df.shape[0]

hclust = AgglomerativeClustering(n_clusters=k,metric="euclidean",linkage="ward")
hclust.fit(df)

#4 Scoruri Silhouette

print(hclust.labels_)
#Over All
silhouette_avg = silhouette_score(df,hclust.labels_)
print(silhouette_avg)

#Nivel de intanta
silhouette = silhouette_samples(df,hclust.labels_)
print(silhouette)

#Nivel de Partitie
silhouette_df = pd.DataFrame({"sil":silhouette,"cluster":hclust.labels_})
silhouette_mean = silhouette_df.groupby(by="cluster").mean()
print(silhouette_mean)

#5 Dendrograma

#Dendrograma cu prag(elimina a doua linie daca nu vrei prag)
dendrogram(linkage_data,labels=labels.to_list())
plt.axhline(y=th, color='r', linestyle='--', label=f'Threshold = {th:.2f}')
plt.show()

#6 Plot Silhouette - Mi e prea lene sa l fac cum trebuie
silhouette_df = silhouette_df.sort_values(by=['cluster','sil'],ascending=[True,False])
print(silhouette_df)


plt.barh([str(x) for x in range(silhouette_df.shape[0])],silhouette_df['sil'])
plt.show()

