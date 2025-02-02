import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#0A. Data Cleaning
df = pd.read_csv("dataIN/AC/Y_DNA_Tari.csv")
labels = df["Country"]
df_pca = df.drop(columns=["Country","Code"]).dropna().astype(float)

#0B Normalizare
mean = df_pca.mean()
std = df_pca.std()
df_pca = (df_pca - mean) / std

#0C PCA (Valori si Vectori propri)
cov = np.cov(df_pca.T)
eigenvalue,eigenvector = np.linalg.eig(cov)

#0D Indici sortati
indices = np.argsort(eigenvalue)[::-1]

#1. Varianta Componentelor

varianta = eigenvalue[indices]/np.sum(eigenvalue)

print("Varianta Componente")
for i in range(len(varianta)):
    print("PC"+str(i+1)+": "+str(varianta[i]))
print()

#2. Plot de Variatie

#Criteriu Varinta explicativa peste 80%
sumf = 0
ind = 0
while sumf<0.8 and ind<len(varianta):
    sumf+=varianta[ind]
    ind+=1

print("Varianta: "+str(sumf))
print("Nr Componente Varianta: "+str(ind))
print()

#Criteriu Kaiser
indK = 0
while eigenvalue[indK]>1 and indK<len(eigenvalue):
    indK+=1

#Cirteriul Catell
diff = []
indC = 0
for i in range(len(eigenvalue)-1):
    diff.append(eigenvalue[i]-eigenvalue[i+1])

calc = 0.1
while 0 < calc and indC < len(diff)-1:
    calc = diff[indC]-diff[indC+1]
    indC+=1

plt.scatter(range(1,len(varianta)+1),varianta)
plt.plot(range(1,len(varianta)+1),varianta)
#Criteriul Variantei
plt.plot([ind,ind],[0,0.5],color="red")
plt.annotate("Varianta",(ind,0.5))
#Criteriul Kaiser
plt.plot([indK,indK],[0,0.5],color="yellow")
plt.annotate("Kaiser",(indK,0.5))
#Criteriul Catell
plt.plot([indC,indC],[0,0.5],color="green")
plt.annotate("Catell",(indC,0.5))
plt.title('Scree Plot')
plt.xlabel('PC')
plt.ylabel('Varianta')
plt.grid()
plt.show()

#3. Corelatii Factoriale
corr_df = pd.DataFrame(eigenvector[:,indices]*np.sqrt(eigenvalue[indices]),columns=["PC"+str(i+1) for i in range(len(varianta))],index=df_pca.columns)

#4. Corelograma
fg,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_df,vmax=1,vmin=-1,cmap="coolwarm",annot=True,ax=ax)
plt.show()

#5. Cercul Corelatilor

# r=1
# a=0 b=0
# x^2 + y^2 = 1 => y^2 = +-sqrt(1-x^2)
# x^2 + y^2 = 0.25 => y^2 = +-sqrt(0.25-x^2)
x_half = []
y_half = []
x_cerc = []
y_cerc = []

for x in range(-1000,1001):
    x_cerc.append(x/1000)
    y_cerc.append(-math.sqrt(1-(x/1000)**2))
    x_half.append(x/2000)
    y_half.append(-math.sqrt(0.25-(x/2000)**2))

for x in range(1000,-1001,-1):
    x_cerc.append(x/1000)
    y_cerc.append(math.sqrt(1-(x/1000)**2))
    x_half.append(x/2000)
    y_half.append(math.sqrt(0.25-(x/2000)**2))

#PC1 PC2
plt.figure(figsize=(10,10))
plt.plot([-1,1],[0,0])
plt.plot([0,0],[-1,1])
plt.plot(x_half,y_half)
plt.plot(x_cerc,y_cerc)

pc1_scatter = corr_df["PC1"].to_numpy()
pc2_scatter = corr_df["PC2"].to_numpy()
plt.scatter(pc1_scatter,pc2_scatter)
for lb,x,y in zip(df_pca.columns,pc1_scatter,pc2_scatter):
    plt.annotate(lb,(x,y))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


#6A. Componente Principale (proiectie)
pc = pd.DataFrame(eigenvector[:,indices],columns=["PC"+str(i+1) for i in range(len(varianta))],index=df_pca.columns)
df_pc = df_pca.dot(pc)
print(df_pc)

#6B. Scoruri
scoruri = df_pc / np.sqrt(eigenvalue[indices])
print(scoruri)

#7A. Plot Componente(proiectie)

#PC1 PC2
plt.scatter(df_pc['PC1'],df_pc['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.plot([-10,10],[0,0],color="blue")
plt.plot([0,0],[-10,10],color="blue")
for lb,x,y in zip(labels,df_pc['PC1'],df_pc['PC2']):
    plt.annotate(lb,(x,y))
plt.title('First 2 Components Projections')
plt.show()


#7A. Plot Componente(proiectie)

#PC1 PC2
plt.scatter(scoruri['PC1'],scoruri['PC2'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.plot([-5,5],[0,0],color="blue")
plt.plot([0,0],[-5,5],color="blue")
for lb,x,y in zip(labels,scoruri['PC1'],scoruri['PC2']):
    plt.annotate(lb,(x,y))
plt.title('First 2 Components Scores')
plt.show()

#8A. Calcul Cos^2 (toate axele)

squared_pc = df_pc**2
squared_sum = squared_pc.sum(axis=1)
df_cos = squared_pc.div(squared_sum,axis=0)
print(df_cos)

#8B. Calcul Cos^2 (primele 2 componente)

squared_pc = df_pc[['PC1','PC2']]**2
squared_sum = squared_pc.sum(axis=1)
df_cos = squared_pc.div(squared_sum,axis=0)
print(df_cos)

#9. Calcul Contributii(toate componentele)

squared_pc = df_pc**2
contributii = squared_pc/(eigenvalue[indices]*squared_pc.shape[0])
print(contributii)

#10. Comunalitati(2 componentele ... daca luam toate componentele aceasta va fi la 100%)

squared_corr = corr_df[['PC1','PC2']]**2
comunatiati_df = pd.DataFrame(squared_corr.sum(axis=1),columns=['PC1+PC2'])
print(comunatiati_df)

#11. Corelograma Comunalitati

sns.heatmap(comunatiati_df,vmin=0,vmax=1,annot=True)
plt.show()