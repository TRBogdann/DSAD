import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import factor_analyzer as fa
from numpy.ma.extras import unique

drop_list = ["Country"]
df = pd.read_csv("./dataIN/AC/MortalityEU.csv")


#0 Data Cleaning + Standardizare
df_norm = (df.replace(":",None).dropna())
labels = df_norm["Country"]
labels.index = [i for i in range(labels.shape[0])]
df_norm = df_norm.drop(columns=drop_list).astype(float)
df_norm = (df_norm-df_norm.mean())/df_norm.std()
print(df_norm)

#1. Test Barlett

# Test Bartlett: H0:Nu exista factoriabilitate, H1:exista factoriabilitate
# Pt un grad de incredere de 95% => comparam p-value cu 0.05 (1-0.95)
# Daca p-value < 0.05 => Respingem H0 si Acceptam H1 => Putem realiza analiza pe factori pentru setul curent de date

chi_val,p_val = fa.calculate_bartlett_sphericity(df_norm)
print("Chi-Value: "+str(chi_val))
print("P-Value: "+str(p_val))

#2. Test KMO

#Test KMO - arata gradul de factoriabilitate
# indicele KMO trebuie sa fie > 0.5

kmo_all,kmo_model = fa.calculate_kmo(df_norm)
print("KMO: "+str(kmo_model))


#0A. Prima analiza pentru determinarea numarului de factori
model_fa = fa.FactorAnalyzer(n_factors=df_norm.shape[1],rotation=None)
model_fa.fit(df_norm)
eigenvalue,eigenvector = model_fa.get_eigenvalues()

#3. Calcul varianta
variance = eigenvalue/np.sum(eigenvalue)

#0B. Testare numar factori semnificativi

#Criteriul Kaiser
indK = 0
while indK < len(eigenvalue) and eigenvalue[indK]>=1:
    indK+=1

#Criteriul Variantei
indV = 0
var = 0
while indV < len(eigenvalue) and var<0.8:
    var+=variance[indV]
    indV+=1

#Criteriul lui Catell
indC = 0
val = 0.1
diff = []
for i in range(len(eigenvalue)-1):
    diff.append(eigenvalue[i] - eigenvalue[i+1])
while indC < len(diff)-1 and val>=0:
    val = diff[indC]-diff[indC+1]
    indC += 1

print("Varianta Factori: ")
print(variance)
#Plot varianta

print("Catell: "+str(indC))
print("Kaiser: "+str(indK))
print("Variatie: "+str(indV))

plt.scatter([i for i in range(len(variance))],variance)
plt.plot([i for i in range(len(variance))],variance)

plt.bar(["F"+str(i+1) for i in range(len(variance))],[0 for x in range(len(variance))])

if indC>0:
    plt.plot([indC-1,indC-1],[0,0.5],color="red")
    plt.annotate("Catell",(indC-1,0.5))

if indK>0:
    plt.plot([indK-1,indK-1],[0,0.5],color="green")
    plt.annotate("Kaiser", (indK-1, 0.5))

if indV>0:
    plt.plot([indV-1,indV-1],[0,0.5],color="yellow")
    plt.annotate("Varianta", (indV-1, 0.5))

plt.show()
#In cazul curent am ales criteriul Catell
model = fa.FactorAnalyzer(n_factors=indC,rotation="varimax")
model.fit(df_norm)

#3 Calcul Varianta
eigenvalue,eigenvector = model.get_eigenvalues()
variance = eigenvalue/np.sum(eigenvalue)

print("Varianta componente: ")
print(variance)

#4.Corelatii Factoriale(Loadings)
corr_df = pd.DataFrame(model.loadings_ ,index = df_norm.columns,
                       columns= ["F"+str(i+1) for i in range(indC)])
print(corr_df)

#5. Corelograma (Loadings)
fig,ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr_df,vmin=-1,vmax=1,cmap="coolwarm",annot=True,ax=ax)
plt.show()

#6. Cercul Corelatilor F1 si F2

x_full = []
y_full = []
x_half = []
y_half = []
for i in range(-1000,1001):
    x_full.append(i/1000)
    y_full.append(math.sqrt(1-(i/1000)**2))

for i in range(1000,-1001,-1):
    x_full.append(i/1000)
    y_full.append(-math.sqrt(1-(i/1000)**2))

for i in range(-500,501):
    x_half.append(i/1000)
    y_half.append(math.sqrt(0.25-(i/1000)**2))

for i in range(500,-501,-1):
    x_half.append(i/1000)
    y_half.append(-math.sqrt(0.25-(i/1000)**2))


plt.figure(figsize=(10,10))
plt.plot(x_full,y_full)
plt.plot(x_half,y_half)
plt.plot([0,0],[-1,1],color="blue")
plt.plot([-1,1],[0,0],color="blue")
plt.xlabel("F1")
plt.ylabel("F2")
plt.title("Corelatii F1 F2")
plt.scatter(corr_df['F1'],corr_df['F2'])
for i in range(len(df_norm.columns)):
    plt.annotate(df_norm.columns[i],(corr_df['F1'][i],corr_df['F2'][i]))
    plt.arrow(0,0,corr_df['F1'][i],corr_df['F2'][i])
plt.show()

#7 Comunalitati si unicitati(varianta specifica) F1,F2,F3

#A
communities = model.get_communalities()
uniqueness = 1 - communities

print("Comunalitati: ")
print(communities)

print("Unicitati: ")
print(uniqueness)
#sau
uniqueness = model.get_uniquenesses()
print(uniqueness)

#B
sq_df = corr_df**2
communities = pd.DataFrame(np.sum(sq_df,axis=1),columns=["F1+F2+F3"])
print(communities)

#8 Corelograma unicitati si comunalitati
fig,ax = plt.subplots(figsize=(10,10))
plt.title("Comunalitati F1+F2+F3")
sns.heatmap(communities,vmin=-1,vmax=1,cmap="coolwarm",annot=True,ax=ax)
plt.show()

fig,ax = plt.subplots(figsize=(10,10))
plt.title("Unicitati/Varianta Specifica F1+F2+F3")
sns.heatmap(1-communities,vmin=-1,vmax=1,cmap="coolwarm",annot=True,ax=ax)
plt.show()

#9 Scoruri
df_efa = pd.DataFrame(model.transform(df_norm),columns=["F1","F2","F3"])
print(df_efa)

#10 Plot F1,F2
print(labels)
plt.scatter(df_efa["F1"],df_efa["F2"])

for i in range(len(labels)):
    plt.annotate(labels[i],(df_efa["F1"][i],df_efa["F2"][i]))
plt.grid()
plt.plot([0,0],[-5,5])
plt.plot([-5,5],[0,0])
plt.xlabel("F1")
plt.ylabel("F2")
plt.show()