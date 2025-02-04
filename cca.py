import math

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cross_decomposition import CCA
from scipy.stats import chi2
import seaborn as sns
import matplotlib.pyplot as plt

#0A. Data Cleaing
df_X = pd.read_csv("./dataIN/AC/ElectricityProduction.csv",index_col=1).drop(columns=["CountryCode"]).astype(float)
df_Y =pd.read_csv("./dataIN/AC/Emissions.csv",index_col=1).drop(columns=["CountryCode"]).astype(float)

df_X = df_X.replace(np.nan,0)
df_Y = df_Y.replace(np.nan,0)

#0B Normalizare

df_X = (df_X - df_X.mean()) / df_X.std()
df_Y = (df_Y - df_Y.mean()) / df_Y.std()

df_X = df_X[df_X.index.isin(df_Y.index)]
df_Y = df_Y[df_Y.index.isin(df_X.index)]


print(df_X.shape)
print(df_Y.shape)
# Determinare numar de componente


p = len(df_X.columns)
q = len(df_Y.columns)
m = min(p,q)

#1. Scoruri (Z si U)
model = CCA(m)
z,u = model.fit_transform(df_X,df_Y)

#2. Corelatii Canonice
z1 = normalize(z,axis=0)
u1 = normalize(u,axis=0)

r = np.diag(np.corrcoef(z1,u1,rowvar=False)[:m,m:])
print(r)

#3 Test Barlett

r2 = r * r
n = df_X.shape[0]

x = 1 - r2
df = [(p - k + 1) * (q - k + 1) for k in range(1, m + 1)]
l = np.flip(np.cumprod(np.flip(x)))
chi2_calc = (-n + 1 + (p + q + 1) / 2) * np.log(l)
p_val = 1 - chi2.cdf(chi2_calc,df)

print(p_val)

#4 Calcul Corelatii

corr_X = np.corrcoef(df_X,z,rowvar=False)[:p,p:]
corr_Y = np.corrcoef(df_Y,u,rowvar=False)[:q,q:]

corr_X = pd.DataFrame(corr_X,index=df_X.columns,columns=["Z"+str(i) for i in range(m)])
corr_Y = pd.DataFrame(corr_Y,index=df_Y.columns,columns=["U"+str(i) for i in range(m)])


#5 Cercul corelatilor (Z1,Z2)

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
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.title("Corelatii Z1 Z2")
plt.scatter(corr_X['Z1'],corr_X['Z2'])
for i in range(len(corr_X.index)):
    plt.annotate(corr_X.index[i],(corr_X['Z1'][i],corr_X['Z2'][i]))
    plt.arrow(0,0,corr_X['Z1'][i],corr_X['Z2'][i])
plt.show()

#6 Corelograma

sns.heatmap(corr_X,vmin=-1,vmax=1,cmap="coolwarm",annot=True)
plt.show()

sns.heatmap(corr_Y,vmin=-1,vmax=1,cmap="coolwarm",annot=True)
plt.show()

#7 Biplot
df_z = pd.DataFrame(z1,columns=["Z"+str(i+1) for i in range(m)],index=df_X.index)
df_u = pd.DataFrame(u1,columns=["U"+str(i+1) for i in range(m)],index=df_Y.index)

plt.scatter(df_z["Z1"],df_z["Z2"],color="red")
plt.scatter(df_u["U1"],df_u["U2"],color="blue")
plt.title("Biplot Z1/Z2-U1/U2")
plt.xlabel("Z1/U1")
plt.ylabel("Z2/U2")

for lb,z1,z2,u1,u2 in zip(df_z.index,df_z["Z1"],df_z["Z2"],df_u["U1"],df_u["U2"]):
    plt.annotate(lb,(z1,z2))
    plt.annotate(lb, (u1, u2))
plt.show()


#8 Varianta  si Redundanta (cod furtuna)
varianta_X = np.sum(corr_X.values[:,:m]*corr_X.values[:,:m],axis=0)
varianta_Y = np.sum(corr_Y.values[:,:m]*corr_Y.values[:,:m],axis=0)
redundanta_X = varianta_X * r2[:m]
redundanta_Y = varianta_Y * r2[:m]

df_varianta_reduntanta = pd.DataFrame({
    "VariantaX" : varianta_X,
    "VariantaY": varianta_Y,
    "RedundantaX": redundanta_X,
    "RedundantaY": redundanta_Y
})
# Tine cont de radacini
df_varianta_reduntanta.index = ["R"+str(i+1) for i in range(m)]

print(df_varianta_reduntanta)