import numpy as np
import pandas as pd
from seaborn import kdeplot
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score
from  sklearn.model_selection import train_test_split
from sklearn.naive_bayes import  GaussianNB
import matplotlib.pyplot as plt
import scipy.stats as sts
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("./dataIN/AC/glass_train.csv")

print(df)

labels = df['glass']
le = LabelEncoder()
encoded = le.fit_transform(labels)

df = df.drop(columns=["glass","Id"])

#0A. Data Cleaning
for col in df.columns:
    df.fillna({col:df[col].mean()})

#0B Standardizare(daca se cere)
values=set(labels.to_numpy())
df_norm = (df-df.mean())/df.std()

#0C LDA
lda = LinearDiscriminantAnalysis(n_components=len(values)-1)
transformed = lda.fit_transform(df_norm,labels)
df_lda = pd.DataFrame(transformed,columns=['L'+str(i+1) for i in range(len(values)-1)])


#1. Calcul Putere de discriminare

# n - numar randuri
# m - numar coloane
# q - numar clase
# g - matrice de greutati    lda.means_ - df.mean, df normalizat => g = lda.means_
# dg - matricea de frecventa
# t - covariatie totala     t = w + b
# w - covariatie intra-grupa
# b - covariatie inter-grupa

# t = (X.T*X)/m (matrice de covariatie)
# b =  g.T*dg*g - diagonalizare
# w = t - b

n = df_norm.shape[0]
m = df_norm.shape[1]
q = len(values)
g = lda.means_
dg = np.diag(lda.priors_)

t = np.cov(df_norm.T)
b = g.T @ dg @ g
w = t - b

#test F(clase - 1, randuri - clase)
# F(q-1,n-q)
# luam valorile de pe diagonalele principale
# Fcalc = Trace(B)*(q-1)^(-1)/Trace(W)/(n-q)^(-1)
# Fcalc reprezinta puterea de discriminare
# Coloanele sunt semnificative pentru LDA daca p-val < 0.05

f = (np.diag(b)/(q-1))/(np.diag(w)/(n-q))
p_value = 1-sts.f.cdf(f,q-1,n-q)
f_df = pd.DataFrame({"F-Value":pd.Series(f),"P-Value":pd.Series(p_value)})
f_df.index = df_norm.columns
print(f_df)

print("Coloane Semnificative")
print(f_df[f_df["P-Value"]<0.05])


#2 Plot instante si centri LD1 LD2

#Centri
centri = df_lda.groupby(by=encoded).mean()
plt.scatter(df_lda["L1"],df_lda["L2"],c=encoded)
plt.scatter(centri["L1"],centri["L2"],c=centri.index,s=200)
plt.xlabel("L1")
plt.ylabel("L2")
plt.title("Proiectie L1,L2")
plt.show()

#3 Plot distributii

kdeplot(x = df_lda['L1'],hue=labels,fill=True)
plt.show()

#4 Predictie Bayes
x_train,x_test,y_train,y_test = train_test_split(df_lda,labels,train_size=0.7)
model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

#5 Evaluare Bayes
print("Acuratete: "+str(accuracy_score(y_test,y_pred)))
print("Matrice de confuzie:")
print(confusion_matrix(y_test,y_pred))

#6 Model Linear (Yup , trebuia sa fac split de la inceput, my bad)

x_train,x_test,y_train,y_test = train_test_split(df_norm,labels,train_size=0.7)
y_pred = lda.predict(x_test)
print(y_pred)

#7 Evaluare LDA

print("Acuratete: "+str(accuracy_score(y_test,y_pred)))
print("Matrice de confuzie:")
print(confusion_matrix(y_test,y_pred))