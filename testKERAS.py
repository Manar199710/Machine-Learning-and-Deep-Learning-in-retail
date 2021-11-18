
# example of making predictions for a regression problem
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from numpy import array
from sklearn import datasets, model_selection
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from sklearn import metrics
import matplotlib.pyplot as plt
#Jeux de données
data=pd.DataFrame(pd.read_csv("jeux_données.csv"))

#on enlève les promo avec une durée >2 mois
data=data[data.ShipDuration<60]
#data.hist(column="ShipDuration")


data=data[data.DepthOfDiscount<50]
data=data[data.DepthOfDiscount>0]
#data.hist(column="DepthOfDiscount")

data["ConditionCode_1.0"].corr(data["upliftRéelPourcentage"])
data["ConditionCode_2.0"].corr(data["upliftRéelPourcentage"])
data["ConditionCode_20.0"].corr(data["upliftRéelPourcentage"])
data["ConditionCode_20.0"].corr(data["upliftRéelPourcentage"])
data["upliftRéelPourcentage"].corr(data["ConditionValue"])
data["upliftRéelPourcentage"].corr(data["DepthOfDiscount"])

#L'occurrence des modalités de UpliftRéelPourcentage
Occurence=pd.DataFrame({'Nombre_occurences':data['upliftRéelPourcentage'].value_counts()})
Occurence=Occurence.sort_index(ascending=True)
Occurence['Pourcentage cumulé'] = 100*Occurence.Nombre_occurences.cumsum()/Occurence.Nombre_occurences.sum()
#on supprime les 3 premiers % (valeurs négatives) et es % > à 99.7% (sup à 103)

data=data[data.upliftRéelPourcentage<100]
data=data[data.upliftRéelPourcentage>0]

#cosntruction le jeu données et variable pour le modèle
data=data.drop(columns=['DomainID','PromoID2','Redemption','CalcMethod','ShipStart','StoreStart','CalcMetricCode'])
Y=data["upliftRéelPourcentage"]
X=data.drop(columns=['upliftRéelPourcentage','SkuID2','upliftRéel','upliftPrédit','CompanyID2','ConditionFormat','DocType','StatusGroupDesc','StatusDesc','AttributeName1','AttributeName2'])

# la séparation des données d'entrainement et celle du test
X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X,Y,test_size=.2)

# normalisation des données
cs=MinMaxScaler()
X_train = cs.fit_transform(X_train)
X_test = cs.transform(X_test)
MaxPorcentage=Y.max()
Y_train=Y_train/100
Y_test=Y_test/100

#Construction du modèle 
model = Sequential()
model.add(Dense(8, input_dim=X_train.shape[1], activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(1, activation="linear"))
opt = Adam(lr=1e-3, decay=1e-3 / 100)
model.compile(loss='mean_squared_error', optimizer=opt)
model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs=100, batch_size=8)


#la prédiction
print("Prédiction :")
preds = model.predict(X_test)

accuracy = metrics.r2_score(Y_test, preds)
print('Cross-Predicted Accuracy: ', accuracy)

# visualisation de Y_test et celle prédit
plt.scatter(Y_test, preds)
plt.xlabel('Valeurs vraies')
plt.ylabel('Predictions')

# le pourcentage absolue de la différence
erreur =abs(preds.flatten() - Y_test)

#la moyenne et l'écart type d'erreur
mean = np.mean(erreur*100)
std = np.std(erreur*100)
erreur.hist(bins=25)# l'istogramme des erreurs

# visualisation des données statistiques du modèle
print(" mean: {:.5f}, std: {:.5f}".format(mean, std))

