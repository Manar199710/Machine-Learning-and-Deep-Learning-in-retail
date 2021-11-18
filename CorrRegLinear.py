#%% importation pandas
import pandas as pd
import pylab as pyplot
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import statsmodels.api as sm
from sklearn import datasets, model_selection, preprocessing,
from scipy import stats
from statistics import mean, variance
#%%

#Jeux de données
data=pd.DataFrame(pd.read_csv("jeux_données.csv"))

#on enlève les promo avec une durée >2 mois
data=data[data.ShipDuration<60]
data.hist(column="ShipDuration")


data=data[data.DepthOfDiscount<50]
data=data[data.DepthOfDiscount>0]
data.hist(column="DepthOfDiscount")

#on enleve les uplift réel pas crédible
data=data[data.upliftRéelPourcentage<100]
data=data[data.upliftRéelPourcentage>-35]
data.hist(column="upliftRéelPourcentage")

#data=data[data.upliftRéel<3500]
#data=data[data.upliftRéel>-250]
#data.hist(column="upliftRéel")
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

data=data[data.upliftRéelPourcentage<103]
data=data[data.upliftRéelPourcentage>0]

#Corrélation
data=data.drop(columns=['DomainID','PromoID2','Redemption','CalcMethod','ShipStart','StoreStart','CalcMetricCode'])

# MAtrice de corrélation 
DataCor=data.corr()
sb.heatmap(data.corr()).plot()

#Données
Y=data["upliftRéelPourcentage"]
X=data.drop(columns=['upliftRéelPourcentage','SkuID2','upliftRéel','upliftPrédit','CompanyID2','ConditionFormat','DocType','StatusGroupDesc','StatusDesc','AttributeName1','AttributeName2'])
X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X,Y,test_size=.2)


#LinearRegression1
lm = LinearRegression()
rfe = RFE(lm)
rfe = rfe.fit(X_train, Y_train)
print('les colonnes du vecteurs X :  \n ',X_train.columns.values)
print('\n le vecteur booléen: \n',rfe.support_)# Printing the boolean results
print('\n Ordre des variables:',rfe.ranking_)  


#linearRegression2
columns=['StoreDuration', 'PriceStandard', 'ConditionValue', 'SpendReasonCode_2.0','SpendReasonCode_3.0', 'SpendReasonCode_7.0', 'ConditionCode_1.0','ConditionCode_2.0', 'ConditionCode_20.0', 'ConditionCode_30.0']
X=data[columns]
X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X,Y,test_size=.2)
reg = LinearRegression().fit(X_train, Y_train)
reg.score(X_train, Y_train)
reg.intercept_
Coef=pd.DataFrame(reg.coef_,X_train.columns,columns=['Coefficients'])
print('les coefficients de notre modèle : \n',Coef)


y_predict=reg.predict(X_test)
erreur=y_predict-Y_test
erreur=erreur[erreur>-20]
mean(erreur)
variance(erreur)
erreur.hist(bins=25)

