import pandas as pd
import seaborn as sb
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn import datasets, preprocessing, model_selection, svm
from scipy import stats
from statistics import mean, variance
from sklearn.model_selection import KFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error


#%%

#Jeux de données
data=pd.DataFrame(pd.read_csv("jeux_données.csv"))

#on enlève les promo avec une durée >2 mois
data=data[data.ShipDuration<60]
data.hist(column="ShipDuration")


data=data[data.DepthOfDiscount<50]
data=data[data.DepthOfDiscount>0]
data.hist(column="DepthOfDiscount")


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

#Préparation des données  pour le modèle 
data=data.drop(columns=['DomainID','PromoID2','Redemption','CalcMethod','ShipStart','StoreStart','CalcMetricCode'])
Y=data["upliftRéelPourcentage"]
X=data.drop(columns=['upliftRéelPourcentage','SkuID2','upliftRéel','upliftPrédit','CompanyID2','ConditionFormat','DocType','StatusGroupDesc','StatusDesc','AttributeName1','AttributeName2'])

# la focntion k-fold pour la validation croisée
def k_fold_cross_val_poly(folds, X, Y):
    kf = KFold(n_splits=folds)
    kf_dict = dict([("fold_%s" % i,[]) for i in range(1, folds+1)])
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        # le jeu de donnée
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        #le modèle de la régression linéaire
        model=LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        kf_dict["fold_%s" % fold].append(test_mse)
        
        # cosntruire un data frame des données prédits 
        y_pred=pd.DataFrame(y_pred,index=y_test.index,columns=["upliftRéelPourcentage"])
        
        # les nouveaux jeu de données 
        Y=pd.DataFrame(index=y_train.index,columns=["upliftRéelPourcentage"])
        X=pd.DataFrame(columns=X_train.columns)
        X=X.append(X_train)
        X=X.append(X_test)
        Y["upliftRéelPourcentage"]=y_train
        Y=pd.concat([Y,y_pred])
        
        # Convertion de la liste array pour la moyenne des erreurs
        kf_dict["fold_%s" % fold] = np.array(kf_dict["fold_%s" % fold])
    return kf_dict

print(k_fold_cross_val_poly(3, X, Y))