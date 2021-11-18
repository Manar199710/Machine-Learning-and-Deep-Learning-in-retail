#%% 1 Importation
import pandas as pd
import pylab as pyplot


#Fact Pricing
dfFactPricing=pd.DataFrame(pd.read_csv("EMEA_FactPricing.csv"))


# %% 2 Prep et pivot sur factpromo
dfFactPromo=pd.DataFrame(pd.read_csv("EMEA_FactPromo.csv"))
dfFactPromo.drop(dfFactPromo.loc[dfFactPromo['CannibType']!=0].index,inplace=True) #on enleve les cannib type différents de 0 pour éviter les doublons
dfFactPromo = dfFactPromo[['DomainID','PromoID2','SkuID2','MetricCode','WithValue','WithoutValue']]  

#set the multi index
dfFactPromo.set_index(['DomainID','PromoID2','SkuID2'], inplace= True) 
index_names = list(dfFactPromo.index.names)
dfFactPromo.reset_index(inplace=True)
index_list = dfFactPromo[index_names].values
index_tuples = [tuple(i) for i in index_list]
dfFactPromo = dfFactPromo.assign(index_tuples=index_tuples) 

dfFactPromo = dfFactPromo.pivot(index="index_tuples",columns='MetricCode',values='WithValue')
index_tuples = dfFactPromo.index
index = pd.MultiIndex.from_tuples(index_tuples, names=index_names)
dfFactPromo.index = index
dfFactPromo.reset_index(inplace=True)

dfFactPromo = dfFactPromo[['DomainID','PromoID2','SkuID2',1010,1040,1100,1135,4100]]



#%% 3 Créa DfPromo (dataframe fina) join dfFactPromo, DimPromoConditio et DimPromo
dfDimPromoCondition=pd.DataFrame(pd.read_csv("EMEA_DimPromoCondition.csv"))
dfDimPromoCondition=dfDimPromoCondition.drop(columns=['ConditionDesc','SpendReasonDesc','PriceCustomer','PriceRetailer'])

dfDimPromo=pd.DataFrame(pd.read_csv("DimPromo.csv"))
dfDimPromo.drop(columns=['AttributeName1','AttributeName2','StatusGroupDesc','StatusDesc'])
dfDimPromo.drop(index=10407,inplace=True)

dfPromo = dfDimPromo.join(dfDimPromoCondition.drop(columns=['DomainID','CompanyID2']).set_index('PromoID2'),on='PromoID2', how='left').set_index(['PromoID2','SkuID2'])
dfPromo = dfPromo.join(dfFactPromo.drop(columns=['DomainID']).set_index(['PromoID2','SkuID2']), how='left')
index_names = list(dfPromo.index.names)
dfPromo=dfPromo.dropna(how='any')
dfPromo.reset_index(inplace=True)

#on enleve les promo de plus de 1 an
dfPromo=dfPromo.drop(dfPromo[dfPromo['ShipDuration']>90].index)

#4100 base prédite
dfPromo['upliftPrédit']=dfPromo[1040]
dfPromo['upliftPrédit pourcentage']=dfPromo['upliftPrédit']/dfPromo[4100]

dfPromo['upliftRéel']=dfPromo[1010]-dfPromo[4100]
dfPromo['upliftRéelPourcentage']=dfPromo['upliftRéel']/dfPromo[4100]

# Transformation du SpendReasonCode et ConditionCode en binaire selon leurs modalités
display(dfPromo.columns)
cols = ['SpendReasonCode','ConditionCode']
for col in cols:
    display('Colonne: ' + col)
    display(dfPromo[col].value_counts())
data_dummies = pd.get_dummies(dfPromo, columns=['SpendReasonCode','ConditionCode'])


# %% enregistrement dans un fichier csv
data_dummies.to_csv("jeux_données.csv", index=False)
