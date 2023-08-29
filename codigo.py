import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv(r'C:\Users\Marquina Denis\Desktop\pro\m\data.csv') #cargo mi data
X=data.loc[:,['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']] #todos datos continuos
y=data.loc[:,['CO2EMISSIONS']] #separo los datos en X y
#separo las variables pra el train y testt
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=4)
np.sum(Xtrain.isnull()) #cuento los misings abs(arroja que no hay datos vacios)
                        #por lo que no realizo asignaciones
#estandarizo
from sklearn.preprocessing import StandardScaler                
sc=StandardScaler()
Xtrain_sc=pd.DataFrame(sc.fit_transform(Xtrain))
Xtrain_sc.columns=['sc_ENGINESIZE','sc_CYLINDERS','sc_FUELCONSUMPTION_COMB'] #coloco columnas e indices adecuados
Xtrain_sc.index=Xtrain.index
#veifico la estandarizacion
np.mean(Xtrain_sc,axis=0)
np.std(Xtrain_sc,axis=0)
import matplotlib.pyplot as plt
import seaborn as sns
plt.hist(Xtrain.FUELCONSUMPTION_COMB,bins=50)
sns.distplot(Xtrain.FUELCONSUMPTION_COMB,hist=True,kde=True,bins=50) # en esta variable se observa ciertos outliers,
cotas_FUELCONS=np.nanpercentile(Xtrain.FUELCONSUMPTION_COMB,[0,95]) #designo percentil 95 para acotar 
[np.min(Xtrain.FUELCONSUMPTION_COMB,),np.max(Xtrain.FUELCONSUMPTION_COMB)] #veo los max y minimos dela data 
Xtrain.loc[Xtrain.FUELCONSUMPTION_COMB<=cotas_FUELCONS[0],'FUELCONSUMPTION_COMB']=cotas_FUELCONS[0]
Xtrain.loc[Xtrain.FUELCONSUMPTION_COMB>=cotas_FUELCONS[1],'FUELCONSUMPTION_COMB']=cotas_FUELCONS[1]
[np.min(Xtrain.FUELCONSUMPTION_COMB,),np.max(Xtrain.FUELCONSUMPTION_COMB)]#verifico lo acotado
Xtrain_f=pd.concat([Xtrain,Xtrain_sc],axis=1) #unifico variables
#sobre el test
              

Xtest_sc=pd.DataFrame(sc.transform(Xtest))
Xtest_sc.columns=['sc_ENGINESIZE','sc_CYLINDERS','sc_FUELCONSUMPTION_COMB'] #coloco columnas e indices adecuados
Xtest_sc.index=Xtest.index
#veifico la estandarizacion
np.mean(Xtest_sc,axis=0)
np.std(Xtest_sc,axis=0)


#veo las cotas
[np.min(Xtest.FUELCONSUMPTION_COMB,),np.max(Xtest.FUELCONSUMPTION_COMB)] #veo los max y minimos dela data 
Xtest.loc[Xtest.FUELCONSUMPTION_COMB<=cotas_FUELCONS[0],'FUELCONSUMPTION_COMB']=cotas_FUELCONS[0]
Xtest.loc[Xtest.FUELCONSUMPTION_COMB>=cotas_FUELCONS[1],'FUELCONSUMPTION_COMB']=cotas_FUELCONS[1]
[np.min(Xtest.FUELCONSUMPTION_COMB,),np.max(Xtest.FUELCONSUMPTION_COMB)]#verifico lo acotado
Xtest_f=pd.concat([Xtest,Xtest_sc],axis=1)

#Analisis de correlaciones
data_train_f=pd.concat([Xtrain_f,ytrain],axis=1)  #concateno las variables
correl_x=Xtrain_f.corr()  #analizo correlaciones
data_train_f2=data_train_f.drop(['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB'],axis=1)#elimino corr exacta

correl_x=data_train_f2.corr()
data_train_f2.columns

from sklearn.linear_model import LinearRegression
Xtrain_f2=Xtrain_f.drop(['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB'],axis=1)# elimino la corelacion exacta
Xtest_f2=Xtest_f.drop(['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB'],axis=1)
reg=LinearRegression() #genra el objeto
reg.fit(Xtrain_f2,ytrain)#entreno el modelo
reg.score(Xtrain_f2,ytrain)#R2=85.1%  TRAIN
reg.score(Xtest_f2,ytest) #R2=86.9%   TEST

correl_CO2EMISSIONS=pd.DataFrame(data_train_f2.corr()['CO2EMISSIONS'])
correl_CO2EMISSIONS['abs']=np.abs(correl_CO2EMISSIONS.CO2EMISSIONS)
correl_CO2EMISSIONS.sort_values(by='abs',inplace=True,ascending=False)
#las variables tienen de 80 a 86 % corellacion por la k se conservo las variables y no realizo otro reentreno

pred_train=reg.predict(Xtrain_f2)  #realizo pedicciones
pred_test=reg.predict(Xtest_f2)
#analisis R2
reg.score(Xtrain_f2,ytrain) )#R2=85.1%  TRAIN ,resulta igual por lo que no elimina variables
reg.score(Xtest_f2,ytest)#R2=86.9%   TEST


