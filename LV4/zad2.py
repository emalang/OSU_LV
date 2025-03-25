#Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoricku varijable „Fuel Type“ kao ulaznu velicinu. 
#Pri tome koristite 1-od-K kodiranje kategorickih velicina. Radi jednostavnosti nemojte skalirati ulazne velicine. Komentirajte dobivene rezultate.
#Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu vozila radi?


import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder 

data=pd.read_csv('data_C02_emission.csv')

ohe=OneHotEncoder()
ohe_df=pd.DataFrame(ohe.fit_transform(data[['Fuel Type']]).toarray())
data=data.join(ohe_df)

data.columns=['Make','Model','Vehicle Class','Engine Size (L)','Cylinders','Transmission','Fuel Type','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','CO2 Emissions (g/km)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']

y=data['CO2 Emissions (g/km)'].copy()
X=data.drop('CO2 Emissions (g/km)',axis=1)

X_train_all, X_test_all, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1)

X_train=X_train_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]
X_test=X_test_all[['Engine Size (L)','Cylinders','Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)','Fuel0', 'Fuel1', 'Fuel2', 'Fuel3']]

linearModel=LinearRegression()
linearModel.fit(X_train, y_train)
y_test_p=linearModel.predict(X_test)

plt.scatter(X_test['Fuel Consumption City (L/100km)'], y_test,c='b', label='Real values',s=5, alpha=0.5)
plt.scatter(X_test['Fuel Consumption City (L/100km)'],y_test_p, c='r',label='Prediction', s=5, alpha=0.5)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

max_Error=max_error(y_test, y_test_p)
print('Vrijednost za maksimalnu pogresku:', max_Error)
print('Model vozila sa maksimalnom pogreskom:',X_test_all[abs(y_test-y_test_p==max_Error)]['Model'].iloc[0])
