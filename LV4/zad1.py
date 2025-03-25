#Skripta zadatak_1.py ucitava podatkovni skup iz data_C02_emission.csv .
#Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih numerickih ulaznih velicina. 

import sklearn
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 

data=pd.read_csv('data_C02_emission.csv')

#a) Odaberite željene numericke velicine specificiranjem liste s nazivima stupaca. Podijelite podatke na skup za ucenje i skup za testiranje u omjeru 80%-20%

numerical_features=['Fuel Consumption City (L/100km)','Fuel Consumption Hwy (L/100km)','Fuel Consumption Comb (L/100km)','Fuel Consumption Comb (mpg)', 'Engine Size (L)', 'Cylinders']
X=data[numerical_features].to_numpy()
y=data['CO2 Emissions (g/km)'].to_numpy()

X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#b) Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova o jednoj numerickoj velicini. 
# Pri tome podatke koji pripadaju skupu za ucenje oznacite plavom bojom, a podatke koji pripadaju skupu za testiranje oznacite crvenom bojom.

plt.figure()
plt.scatter(X_train[:,0], y_train, color='b', label='Training')
plt.scatter(X_test[:,0], y_test, color='r',label='Test')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#c) Izvršite standardizaciju ulaznih velicina skupa za ucenje. Prikažite histogram vrijednosti jedne ulazne velicine prije i nakon skaliranja.
# Na temelju dobivenih parametara skaliranja transformirajte ulazne velicine skupa podataka za testiranje.

scl=MinMaxScaler()

X_train_n=scl.fit_transform(X_train)
X_test_n=scl.transform(X_test)

plt.figure()
plt.subplot(1,2,1)
plt.hist(X_train[:,0], bins=20, color='g', edgecolor='black')
plt.subplot(1,2,2)
plt.hist(X_train_n[:,0], bins=20, color='y', edgecolor='black')
plt.show()
plt.show()

#d) Izgradite linearni regresijski model.Ispišite u terminal dobivene parametre modela i povežite ih s izrazom 4.6.

linearModel=LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)


#e) Izvršite procjenu izlazne velicine na temelju ulaznih velicina skupa za testiranje. Prikažite pomocu dijagrama raspršenja odnos izmedu stvarnih vrijednosti izlazne velicine i procjene dobivene modelom.

y_test_p=linearModel.predict(X_test_n)

plt.figure()
plt.scatter(y_test, y_test_p,color='b', alpha=0.6)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

print('Shape of y_test:', y_test.shape)

#f) Izvršite vrednovanje modela na nacin da izracunate vrijednosti regresijskih metrika na skupu podataka za testiranje.

print('MSE-Mean Squared Error:{:.5f}'.format(mean_squared_error(y_test, y_test_p)))
print('RMSE-Root Mean Squared Error: {:.5f}'.format(root_mean_squared_error(y_test, y_test_p)))
print('MAE-Mean Absolute Error:{:.5f}'.format(mean_absolute_error(y_test, y_test_p)))
print('MAPE-Mean Absolute Percentage Error:{:.5f}'.format(mean_absolute_percentage_error(y_test, y_test_p))+'%')
print('R2:{:.5F}'.format(r2_score(y_test, y_test_p)))

#g) Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ulaznih velicina?

#Dodavanjem relevantnih značajki tj. onih povezanih sa emisijom CO2, metrika greške se smanjuje (model predviđa bolje).
#Dodavanjem irelevantnih ili redundantnih značajki može doći do pogoršanja performansi( overfitting), neznazne promjene i dr.