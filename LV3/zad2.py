#Napišite programski kod koji ce prikazati sljedece vizualizacije:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data_C02_emission.csv')
plt.figure()

#a) Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.

data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
plt.show()

#b) Pomocu dijagrama raspršenja prikažite odnos izmedu gradske potrošnje goriva i emisije C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu
#velicina, obojite tockice na dijagramu raspršenja s obzirom na tip goriva.

data['Fuel Type']=data['Fuel Type'].astype('category')

data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c='Fuel Type', cmap='viridis')
plt.show()

#c) Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva. Primjecujete li grubu mjernu pogrešku u podacima?

data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()

#d) Pomocu stupcastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby.

data.groupby(by='Fuel Type')['Make'].count().plot(kind='bar')
plt.show()

#e) Pomocu stupcastog grafa prikažite na istoj slici prosjecnu C02 emisiju vozila s obzirom na broj cilindara.

data.groupby(by='Cylinders')['CO2 Emissions (g/km)'].mean().plot(kind='bar')
plt.show()