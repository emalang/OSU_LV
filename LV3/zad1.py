import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('data_C02_emission.csv')

#a) Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke velicine konvertirajte u tip category.
print('Broj mjerenja:', len(data))
print('Tip veličina:\n', data.dtypes)

print('Izostale vrijednosti:\n', data.isnull().sum())
data=data.dropna(axis=0)

print("Duplicirane vrijednosti:\n", data.duplicated().sum())
data=data.drop_duplicates()

data['Make']=data['Make'].astype('category')
data['Model']=data['Model'].astype('category')
data['Vehicle Class']=data['Vehicle Class'].astype('category')
data['Transmission']=data['Transmission'].astype('category')
data['Fuel Type']=data['Fuel Type'].astype('category')

#b) Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ime proizvodaca, model vozila i kolika je gradska potrošnja.

print('Tri automobila sa najvećom gradskom potrošnjom:')
print(data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(3)[['Make','Model','Fuel Consumption City (L/100km)']])

print('Tri automnobila sa najmanjom gradskom potrošnjom:')
print(data.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).tail(3)[['Make','Model','Fuel Consumption City (L/100km)']])


#c) Koliko vozila ima velicinu motora izmedu 2.5 i 3.5 L? Kolika je prosjecna C02 emisija plinova za ova vozila?

filtered_vehicles=data[(data['Engine Size (L)']>=2.5) & (data['Engine Size (L)']<=3.5)]
print('Broj vozila sa velicinom motora [2.5,3.5]:', len(filtered_vehicles))

print('Prosjecna CO2 emisija filtriranih vozila:', filtered_vehicles['CO2 Emissions (g/km)'].mean())

#d) Koliko mjerenja se odnosi na vozila proizvodaca Audi? Kolika je prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara?

audi_number=data[data['Make']=='Audi']
print('Broj audi vozila:', len(audi_number))

audi_4cylinders=audi_number[audi_number['Cylinders']==4]
print('Prosjecna Audi CO2 emisija filtriranih vozila:', audi_4cylinders['CO2 Emissions (g/km)'].mean())

#e) Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na broj cilindara?

groupedcars_by_cylinders=data.groupby('Cylinders')
print('Broj automobila prema broju cilindra:', groupedcars_by_cylinders.size())
print('Prosjecna CO2 emisija plinova s obzirom na broj cilindara:', groupedcars_by_cylinders['CO2 Emissions (g/km)'].mean())

#f) Kolika je prosjecna gradska potrošnja u slucaju vozila koja koriste dizel, a kolika za vozila koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?
diesel_cars=data[data['Fuel Type']=='D']
regular_cars=data[data['Fuel Type']=='X']

print('Prosjecna gradska potrosnja[Diesel]:', diesel_cars['Fuel Consumption City (L/100km)'].mean())
print('Prosjecan median[Diesel]:', diesel_cars['Fuel Consumption City (L/100km)'].median())
print('Prosjecna gradska potrosnja[Regular]:', regular_cars['Fuel Consumption City (L/100km)'].mean())
print('Prosjecan median[Regular]:', regular_cars['Fuel Consumption City (L/100km)'].median())

#g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?

filtered_cars=data[(data['Cylinders']==4) & (data['Fuel Type']=='D')] 
print('Vozilo:',filtered_cars.sort_values(by='Fuel Consumption City (L/100km)', ascending=False).head(1))

#h) Koliko ima vozila ima rucni tip mjenjaca (bez obzira na broj brzina)?

cars_with_manual= data[data['Transmission'].str.startswith('M')]
print('Broj automobila sa rucnim mjenjacem: ', len(cars_with_manual))

#i) Izracunajte korelaciju izmedu numerickih velicina. Komentirajte dobiveni rezultat.

print('Korelacija:', data.corr(numeric_only=True))