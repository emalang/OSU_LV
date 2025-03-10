#Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
#ženama. Skripta zadatak_2.py ucitava dane podatke u obliku numpy polja data pri cemu je u
#prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treci
#stupac polja je masa u kg.

import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("data.csv", skiprows=1, delimiter=',')

#a) Na temelju velicine numpy polja data, na koliko osoba su izvršena mjerenja?
print("Broj osoba:", len(data))

#b) Prikažite odnos visine i mase osobe pomocu naredbe matplotlib.pyplot.scatter.
plt.scatter(data[:,1], data[:,2],s=2)
plt.title("Odnos mase i visine")
plt.show()

#c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.
plt.scatter(data[::50,1], data[::50,2], s=5)
plt.title("Odnos mase i visine svake 50.osobe")
plt.show()

#d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom podatkovnom skupu.
print("Minimalna visina:", data[:,1].min())
print("Maksimalna visina:", data[:,1].max())
print("Srednja vrijednost visine:", data[:,1].mean())

#Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili muškarce, stvorite polje koje zadrži bool 
#vrijednosti i njega koristite kao indeks retka. ind = (data[:,0] == 1)

m=(data[:,0]==0)
z=(data[:,0]==1)

print("Minimalna visina za žene:", np.min(data[z,1]))
print("Maksimalna visina za žene:",np.max(data[z,1]))
print("Srednja vrijednost visine za žene:", np.mean(data[z,1]))

print("Minimalna visina za muškarce:", np.min(data[m,1]))
print("Maksimalna visina za muškarce:",np.max(data[m,1]))
print("Srednja vrijednost visine za muškarce:", np.mean(data[m,1]))