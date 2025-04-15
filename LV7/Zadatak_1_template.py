import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 1)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

#1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte nacin generiranja podataka.
# Da, za pokrenuti podatkovni primjer prepoznajem da imamo 3 grupe generiranih podataka.

#Za flagc=2 i dalje imamo 3 grupe , ali su elipticne i nagnute.
#Za flagc=3 imamo 4 grupe, ali neke su malo zgusnute, dok su druge raspršenije.
#Za flagc=4 imamo dvije nelinearne grupe u obliku dva koncentricna kruga.
#Za flagc=5 imamo dvije nelinearne grupe u obliku polumjeseca.


#2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer obojite ovisno o njegovoj pripadnosti pojedinoj grupi. 
# Nekoliko puta pokrenite programski kod. Mijenjate broj K. Što primjecujete? 

X=generate_data(500,1)
k=3

kmeans=KMeans(n_clusters=k, n_init=10, random_state=0)
labels=kmeans.fit_predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels,cmap='viridis')
plt.title(f'K-means klasteriranje (K={k})')
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.show()

#K=2 --> pokušava podijeliti podatke u dvije grupe, pri tome dvije spaja u jednu, a treća ostaje odvojena
#K=3 -->dobar za flagc=1, svaka od tri grupe dobiva svoju boju 
#K=4--> razlamanje podataka u vise grupa nego sto ih stvarno ima , podjela jedne grupe u dvije iako su cjelina

#3. Mijenjajte nacin definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka (koristite optimalni broj grupa). Kako komentirate dobivene rezultate?

# Za flagc=1 kMeans je radio odlicno, grupe su pravilno i jasno odvojene.
# Za flagc=2 kMeans zbog elipticnosti obavlja krivo razdvajanje
#Za flagc=3 kMeans zbog razlictih velicina i gustoce klastera, jedna grupa je pogrešno spojena, dok je druga razrijedena.
#Za flagc=4 kMeans ne moze prepoznati koncentricne krugove.
#Za flagc=5 kMeans ne moze pravilno razdvojiti nelinearne grupe.