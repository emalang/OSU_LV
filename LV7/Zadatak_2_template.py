import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

#1. Otvorite skriptu zadatak_2.py. Ova skripta ucitava originalnu RGB sliku test_1.jpg te ju transformira u podatkovni skup koji dimenzijama odgovara izrazu (7.2) pri cemu je n
#broj elemenata slike, a m je jednak 3. Koliko je razlicitih boja prisutno u ovoj slici? 
unique_colors=np.unique(img_array, axis=0)
print('Broj boja:', unique_colors.shape[0])

#2. Primijenite algoritam K srednjih vrijednosti koji ce pronaci grupe u RGB vrijednostima elemenata originalne slike.
K=5
kmeans=KMeans(n_clusters=K, init='k-means++', random_state=0)
kmeans.fit(img_array)

centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print('Centroidi boja:')
print(centroids)

#3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajucim centrom.
img_array_aprox=centroids[labels] #novi niz RGB vrijednosti- zamjena svakog piksela bojom centra kojem pripada
img_approx=np.reshape(img_array_aprox,(w,h,d)) #povratak u oblik 3D slike

plt.figure()
plt.title(f"Kvantizirana slika (K = {K})")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()

#4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K . Komentirajte dobivene rezultate.
for K in [2, 4, 8, 16, 32]:
    kmeans = KMeans(n_clusters=K, init='random',random_state=0)
    kmeans.fit(img_array)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    img_aprox = np.reshape(centroids[labels], (w, h, d))

    # Prikaz slika
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(img_aprox)
    axs[1].set_title(f"K = {K}")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

#Manji K (npr. 2, 4) – slika gubi puno detalja, boje su “blokovske”, ali konture su još vidljive.
#Srednji K (8–16) – dobar kompromis: smanjenje boja bez prevelikog gubitka kvalitete.
#Veći K (32+) – slika postaje sve sličnija originalnoj, ali gubi se smisao kvantizacije jer broj boja opet raste.

#5. Primijenite postupak na ostale slike.

#6. Graficki prikažite ovisnost J o broju grupa K . Koristite atribut inertia objekta klase KMeans. 
# Možete li uociti lakat koji upucuje na optimalni broj grupa?
inertias=[]
K_values=range(1,21)

for K in K_values:
    k_means=KMeans(n_clusters=K, init='random',random_state=0)
    k_means.fit(img_array)
    inertias.append(k_means.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_values, inertias, marker='o')
plt.title("Lakat metoda – ovisnost J o broju grupa K")
plt.xlabel("Broj grupa K")
plt.ylabel("Vrijednost J (inertia_)")
plt.grid(True)
plt.show()

#Lakat se nalazi negdje između K = 4 i K = 6, što sugerira da je to optimalan broj boja (grupa) za kvantizaciju slike.
#Manji K → velika kompresija, gubitak boja i detalja
#Prevelik K → kvaliteta raste, ali više nema velike koristi (više boja, manje uštede).

# 7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku. Što primjecujete?
# Za svaku grupu napravi binarnu sliku

K = 5
kmeans = KMeans(n_clusters=K, init='random', random_state=0)
kmeans.fit(img_array)
labels = kmeans.labels_

for i in range(K):
    # Kreiraj binarnu masku: 1 (bijelo) ako piksel pripada grupi i, 0 (crno) inače
    mask = (labels == i).astype(np.uint8)
    binary_mask = np.reshape(mask, (w, h))

    # Množi s 255 da dobijemo sliku koju možemo prikazati
    binary_image = binary_mask * 255

    plt.figure()
    plt.title(f"Binarna crno-bijela slika za grupu {i+1}")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

#Svaka binarna slika (maska) pokazuje:
#bijelom bojom (1) → gdje se nalaze pikseli koji pripadaju toj grupi (boji),
#crnom bojom (0) → gdje se ne nalaze (sve ostalo).