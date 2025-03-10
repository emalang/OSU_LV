#Skripta zadatak_3.py ucitava sliku ’road.jpg ’. Manipulacijom odgovarajuce numpy matrice pokušajte:
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

img=plt.imread('road.jpg')

#a) posvijetliti sliku,
plt.imshow(img, cmap='gray', alpha=0.75)
plt.title('Posvijetljena slika')
plt.show()

#b) prikazati samo drugu cetvrtinu slike po širini,
img_quarter=img[:, img.shape[1]//4:img.shape[1]//2]
plt.imshow(img_quarter, cmap='gray')
plt.title("Prikaz druge četvrtine slike")
plt.show()

#c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu,
img_rotated=np.rot90(img, k=3)
plt.imshow(img_rotated, cmap='gray')
plt.title("Rotirana slika")
plt.show()

#d) zrcaliti sliku.
img_mirrored=np.fliplr(img)
plt.imshow(img_mirrored, cmap='gray')
plt.title("Zrcaljena slika")
plt.show()