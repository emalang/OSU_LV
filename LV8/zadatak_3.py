#Napišite skriptu koja ce ucitati izgradenu mrežu iz zadatka 1. Nadalje, skripta 
#treba ucitati sliku test.png sa diska. Dodajte u skriptu kod koji ce prilagoditi sliku za mrežu, 
#klasificirati sliku pomocu izgradene mreže te ispisati rezultat u terminal. Promijenite sliku 
#pomocu nekog grafickog alata (npr. pomocu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite 
#skriptu. Komentirajte dobivene rezultate za razlicite napisane znamenke.

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt

model=load_model('mnist_model.h5')

img = Image.open("test.png").convert('L')

img = img.resize((28, 28))                 
img_array = np.array(img).astype('float32') / 255 
img_array = np.expand_dims(img_array, axis=-1)  
img_array = np.expand_dims(img_array, axis=0)   

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

plt.imshow(img_array[0, :, :, 0], cmap='gray')
plt.title(f"Predviđena znamenka: {predicted_class}")
plt.axis('off')
plt.show()

print(f"\nModel predviđa da je znamenka: {predicted_class}")
