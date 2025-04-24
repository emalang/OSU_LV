#Napišite skriptu koja ce ucitati izgradenu mrežu iz zadatka 1 i MNIST skup 
#podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
#skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu 
#mrežom.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model

#Ucitavanje modela iz memorije
model=keras.models.load_model('mnist_model.h5')

#Ucitavanje MNIST skupa
(x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data()

#Priprema podataka
x_test_s = x_test.astype("float32") / 255
x_test_s = np.expand_dims(x_test_s, -1)

#Predikcije
y_pred = model.predict(x_test_s)
y_pred_labels = np.argmax(y_pred, axis=1)

#Pronalazak losih klasifikacija
misclassified_indices = np.where(y_pred_labels != y_test)[0]

#Prikaz svih 9 pogresnih klasifikacija
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_indices[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}, Predviđeno: {y_pred_labels[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()