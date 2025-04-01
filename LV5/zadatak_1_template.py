import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split

#Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije ulazne velicine.
# Podaci su podijeljeni na skup za ucenje i skup za testiranje modela.
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


#a) Prikažite podatke za ucenje u  x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite  s obzirom na klasu. 
# Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi marker (npr. ’x’). 
# Koristite funkciju scatter koja osim podataka prima i parametre c i cmap kojima je moguce definirati boju svake klase.

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm',marker='o', label='Trening skup', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm',marker='x', label='Test skup', alpha=0.8)
plt.legend()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Prikaz trening i test podataka obojenih po klasi')
plt.show()

#b) Izgradite model logisticke regresije pomocu scikit-learn biblioteke na temelju skupa podataka za ucenje.

LogRegressionModel=LogisticRegression()
LogRegressionModel.fit(X_train, y_train)

#c) Pronadite u atributima izgradenog modela parametre modela. Prikažite granicu odluke naucenog modela u ravnini 
#  x1 − x2 zajedno s podacima za ucenje. Napomena: granica odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.

theta1, theta2=LogRegressionModel.coef_[0]  #Dohvaćamo koeficijente (θ1,θ2)
theta0=LogRegressionModel.intercept_[0] #Dohvaćamo koeficijent θ0

def decision_boundary(x1):
    return (-theta0 - theta1 * x1) / theta2

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Trening skup', alpha=0.8)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test skup', alpha=0.8)

x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
plt.plot(x_values, decision_boundary(x_values), color='black', label='Granica odluke')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistička regresija: podaci i granica odluke')
plt.legend()
plt.show()

#d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgradenog modela logisticke regresije. 
# Izracunajte i prikažite matricu zabune na testnim podacima. Izracunate tocnost, preciznost i odziv na skupu podataka za testiranje.

y_pred = LogRegressionModel.predict(X_test)  #predvida klase za testni skup 

cm=confusion_matrix(y_test, y_pred)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LogRegressionModel.classes_)
disp.plot(cmap='Blues')
plt.title('Matrica zabune - test skup')
plt.show()

accuracy=accuracy_score( y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test, y_pred)

print(f"Tocnost (accuracy): {accuracy:.2f}")
print(f"Preciznost (precision): {precision:.2f}")
print(f"Odziv (recall): {recall:.2f}")

#e) Prikažite skup za testiranje u ravnini x1 −x2. 
# Zelenom bojom oznacite dobro klasificirane primjere dok pogrešno klasificirane primjere oznacite crnom bojom.

correct = y_test == y_pred
incorrect = ~correct

plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Točno klasificirani', marker='o')
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='black', label='Pogrešno klasificirani', marker='x')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Test skup: točne i pogrešne klasifikacije')
plt.legend()
plt.grid(True)
plt.show()
