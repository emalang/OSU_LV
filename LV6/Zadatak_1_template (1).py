import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

#6.5.1.1.
#Izradite algoritam KNN na skupu podataka za ucenje (uz K=5). Izracunajte tocnost klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. 
# Usporedite dobivene rezultate s rezultatima logisticke regresije. Što primjecujete vezano uz dobivenu granicu odluke KNN modela?

KNN_model=KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)

y_train_knn_p=KNN_model.predict(X_train_n)
y_test_knn_p=KNN_model.predict(X_test_n)

print('KNN model:')
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_knn_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_knn_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN granica odluke (Tocnost: " + "{:0.3f}".format(accuracy_score(y_train, y_train_knn_p)) + ")")
plt.tight_layout()
plt.show()

#Zakljucak:
# - Logisticka regresija ima linearnu granicu odluke.
# - KNN model ima nelinearnu granicu koja bolje prati raspodjelu podataka.
# - KNN model ima vecu tocnost nego logiasticka regresija

#6.5.1.2.
#Kako izgleda granica odluke kada je K =1 i kada je K = 100?

for K in [1, 100]:
    model = KNeighborsClassifier(n_neighbors=K)
    model.fit(X_train_n, y_train)

    y_train_pred = model.predict(X_train_n)
    y_test_pred = model.predict(X_test_n)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nKNN (K={K}):")
    print(f"Točnost na skupu za učenje: {train_acc:.3f}")
    print(f"Točnost na skupu za testiranje: {test_acc:.3f}")

    #Vizualizacija granice odluke
    plot_decision_regions(X_train_n, y_train, classifier=model)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title(f"KNN granica odluke (K={K})\nTočnost: {train_acc:.3f}")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

#Zakljucak:
#K=1:
#    Granica odluke je jako nazubljena i detaljna.
#    Model savršeno pamti trening podatke (vrlo visoka točnost), ali se može loše ponašati na test skupu zbog overfittinga.
#    Granica se "uvlači" između točaka svake klase, prati čak i outliere.

#K=100:
#    Granica je gotovo linearna ili blago zakrivljena.
#    Model ignorira lokalne obrasce i pokušava generalizirati na temelju cijelog skupa.
#    To dovodi do underfittinga, i niže točnosti ako su klase složenije razdvojene.

#6.5.2.
#Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN za podatke iz Zadatka 1.

param_grid={'n_neighbors':np.arange(1,31)}

knn_cv=GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn_cv.fit(X_train_n, y_train)

print("\nUnakrsna validacija - odabir najboljeg K:")
print(f"Najbolji K: {knn_cv.best_params_['n_neighbors']}") #optimalna vrijednost K prema validaciji
print(f"Najbolja prosječna točnost (cv): {knn_cv.best_score_:.3f}") #prosječna tocnost na 5-fold cross-validaciji

#6.5.3.
#Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju te prikažite dobivenu granicu odluke.
# Mijenjajte vrijednost hiperparametra C i γ. 
# Kako promjena ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje? 
#Mijenjajte tip kernela koji se koristi. Što primjecujete?

C_value=1
gamma_value=100

svm_rbf=svm.SVC(C=C_value, kernel='rbf', gamma=gamma_value)
svm_rbf.fit(X_train_n, y_train)

y_test_pred=svm_rbf.predict(X_test_n)
test_acc= accuracy_score( y_test, y_test_pred)

print(f"\nSVM RBF (C={C_value}, gamma={gamma_value})")
print(f"Točnost na test skupu: {test_acc:.3f}")

plot_decision_regions(X_train_n, y_train, classifier=svm_rbf)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title(f'SVM RBF kernel\nC={C_value}, gamma={gamma_value}\nTočnost test: {test_acc:.3f}')
plt.tight_layout()
plt.show()

#Komentari:
#C--> Kontrolira koliko jako model pokušava izbjeći pogreške.
#γ (gamma)-->Kontrolira koliko daleko model "vidi" — koliki utjecaj imaju točke.

#RBF kernel s pažljivo odabranim C i γ (npr. C=1, γ=1) daje najbolji balans između točnosti i granice odluke.
# Preveliki C i γ → overfitting (model nauči previše detalja), granica je "nazubljena".
# Premali C i γ → underfitting (model previše pojednostavljen), granica previše glatka.

#linear kernel nije dovoljan jer podaci nisu linearno odvojivi.
# rbf kernel najčešće daje najbolju točnost i najrealističniju granicu odluke.

#6.5.4.
#Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ algoritma SVM za problem iz Zadatka 1.

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 10]
}

svm=svm.SVC(kernel='rbf')
grid_svm=GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, scoring='accuracy')
grid_svm.fit(X_train_n,y_train)

print("\nOptimalni hiperparametri (SVM RBF):")
print(f"Najbolji C: {grid_svm.best_params_['C']}")
print(f"Najbolji gamma: {grid_svm.best_params_['gamma']}")
print(f"Najbolja prosječna točnost (cv): {grid_svm.best_score_:.3f}")