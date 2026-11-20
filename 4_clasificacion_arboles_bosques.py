import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="especie")

print("Primeros registros:")
print(X.head())
print("\nCantidad de registros:", X.shape[0])
print("Clases:", iris.target_names)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)
arbol = DecisionTreeClassifier(max_depth=3, random_state=42)
arbol.fit(X_train, y_train)
bosque = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
bosque.fit(X_train, y_train)
pred_arbol = arbol.predict(X_test)
pred_bosque = bosque.predict(X_test)
precision_arbol = accuracy_score(y_test, pred_arbol)
precision_bosque = accuracy_score(y_test, pred_bosque)

print("\nPrecision del arbol de decision:", round(precision_arbol, 4))
print("Precision del bosque aleatorio:", round(precision_bosque, 4))

print("\nReporte de clasificacion - Arbol de decision")
print(classification_report(y_test, pred_arbol, target_names=iris.target_names))

print("\nReporte de clasificacion - Bosque aleatorio")
print(classification_report(y_test, pred_bosque, target_names=iris.target_names))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_arbol = confusion_matrix(y_test, pred_arbol)
cm_bosque = confusion_matrix(y_test, pred_bosque)

sns.heatmap(
    cm_arbol, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=axes[0]
)
axes[0].set_title("Arbol de decision")
axes[0].set_xlabel("Prediccion")
axes[0].set_ylabel("Valor real")

sns.heatmap(
    cm_bosque, annot=True, fmt="d", cmap="Greens",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    ax=axes[1]
)
axes[1].set_title("Bosque aleatorio")
axes[1].set_xlabel("Prediccion")
axes[1].set_ylabel("Valor real")

plt.tight_layout()
plt.show()
plt.figure(figsize=(16, 8))
plot_tree(
    arbol,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)
plt.title("Arbol de decision para clasificar flores Iris")
plt.show()
importancias = pd.Series(
    bosque.feature_importances_,
    index=iris.feature_names
).sort_values(ascending=False)

print("\nImportancia de las caracteristicas:")
print(importancias)

importancias.sort_values().plot(kind="barh", figsize=(8, 4))
plt.title("Importancia de las caracteristicas - Random Forest")
plt.xlabel("Importancia")
plt.ylabel("Caracteristica")
plt.tight_layout()
plt.show()
resultados = pd.DataFrame({
    "Modelo": ["Arbol de decision", "Bosque aleatorio"],
    "Precision": [precision_arbol, precision_bosque]
})

print("\nComparacion de modelos:")
print(resultados)
