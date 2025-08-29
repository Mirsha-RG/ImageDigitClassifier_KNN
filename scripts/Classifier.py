# -------------------------------
# Librerías necesarias
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Cargar dataset de dígitos (0-9)
digits = load_digits()

# Visualizar algunos datos y targets
digits.target[::100]   # Ejemplo de targets cada 100 imágenes
digits.images[15]      # Imagen número 15
digits.data[15]        # Vector de pixeles de la imagen 15

# -------------------------------
# Mostrar algunas imágenes con sus etiquetas
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4)) 
for ax, image, target in zip(axes.ravel(), digits.images, digits.target):
    ax.imshow(image, cmap=plt.cm.gray_r)  # Mostrar imagen en escala de grises
    ax.set_xticks([])                      # Ocultar ticks en X
    ax.set_yticks([])                      # Ocultar ticks en Y
    ax.set_title(target)                   # Poner el número como título

plt.tight_layout()
plt.show()  

# -------------------------------
# Separar datos y etiquetas
X, y = digits.data, digits.target
# Dividir en entrenamiento y prueba (70%-30%), manteniendo la proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# -------------------------------
# Entrenar un KNN manual básico
knn = KNeighborsClassifier(n_neighbors=10)  # KNN con 10 vecinos
knn.fit(X_train, y_train)                   # Entrenar modelo
prediccion = knn.predict(X_test)            # Predicción en test
esperado = y_test

# Mostrar métricas del KNN manual
print(classification_report(y_test, prediccion, zero_division=0))

# Matriz de confusión del KNN manual
confusion_matrix(y_test, prediccion)

# -------------------------------
# GridSearchCV para optimizar hiperparámetros del KNN
model = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],       # Número de vecinos a probar
    'weights': ['uniform', 'distance'],# Peso uniforme o según distancia
    'p': [1, 2]                        # 1=Manhattan, 2=Euclidiana
}
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Validación cruzada

# Configurar y entrenar GridSearchCV
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kf)
grid_search.fit(X_train, y_train)

# -------------------------------
# Resultados del GridSearch
print("\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)

print("\nMejor score en validación cruzada:")
print(grid_search.best_score_)

# -------------------------------
# Mejor modelo entrenado con los parámetros óptimos
best_knn = grid_search.best_estimator_
pred_best = best_knn.predict(X_test)

print("\nReporte de clasificación del mejor KNN:")
print(classification_report(y_test, pred_best, zero_division=0))

# -------------------------------
# Matrices de confusión comparativas
cm_manual = confusion_matrix(y_test, prediccion)
cm_best = confusion_matrix(y_test, pred_best)

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# KNN manual
sns.heatmap(cm_manual, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title("KNN Manual")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Verdadero")

# KNN optimizado
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title("KNN Optimizado")
axes[1].set_xlabel("Predicción")
axes[1].set_ylabel("Verdadero")

plt.tight_layout()
plt.show()

# -------------------------------
# Comparación de métricas entre KNN manual y optimizado
metrics_manual = [
    accuracy_score(y_test, prediccion),
    f1_score(y_test, prediccion, average='weighted'),
    precision_score(y_test, prediccion, average='weighted'),
    recall_score(y_test, prediccion, average='weighted')
]

metrics_best = [
    accuracy_score(y_test, pred_best),
    f1_score(y_test, pred_best, average='weighted'),
    precision_score(y_test, pred_best, average='weighted'),
    recall_score(y_test, pred_best, average='weighted')
]

labels = ['Accuracy', 'F1', 'Precision', 'Recall']

# Crear DataFrame para graficar
df_metrics = pd.DataFrame({
    'Métrica': labels*2,
    'Valor': metrics_manual + metrics_best,
    'Modelo': ['Manual']*4 + ['Optimizado']*4
})

# Graficar comparación de métricas
plt.figure(figsize=(8,5))
sns.barplot(x='Métrica', y='Valor', hue='Modelo', data=df_metrics)
plt.ylim(0,1)
plt.title("Comparación de métricas: KNN Manual vs Optimizado")
plt.show()
