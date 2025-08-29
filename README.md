# ImageDigitClassifier_KNN

Modelo de machine learning que detecta y clasifica dígitos en imágenes. 
Permite reconocer números escritos a mano o generados digitalmente, usando KNN optimizado con GridSearchCV y validación cruzada.

## 🔹 Características

- Clasificación de dígitos del 0 al 9.  
- Optimización de hiperparámetros con **GridSearchCV** y **K-Fold**.  
- Evaluación con métricas: **Accuracy**, **F1-score**, **Precision** y **Recall**.  
- Visualización de resultados con **matrices de confusión** y gráficos comparativos con Seaborn.  
- Comparación entre modelo manual y modelo optimizado.  

---

## 🔹 Resultados

- Mejor KNN encontrado: {'n_neighbors': 3, 'weights': 'distance', 'p': 2}  
- Mejor score en validación cruzada: 0.985  
- Matrices de confusión y métricas comparativas muestran la mejora respecto al modelo manual.

### Matriz de confusión - KNN Manual
![Matriz manual](docs/matriz_confusion_manual.png)

### Matriz de confusión - KNN Optimizado
![Matriz optimizado](docs/matriz_confusion_optimizado.png)

Puedes consultar todos los resultados completos aquí: 👉 [docs/resultados.txt](docs/resultados.txt)


---

## 🔹 Uso
- Entrena un KNN básico con n_neighbors=10.  
- Optimiza los hiperparámetros con GridSearchCV y K-Fold.  
- Evalúa el mejor modelo en datos de prueba.  
- Visualiza matrices de confusión y métricas con gráficos de Seaborn.

---

## 🔹 Tecnologías / Librerías

- Python 3.x  
- scikit-learn  
- pandas  
- numpy  
- matplotlib  
- seaborn
  
---

## 🔹 Instalación

1. Clonar el repositorio:

```bash
git clone https://github.com/tu_usuario/DigitDetector.git
