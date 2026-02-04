---
name: Implementación SHAP Catboost
overview: Implementar análisis SHAP completo en 6_Catboost.ipynb para interpretar la importancia y el impacto de cada variable en las predicciones del modelo global de CatBoost.
todos:
  - id: import-shap
    content: Añadir celda de importación de SHAP y configuración
    status: completed
  - id: explainer-values
    content: Crear TreeExplainer y calcular SHAP values
    status: completed
  - id: bar-plot
    content: Implementar gráfico de importancia global (bar plot)
    status: completed
  - id: beeswarm-plot
    content: Implementar beeswarm plot
    status: completed
  - id: dependence-plots
    content: Crear dependence plots para price, lag_1, lag_12
    status: completed
  - id: waterfall-plot
    content: Implementar waterfall plot para predicción individual
    status: completed
  - id: importance-table
    content: Crear tabla de importancia y guardar CSV
    status: completed
  - id: summary
    content: Añadir celda de resumen e interpretación
    status: completed
isProject: false
---

# Implementación Completa de SHAP en CatBoost

## Contexto

El notebook [6_Catboost.ipynb](final_version_TFG/6_Catboost.ipynb) actualmente entrena un modelo CatBoost global pero no incluye análisis de interpretabilidad. SHAP (SHapley Additive exPlanations) permitirá entender qué variables son más importantes y cómo afectan a las predicciones.

Ya existe una implementación de referencia en [catBoostRegressor.ipynb](new_models/catBoostRegressor.ipynb) que se puede adaptar.

## Variables disponibles después del entrenamiento

```python
model          # CatBoostRegressor entrenado
X_train        # Features de entrenamiento
X_test         # Features de test
y_test         # Target de test (log)
y_test_original # Target de test (escala original)
feature_cols   # Lista de nombres de features
categorical_features  # ['brand', 'supermarket', 'variant', 'pack_size']
```

## Implementación

### Celda 13: Importar SHAP y configurar

```python
# Instalar si es necesario: !pip install shap
import shap
shap.initjs()  # Para visualizaciones interactivas en Jupyter
```

### Celda 14: Crear explainer y calcular SHAP values

```python
# TreeExplainer está optimizado para modelos basados en árboles (CatBoost, XGBoost, etc.)
explainer = shap.TreeExplainer(model)

# Calcular SHAP values para el conjunto de test
# Nota: puede tardar unos segundos dependiendo del tamaño de X_test
shap_values = explainer.shap_values(X_test)

print(f"Shape de shap_values: {shap_values.shape}")
print(f"Expected value (baseline): {explainer.expected_value:.4f}")
```

### Celda 15: Gráfico de Importancia Global (Bar Plot)

```python
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Importancia Global de Variables (SHAP)", fontsize=14)
plt.tight_layout()
plt.savefig("../images_markdown/shap_importance_bar.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Celda 16: Beeswarm Plot (Impacto y Dirección)

```python
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Impacto de Variables en Predicciones (SHAP Beeswarm)", fontsize=14)
plt.tight_layout()
plt.savefig("../images_markdown/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Celda 17: Dependence Plots para variables clave

```python
# Dependence plot para PRECIO (variable exógena principal)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Precio
plt.subplot(1, 3, 1)
shap.dependence_plot("price", shap_values, X_test, show=False)
plt.title("Efecto del Precio")

# Lag_1 (autocorrelación)
plt.subplot(1, 3, 2)
shap.dependence_plot("lag_1", shap_values, X_test, show=False)
plt.title("Efecto del Lag_1")

# Lag_12 (estacionalidad)
plt.subplot(1, 3, 3)
shap.dependence_plot("lag_12", shap_values, X_test, show=False)
plt.title("Efecto del Lag_12 (Estacionalidad)")

plt.tight_layout()
plt.savefig("../images_markdown/shap_dependence.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Celda 18: Waterfall Plot para predicciones individuales

```python
# Seleccionar una observación de ejemplo (ej: la primera del test)
idx = 0

# Crear Explanation object para waterfall
explanation = shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    data=X_test.iloc[idx],
    feature_names=X_test.columns.tolist()
)

plt.figure(figsize=(12, 6))
shap.plots.waterfall(explanation, show=False)
plt.title(f"Descomposición de Predicción (Observación {idx})", fontsize=12)
plt.tight_layout()
plt.savefig("../images_markdown/shap_waterfall.png", dpi=150, bbox_inches='tight')
plt.show()
```

### Celda 19: Tabla de Importancia de Variables

```python
# Crear DataFrame con ranking de importancia
importance_df = pd.DataFrame({
    'Variable': X_test.columns,
    'SHAP_Mean_Abs': np.abs(shap_values).mean(axis=0),
    'SHAP_Std': np.abs(shap_values).std(axis=0)
}).sort_values('SHAP_Mean_Abs', ascending=False)

# Añadir porcentaje de importancia
importance_df['Importancia_%'] = (
    importance_df['SHAP_Mean_Abs'] / importance_df['SHAP_Mean_Abs'].sum() * 100
)

print("=" * 60)
print("RANKING DE IMPORTANCIA DE VARIABLES (SHAP)")
print("=" * 60)
print(importance_df.to_string(index=False, float_format='{:.4f}'.format))
print("=" * 60)

# Guardar como CSV
importance_df.to_csv('shap_importance.csv', index=False)
```

### Celda 20: Análisis de Interacciones (opcional, más avanzado)

```python
# Interaction values (puede ser computacionalmente costoso)
# Descomenta si quieres ver interacciones entre variables
# shap_interaction = explainer.shap_interaction_values(X_test[:100])
# shap.summary_plot(shap_interaction, X_test[:100])
```

### Celda 21: Resumen de hallazgos

```python
# Mostrar top 5 variables más importantes
print("\n" + "=" * 60)
print("TOP 5 VARIABLES MÁS IMPORTANTES")
print("=" * 60)
for i, row in importance_df.head(5).iterrows():
    print(f"  {row['Variable']}: {row['Importancia_%']:.1f}%")
print("=" * 60)

# Interpretación automática
print("\n📊 INTERPRETACIÓN:")
top_var = importance_df.iloc[0]['Variable']
if 'lag' in top_var:
    print(f"  - La variable más importante es {top_var}, indicando alta autocorrelación")
if 'price' in importance_df.head(3)['Variable'].values:
    print("  - El precio es una variable significativa para las predicciones")
if 'month' in importance_df.head(5)['Variable'].values:
    print("  - Existe componente estacional capturado por el mes")
```

## Archivos a modificar

- [6_Catboost.ipynb](final_version_TFG/6_Catboost.ipynb): Añadir celdas 13-21 después de la evaluación del modelo

## Dependencias

Asegurar que SHAP esté instalado:

```bash
pip install shap
```

## Resultados esperados

- Bar plot mostrando importancia global de cada feature
- Beeswarm plot mostrando impacto y dirección de cada variable
- Dependence plots para price, lag_1 y lag_12
- Waterfall plot explicando una predicción individual
- CSV con ranking de importancia numérico
