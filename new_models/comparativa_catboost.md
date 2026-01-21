# Comparativa Detallada: CatBoost para Predicción de Ventas

Este documento proporciona un análisis exhaustivo de dos implementaciones de modelos CatBoost para la predicción de series temporales de ventas: **catBoost.ipynb** y **catBoostRegressor.ipynb**.

---

## 📋 Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [catBoost.ipynb - Análisis Detallado](#2-catboostipynb---análisis-detallado)
3. [catBoostRegressor.ipynb - Análisis Detallado](#3-catboostregressoripynb---análisis-detallado)
4. [Comparación Detallada](#4-comparación-detallada)
5. [Métricas y Resultados](#5-métricas-y-resultados)
6. [Conclusiones y Recomendaciones](#6-conclusiones-y-recomendaciones)

---

## 1. Resumen Ejecutivo

| Característica | catBoost.ipynb | catBoostRegressor.ipynb |
|----------------|----------------|------------------------|
| **Enfoque** | Panel de datos completo | Datos disponibles únicamente |
| **Features categóricas** | `series_id` | `brand`, `supermarket`, `variant`, `pack_size` |
| **Función de pérdida** | RMSE | MAE |
| **Iteraciones** | 5,000 | 500 (con early stopping) |
| **Learning rate** | 0.03 | 0.1 |
| **Profundidad** | 8 | 6 |
| **Predicción futura** | No | Sí (método recursivo) |
| **MAE en test** | 11,203.80 | 8,686.28 |
| **R²** | 0.7058 | N/A |

---

## 2. catBoost.ipynb - Análisis Detallado

### 2.1 Carga y Preparación de Datos

El notebook comienza importando los datos desde `Datos_Market_copy.xlsx` y utilizando una clase utilitaria `SalesAnalysis` para el preprocesamiento inicial.

```python
raw_data = pd.read_excel("../data/Datos_Market_copy.xlsx")
sa = SalesAnalysis(raw_data)
data = sa.data
```

**Filtrado de datos:**
- Se filtran únicamente las 3 marcas principales: `brand-35`, `brand-14`, `brand-15`
- Total de datos tras filtrado: **3,403 filas**

**Creación del identificador de serie:**
```python
data['series_id'] = (
    data['brand'].astype(str) + '_' + 
    data['supermarket'].astype(str) + '_' + 
    data['variant'].astype(str) + '_' + 
    data['pack.size'].astype(str)
)
```

Esta combinación genera un identificador único para cada combinación de marca, supermercado, variante y tamaño de empaque.

### 2.2 Análisis de Completitud de Series

El notebook implementa una función para detectar series con datos incompletos:

```python
def series_less_than_36(data: pd.DataFrame, months: int = 36) -> pd.DataFrame:
    # Identifica series con menos de 36 meses de datos
```

**Resultados del análisis:**
- **67 series** (59.29%) tienen datos completos (≥36 meses)
- **46 series** (40.71%) tienen datos incompletos (<36 meses)

### 2.3 Construcción del Panel Completo (Punto Diferenciador Clave)

**Este es el aspecto más distintivo de este notebook.** Se crea un calendario completo para evitar discontinuidades temporales:

```python
all_months = pd.date_range(data["date"].min(), data["date"].max(), freq="ME")
uniques = pd.DataFrame({"series_id": data["series_id"].unique()})
all_months_df = pd.DataFrame({"date": all_months})
full = uniques.assign(_key=1).merge(all_months_df.assign(_key=1), on="_key").drop("_key", axis=1)
full = full.merge(data, on=["series_id", "date"], how="left")
```

Este proceso:
1. Genera todas las fechas posibles (rango completo)
2. Crea un producto cartesiano entre series y fechas
3. Hace un left join con los datos reales
4. **Resultado: 100% de las series tienen 36 datos** (incluyendo NaN donde no había datos originales)

### 2.4 Tratamiento de Valores Faltantes

```python
# Crear flag de datos faltantes
full["missing"] = full["volume.sales"].isna().astype(int)

# Rellenar ventas faltantes con 0
full["volume.sales"] = full["volume.sales"].fillna(0.0)

# Rellenar precios faltantes con forward fill y mediana
full["price"] = full.groupby("series_id")["price"].ffill()
full["price"] = full["price"].fillna(full["price"].median())
```

**Estrategia de imputación:**
- `missing`: Variable binaria (1/0) que indica si el dato era originalmente faltante
- `volume.sales`: Se imputan con 0 los meses sin ventas
- `price`: Se usa forward fill por serie, y mediana global como fallback

### 2.5 Transformación de la Variable Objetivo

```python
full["y"] = np.log1p(full["volume.sales"])
```

Se aplica transformación logarítmica `log(1+x)` para:
- Reducir asimetría de la distribución
- Estabilizar la varianza
- Manejar valores cero (gracias a +1)

### 2.6 Ingeniería de Features

#### Features de Calendario
```python
full["year"] = full["date"].dt.year
full["month"] = full["date"].dt.month
full["quarter"] = full["date"].dt.quarter

# Codificación cíclica para capturar estacionalidad
full["month_sin"] = np.sin(2*np.pi*full["month"]/12)
full["month_cos"] = np.cos(2*np.pi*full["month"]/12)
```

La codificación seno/coseno es importante porque:
- Captura la naturaleza cíclica del mes (diciembre está "cerca" de enero)
- Proporciona una representación continua de la estacionalidad

#### Lags (Retardos)
```python
def add_lags(panel, col, lags=(1,2,3,6,12)):
    for l in lags:
        panel[f"{col}_lag_{l}"] = panel.groupby("series_id")[col].shift(l)
    return panel

full = add_lags(full, "y")           # Lags 1,2,3,6,12 de la variable objetivo
full = add_lags(full, "price", lags=(1,))  # Lag 1 del precio
```

#### Medias Móviles (Rolling Means)
```python
# Shift(1) CRÍTICO para evitar data leakage
full["y_roll_mean_3"] = full.groupby("series_id")["y"].shift(1).rolling(3, min_periods=1).mean()
full["y_roll_mean_6"] = full.groupby("series_id")["y"].shift(1).rolling(6, min_periods=1).mean()
full["y_roll_mean_12"] = full.groupby("series_id")["y"].shift(1).rolling(12, min_periods=1).mean()
```

**Nota importante:** Se usa `shift(1)` antes del rolling para evitar **data leakage** (filtración de información del futuro).

#### Cambio de Precio
```python
full["price_change"] = full["price"] - full["price_lag_1"]
```

### 2.7 Definición del Modelo

**Features utilizadas:**
```python
feature_cols = (
    ["series_id"]  # Categórica
    + ["price", "missing", "year", "month", "quarter", "month_sin", "month_cos"]
    + [c for c in full.columns if c.startswith("y_lag_") or c.startswith("price_lag_")]
    + ["y_roll_mean_3", "y_roll_mean_6", "y_roll_mean_12", "price_change"]
)

cat_features = ["series_id"]  # Solo series_id como categórica
```

**Configuración del modelo:**
```python
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=5000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    eval_metric="RMSE",
    verbose=200
)
```

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| `loss_function` | RMSE | Penaliza más los errores grandes |
| `iterations` | 5,000 | Modelo con muchas iteraciones |
| `learning_rate` | 0.03 | Tasa conservadora para mejor generalización |
| `depth` | 8 | Árboles profundos para capturar patrones complejos |

### 2.8 Split Temporal

```python
cutoff = pd.Timestamp("2023-06-30")
train_idx = full["date"] <= cutoff
valid_idx = full["date"] > cutoff
```

- **Train:** Desde inicio hasta 30/06/2023 (2,814 filas)
- **Test:** Desde 01/07/2023 hasta 31/12/2023 (589 filas)

### 2.9 Uso de Pool de CatBoost

```python
train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)
```

Los `Pool` de CatBoost permiten:
- Manejo eficiente de memoria
- Procesamiento optimizado de features categóricas
- Mejor performance en entrenamiento

### 2.10 Evaluación y Visualización

**Métricas obtenidas:**
- **MAE:** 11,203.80
- **RMSE:** 28,190.91
- **R²:** 0.7058

El notebook incluye visualizaciones extensas:
1. Gráfico agregado: Predicciones vs reales por fecha
2. Scatter plot con línea perfecta
3. Distribución de errores (histograma y boxplot)
4. Series individuales representativas
5. Resumen de métricas por serie

---

## 3. catBoostRegressor.ipynb - Análisis Detallado

### 3.1 Carga de Datos

Similar al primer notebook, pero **incluye la marca "other"** en los datos cargados (143 series vs 113):

```python
data = data[
    data["brand"].isin(["brand-35", "brand-14", "brand-15"])
]  # Aunque el output muestra 143 combinaciones que incluyen 'other'
```

**Datos cargados:** 4,306 filas (más que catBoost.ipynb)

### 3.2 Creación de Features (Sin Panel Completo)

**Diferencia fundamental:** Este notebook NO crea un calendario completo. Trabaja únicamente con los datos disponibles.

#### Features Temporales
```python
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
```

**Nota:** No utiliza codificación cíclica (sin/cos) ni quarter.

#### Lags
```python
data['lag_1'] = data.groupby('series_id')['volume.sales'].shift(1)
data['lag_2'] = data.groupby('series_id')['volume.sales'].shift(2)
data['lag_3'] = data.groupby('series_id')['volume.sales'].shift(3)
data['lag_12'] = data.groupby('series_id')['volume.sales'].shift(12)

data['price_lag_1'] = data.groupby('series_id')['price'].shift(1)
data['price_lag_12'] = data.groupby('series_id')['price'].shift(12)
```

**Diferencias con catBoost.ipynb:**
- No incluye `lag_6` de la variable objetivo
- Incluye `price_lag_12` (estacionalidad del precio)

#### Rolling Means
```python
data['volume_sales_shifted'] = data.groupby('series_id')['volume.sales'].shift(1)
data['rolling_mean_3'] = data.groupby('series_id')['volume_sales_shifted'].rolling(window=3, min_periods=1).mean()
data['rolling_mean_6'] = data.groupby('series_id')['volume_sales_shifted'].rolling(window=6, min_periods=1).mean()
```

**Diferencia:** No incluye `rolling_mean_12`.

### 3.3 Transformación del Target

```python
data['target_log'] = np.log1p(data['volume.sales'])
```

Misma transformación que catBoost.ipynb.

### 3.4 Definición de Features

```python
feature_cols = [
    'month', 'year',
    'lag_1', 'lag_2', 'lag_3', 'lag_12',
    'rolling_mean_3', 'rolling_mean_6',
    'price', 'price_lag_1', 'price_lag_12',
    'brand', 'supermarket', 'variant', 'pack_size'
]

categorical_features = ['brand', 'supermarket', 'variant', 'pack_size']
```

**Diferencia crítica:** 
- **NO usa `series_id` como feature categórica**
- **USA las componentes individuales**: brand, supermarket, variant, pack_size
- **15 features** vs más features en catBoost.ipynb

### 3.5 Split Temporal

```python
train_cutoff = pd.Timestamp('2023-06-30')
test_start = pd.Timestamp('2023-07-01')
test_end = pd.Timestamp('2023-12-31')

train_data = data[data['date'] <= train_cutoff].copy()
test_data = data[(data['date'] >= test_start) & (data['date'] <= test_end)].copy()
```

- **Train:** 3,579 filas
- **Test:** 727 filas

### 3.6 Manejo de NaN (Diferente Estrategia)

```python
# Eliminar filas donde lag_12 es NaN
train_data_clean = train_data.dropna(subset=['lag_12']).copy()
test_data_clean = test_data.dropna(subset=['lag_12']).copy()
```

**Estrategia:** En lugar de imputar, **elimina las filas** donde faltan valores de lag_12.

**Consecuencia:**
- Train: 3,579 → **1,968 filas** (pérdida de 1,611 filas = 45%)
- Test: 727 → **672 filas** (pérdida de 55 filas = 7.5%)

### 3.7 Configuración del Modelo

```python
model = CatBoostRegressor(
    loss_function='MAE',        # ← Diferente: MAE vs RMSE
    iterations=500,             # ← Diferente: 500 vs 5000
    learning_rate=0.1,          # ← Diferente: 0.1 vs 0.03
    depth=6,                    # ← Diferente: 6 vs 8
    random_seed=42,
    verbose=100,
    cat_features=cat_indices
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,   # ← Nuevo: early stopping
    verbose=100
)
```

| Hiperparámetro | Valor | Comparación |
|----------------|-------|-------------|
| `loss_function` | MAE | Más robusto a outliers |
| `iterations` | 500 | Entrenamiento más rápido |
| `learning_rate` | 0.1 | Tasa más agresiva |
| `depth` | 6 | Árboles menos profundos |
| `early_stopping` | 50 | Previene sobreajuste |

**Resultado del entrenamiento:**
- El modelo se detuvo en la iteración **166** por early stopping
- Mejor test RMSE en iteración 166

### 3.8 Métricas de Evaluación

```python
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / (denominator + 1e-9)) * 100

def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error"""
    numerator = np.sum(np.abs(y_true - y_pred))
    denominator = np.sum(np.abs(y_true))
    return (numerator / (denominator + 1e-9)) * 100
```

**Métricas obtenidas:**
- **MAE:** 8,686.28
- **sMAPE:** 36.58%
- **WAPE:** 25.69%

### 3.9 Predicciones Futuras (Feature Exclusiva)

Este notebook incluye una **función de predicción recursiva** para generar forecasts futuros:

```python
def generate_future_predictions(model, last_data, horizon=6, price_scenario='same'):
    """
    Genera predicciones futuras usando método recursivo.
    
    Parameters:
    -----------
    model: CatBoostRegressor entrenado
    last_data: DataFrame con los últimos datos conocidos
    horizon: número de meses a predecir
    price_scenario: 'same', 'plus3', 'minus3'
    """
```

**Características del método recursivo:**
1. Usa predicciones anteriores como inputs para predicciones siguientes
2. Actualiza lags dinámicamente: `lag_1 = predicción_t-1`, `lag_2 = lag_1_anterior`, etc.
3. Actualiza rolling means con nuevas predicciones
4. Mantiene lag_12 estático (estacionalidad histórica)

**Escenarios de precio:**
- `same`: Mantiene el último precio conocido
- `plus3`: Incremento del 3% sobre precio base
- `minus3`: Decremento del 3% sobre precio base

**Predicciones generadas:** 774 predicciones por escenario (129 series × 6 meses)

### 3.10 Exportación de Resultados

```python
# Métricas por serie
metrics_df.to_csv('catboost_metrics_by_series.csv', index=False)

# Predicciones futuras
forecast_same.to_csv('catboost_forecast_same_price.csv', index=False)
forecast_plus3.to_csv('catboost_forecast_plus3_price.csv', index=False)
forecast_minus3.to_csv('catboost_forecast_minus3_price.csv', index=False)

# Predicciones del test set
test_predictions.to_csv('catboost_test_predictions.csv', index=False)
```

---

## 4. Comparación Detallada

### 4.1 Estrategia de Datos

| Aspecto | catBoost.ipynb | catBoostRegressor.ipynb |
|---------|----------------|------------------------|
| **Manejo de series incompletas** | Crea panel completo + imputa | Elimina filas con NaN |
| **Flag de datos faltantes** | Sí (`missing`) | No |
| **Datos de entrenamiento** | 2,814 filas | 1,968 filas |
| **Datos de test** | 589 filas | 672 filas |
| **Series evaluadas** | 113 | 122 |

**Ventajas de cada enfoque:**

**catBoost.ipynb (Panel completo):**
- ✅ No pierde información temporal
- ✅ Permite al modelo aprender patrones de "meses sin ventas"
- ✅ El flag `missing` puede ser predictivo
- ❌ Introduce datos artificiales (imputados)

**catBoostRegressor.ipynb (Eliminación):**
- ✅ Trabaja solo con datos reales
- ✅ Más simple de implementar
- ❌ Pierde ~45% de datos de entrenamiento
- ❌ Puede perder patrones importantes

### 4.2 Ingeniería de Features

| Feature | catBoost.ipynb | catBoostRegressor.ipynb |
|---------|----------------|------------------------|
| **month** | ✅ + sin/cos | ✅ (solo numérico) |
| **year** | ✅ | ✅ |
| **quarter** | ✅ | ❌ |
| **lag_1, lag_2, lag_3** | ✅ | ✅ |
| **lag_6** | ✅ | ❌ |
| **lag_12** | ✅ | ✅ |
| **rolling_mean_3, _6** | ✅ | ✅ |
| **rolling_mean_12** | ✅ | ❌ |
| **price_lag_1** | ✅ | ✅ |
| **price_lag_12** | ❌ | ✅ |
| **price_change** | ✅ | ❌ |
| **missing flag** | ✅ | ❌ |

### 4.3 Features Categóricas

| Notebook | Features Categóricas | Implicación |
|----------|---------------------|-------------|
| **catBoost.ipynb** | `series_id` | Aprende patrones específicos por serie |
| **catBoostRegressor.ipynb** | `brand`, `supermarket`, `variant`, `pack_size` | Aprende patrones generalizables por componentes |

**Análisis:**
- `series_id`: Permite embeddings específicos por serie, pero puede sobreajustar con pocas observaciones por serie
- Componentes separados: Mayor generalización, puede transferir conocimiento entre series similares

### 4.4 Hiperparámetros

| Parámetro | catBoost.ipynb | catBoostRegressor.ipynb | Impacto |
|-----------|----------------|------------------------|---------|
| `loss_function` | RMSE | MAE | MAE es más robusto a outliers |
| `iterations` | 5,000 | 500 | Mayor capacidad vs. riesgo de sobreajuste |
| `learning_rate` | 0.03 | 0.1 | Convergencia lenta/estable vs. rápida/riesgosa |
| `depth` | 8 | 6 | Mayor complejidad vs. mejor generalización |
| `early_stopping` | No | 50 rounds | Previene sobreajuste |

### 4.5 Funcionalidad Adicional

| Funcionalidad | catBoost.ipynb | catBoostRegressor.ipynb |
|---------------|----------------|------------------------|
| Predicción en test set | ✅ | ✅ |
| Predicción futura recursiva | ❌ | ✅ |
| Escenarios de precio | ❌ | ✅ (same, +3%, -3%) |
| Visualizaciones | ✅ Extensas | ✅ Moderadas |
| Exportación a CSV | ❌ | ✅ |
| Métricas por serie | ✅ | ✅ |
| R² Score | ✅ | ❌ |
| sMAPE/WAPE | ❌ | ✅ |

---

## 5. Métricas y Resultados

### 5.1 Comparación de Performance

| Métrica | catBoost.ipynb | catBoostRegressor.ipynb | Mejor |
|---------|----------------|------------------------|-------|
| **MAE** | 11,203.80 | 8,686.28 | catBoostRegressor ✅ |
| **RMSE** | 28,190.91 | N/A | - |
| **R²** | 0.7058 | N/A | - |
| **sMAPE** | N/A | 36.58% | - |
| **WAPE** | N/A | 25.69% | - |

### 5.2 Análisis de la Diferencia en MAE

La diferencia de ~2,500 en MAE podría explicarse por:

1. **Función de pérdida:** MAE optimiza directamente para el error absoluto medio
2. **Early stopping:** Previene sobreajuste en catBoostRegressor
3. **Datos de evaluación diferentes:** Distintas filas en test set
4. **Features categóricas:** El uso de componentes individuales puede generalizar mejor

### 5.3 Series con Mayor/Menor Error

**catBoost.ipynb - Peor desempeño:**
- `brand-35_supermarket-D_standard_351 - 500 GR`: MAE = 129,338
- `brand-35_supermarket-A_standard_351 - 500 GR`: MAE = 122,714

**catBoostRegressor.ipynb - Peor desempeño:**
- `brand-35_supermarket-D_standard_351 - 500 GR`: MAE = 77,119
- `brand-35_supermarket-A_standard_351 - 500 GR`: MAE = 84,779

Las mismas series problemáticas aparecen en ambos notebooks, lo que sugiere que son inherentemente difíciles de predecir (posiblemente por alta variabilidad o patrones inusuales).

---

## 6. Conclusiones y Recomendaciones

### 6.1 Conclusiones

1. **catBoostRegressor.ipynb obtiene mejor MAE** (8,686 vs 11,204), probablemente por:
   - Optimización directa de MAE
   - Early stopping que previene sobreajuste
   - Mejor generalización con features categóricas descompuestas

2. **catBoost.ipynb es más completo** en términos de:
   - Manejo de datos faltantes (no pierde información)
   - Features de calendario más sofisticadas (sin/cos)
   - Visualizaciones más extensas

3. **catBoostRegressor.ipynb es más práctico** para producción:
   - Genera predicciones futuras
   - Incluye análisis de sensibilidad a precios
   - Exporta resultados a CSV

### 6.2 Recomendaciones

**Para mejorar catBoost.ipynb:**
- Implementar early stopping
- Considerar cambiar a MAE como función de pérdida
- Añadir predicción recursiva futura
- Exportar resultados a CSV

**Para mejorar catBoostRegressor.ipynb:**
- Implementar el enfoque de panel completo en lugar de eliminar NaN
- Añadir codificación cíclica (sin/cos) para el mes
- Incluir `lag_6` y `rolling_mean_12`
- Añadir el flag `missing` como feature

**Enfoque híbrido recomendado:**
```python
# Combinar lo mejor de ambos notebooks:
# 1. Panel completo + flag missing (de catBoost.ipynb)
# 2. Features categóricas descompuestas (de catBoostRegressor.ipynb)
# 3. MAE + early stopping (de catBoostRegressor.ipynb)
# 4. Sin/cos encoding + más features (de catBoost.ipynb)
# 5. Predicción recursiva futura (de catBoostRegressor.ipynb)
```

### 6.3 Tabla Resumen Final

| Criterio | Ganador | Razón |
|----------|---------|-------|
| **Precisión (MAE)** | catBoostRegressor | 8,686 vs 11,204 |
| **Manejo de datos** | catBoost | No pierde información |
| **Features engineering** | catBoost | Más sofisticado |
| **Producción/Uso real** | catBoostRegressor | Predicciones futuras + CSV |
| **Interpretabilidad** | catBoostRegressor | Features individuales |
| **Prevención sobreajuste** | catBoostRegressor | Early stopping |

---

## Anexo: Estructura de Features

### catBoost.ipynb
```
Features (total: ~18):
├── Categóricas
│   └── series_id
├── Temporales
│   ├── year, month, quarter
│   └── month_sin, month_cos
├── Lags de ventas
│   └── y_lag_1, y_lag_2, y_lag_3, y_lag_6, y_lag_12
├── Lags de precio
│   └── price_lag_1
├── Rolling means
│   └── y_roll_mean_3, y_roll_mean_6, y_roll_mean_12
├── Precio
│   └── price, price_change
└── Indicador
    └── missing
```

### catBoostRegressor.ipynb
```
Features (total: 15):
├── Categóricas
│   └── brand, supermarket, variant, pack_size
├── Temporales
│   └── month, year
├── Lags de ventas
│   └── lag_1, lag_2, lag_3, lag_12
├── Lags de precio
│   └── price_lag_1, price_lag_12
├── Rolling means
│   └── rolling_mean_3, rolling_mean_6
└── Precio
    └── price
```

---

*Documento generado para el análisis comparativo de modelos CatBoost en el proyecto TFG_ADE*
