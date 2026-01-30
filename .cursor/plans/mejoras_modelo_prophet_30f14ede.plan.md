---
name: Mejoras modelo Prophet
overview: Plan para mejorar el modelo Prophet en 5_Prophet.ipynb, implementando estacionalidad, transformación de datos, validación cruzada y tuning de hiperparámetros para reducir MAE y RMSE.
todos:
  - id: enable-seasonality
    content: Habilitar yearly_seasonality=True y cambiar a seasonality_mode='multiplicative'
    status: pending
  - id: log-transform
    content: Implementar transformación log1p en variable objetivo y reversión con expm1
    status: pending
  - id: normalize-price
    content: Normalizar el regresor de precio (price_normalized)
    status: pending
  - id: hyperparams
    content: "Ajustar hiperparámetros: changepoint_prior_scale=0.05, seasonality_prior_scale=10"
    status: pending
  - id: cross-validation
    content: Implementar validación cruzada con prophet.diagnostics
    status: pending
  - id: evaluate-all-series
    content: Evaluar modelo mejorado en todas las series y comparar métricas
    status: pending
  - id: visualize-components
    content: Visualizar componentes del modelo (tendencia, estacionalidad) para análisis
    status: pending
isProject: false
---

# Mejoras del Modelo Prophet

## Problema Actual

El modelo Prophet en [`5_Prophet.ipynb`](final_version_TFG/5_Prophet.ipynb) tiene un rendimiento deficiente:

- **MAE**: ~34,611
- **RMSE**: ~38,318

La configuración actual es muy básica:

```python
model = Prophet(
    yearly_seasonality=False,  # Deshabilitada
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="additive",
)
model.add_regressor("price")  # Sin normalizar
```

---

## Mejoras Propuestas

### 1. Habilitar Estacionalidad Anual

Con datos mensuales de ventas, la estacionalidad anual es casi segura. Prophet la detecta automáticamente si los datos tienen > 2 ciclos completos (>24 meses).

```python
yearly_seasonality=True,  # Habilitar
```

### 2. Usar Estacionalidad Multiplicativa

Para datos de ventas con alta varianza, el modo multiplicativo captura mejor la relación entre tendencia y estacionalidad.

```python
seasonality_mode="multiplicative"  # Cambiar de "additive"
```

### 3. Transformar la Variable Objetivo

Aplicar transformación **log1p** para estabilizar la varianza:

```python
# Antes de entrenar
df_train['y'] = np.log1p(df_train['y'])

# Después de predecir
forecast['yhat'] = np.expm1(forecast['yhat'])
```

### 4. Normalizar el Regresor de Precio

El precio debe normalizarse para mejor convergencia:

```python
price_mean = df_train['price'].mean()
price_std = df_train['price'].std()
df_train['price_normalized'] = (df_train['price'] - price_mean) / price_std
model.add_regressor('price_normalized')
```

### 5. Ajustar Hiperparámetros Clave

| Parámetro | Valor Recomendado | Razón |

| ------------------------- | ----------------- | ------------------------------------------------------------------------------- |

| `changepoint_prior_scale` | 0.05 | Controla flexibilidad de tendencia. Valores bajos (0.01-0.1) evitan overfitting |

| `seasonality_prior_scale` | 10 | Controla fuerza de estacionalidad. Valores altos permiten patrones más fuertes |

| `interval_width` | 0.95 | Intervalos de confianza más amplios |

### 6. Implementar Validación Cruzada

Prophet tiene validación cruzada específica para series temporales:

```python
from prophet.diagnostics import cross_validation, performance_metrics

df_cv = cross_validation(
    model,
    initial='365 days',
    period='30 days',
    horizon='180 days'
)
metrics = performance_metrics(df_cv)
```

### 7. Considerar Growth Logistic (Opcional)

Para evitar predicciones negativas:

```python
model = Prophet(
    growth='logistic',
    # ...
)
df_train['floor'] = 0
df_train['cap'] = df_train['y'].max() * 2
```

---

## Modelo Mejorado Propuesto

```python
def train_prophet_improved(df_one_series, use_log_transform=True):
    df = df_one_series.sort_values('date').copy()
    df = df.groupby("date", as_index=False).agg({
        "volume.sales": "sum",
        "price": "mean"
    })
    df = df.set_index("date").asfreq("M").reset_index()
    df = df.rename(columns={"date": "ds", "volume.sales": "y"})

    # Transformación log
    if use_log_transform:
        df['y'] = np.log1p(df['y'])

    # Normalizar precio
    price_mean = df['price'].mean()
    price_std = df['price'].std()
    if price_std > 0:
        df['price_norm'] = (df['price'] - price_mean) / price_std

    # Modelo
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )

    if price_std > 0:
        model.add_regressor('price_norm')

    model.fit(df.dropna(subset=['y']))

    return model, {'price_mean': price_mean, 'price_std': price_std, 'use_log': use_log_transform}
```

---

## Archivos a Modificar

- [`final_version_TFG/5_Prophet.ipynb`](final_version_TFG/5_Prophet.ipynb): Implementar todas las mejoras

---

## Resultado Esperado

Con estas mejoras, se espera una reducción significativa en las métricas de error:

- MAE: Reducción del 30-50%
- RMSE: Reducción del 30-50%
- Mejor captura de patrones estacionales
- Predicciones más estables sin valores negativos
