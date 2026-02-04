# Predicción de Ventas en el Sector Alimentación: Modelos de Series Temporales y Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

## Descripción del Proyecto

Este repositorio contiene el código y análisis desarrollado para el **Trabajo de Fin de Grado (TFG)** en **Administración y Dirección de Empresas (ADE)**. El proyecto aborda la **predicción de ventas de productos alimenticios** en múltiples supermercados utilizando técnicas avanzadas de series temporales y machine learning.

### Objetivo Principal

Desarrollar, implementar y comparar diferentes modelos predictivos para estimar las ventas mensuales de productos alimenticios, considerando múltiples factores como marcas, supermercados, variantes de producto y tamaños de empaque.

---

## Estructura del Proyecto

```
📦 code/
├── 📂 src/                           # Notebooks principales del análisis
│   ├── 1_preprocessing_data.ipynb    # Preprocesamiento y limpieza de datos
│   ├── 2_descriptive_analysis.ipynb  # Análisis exploratorio y descriptivo
│   ├── 3_ARIMA.ipynb                 # Modelo ARIMA univariante
│   ├── 4_ARIMAX.ipynb                # Modelo ARIMAX con variables exógenas
│   └── 5_Catboost.ipynb              # Modelo CatBoost con análisis SHAP
├── 📂 data/                          # Datos (no incluido en repositorio)
│   └── Datos_Market_copy.xlsx        # Dataset original
├── 📄 utilities.py                   # Módulo con funciones auxiliares
├── 📄 requirements.txt               # Dependencias del proyecto
├── 📄 .env                           # Variables de entorno
└── 📄 .gitignore                     # Archivos ignorados
```

---

## Datos Utilizados

### Fuente de Datos

- **Archivo**: `Datos_Market_copy.xlsx`
- **Granularidad original**: Datos semanales
- **Granularidad del análisis**: Datos mensuales (agregación)

### Variables Principales

| Variable       | Tipo       | Descripción                                               |
| -------------- | ---------- | --------------------------------------------------------- |
| `date`         | Temporal   | Fecha de la observación (mensual)                         |
| `volume.sales` | Numérica   | Volumen de ventas (variable objetivo)                     |
| `unit.sales`   | Numérica   | Unidades vendidas                                         |
| `value.sales`  | Numérica   | Valor monetario de las ventas                             |
| `price`        | Numérica   | Precio calculado (`value.sales / unit.sales`)             |
| `brand`        | Categórica | Marca del producto (brand-14, brand-15, brand-35, other)  |
| `supermarket`  | Categórica | Supermercado (A, B, C, D)                                 |
| `variant`      | Categórica | Variante del producto (flavoured, standard, light, vegan) |
| `pack.size`    | Categórica | Tamaño de empaque (5 categorías)                          |

### Identificación de Series Temporales

- **`series_id`**: Identificador único creado como combinación de `brand_supermarket_variant_pack.size`
- **Número de series**: ~105-143 series únicas (según filtros aplicados)

---

## Metodología

### 1. Preprocesamiento de Datos (`1_preprocessing_data.ipynb`)

El proceso de limpieza y preparación incluye:

- **Limpieza de marcas**: Agrupación de marcas minoritarias como "other"
- **Conversión temporal**: Agregación de datos semanales a mensuales
- **Creación de variables**: Cálculo de precio y creación de `series_id`
- **Filtrado de series**:
  - Exclusión de series con historial < 24 meses
  - Tratamiento de series con escala muy baja (< 100 unidades promedio)
  - Eliminación de series con exceso de valores cero
- **Transformaciones**:
  - Transformación Box-Cox para estabilizar varianza
  - Transformación logarítmica (`log1p`) cuando corresponde
- **División temporal**:
  - **Train**: Hasta 2023-05-31 (~80%)
  - **Test**: 2023-06-30 hasta 2023-12-31 (~20%)

### 2. Análisis Exploratorio (`2_descriptive_analysis.ipynb`)

- Estadísticas descriptivas de todas las variables
- Análisis de distribuciones
- Visualizaciones por categorías (marca, supermercado, variante)
- Análisis de correlaciones
- Detección de patrones y estacionalidad

### 3. Modelos Implementados

#### 3.1 ARIMA (`3_ARIMA.ipynb`)

Modelo autorregresivo integrado de media móvil para series univariantes.

**Características:**

- Tests de estacionariedad (ADF, KPSS)
- Selección automática de parámetros (p, d, q) con `auto_arima`
- Diagnóstico completo de residuos
- Análisis de ACF/PACF

#### 3.2 ARIMAX (`4_ARIMAX.ipynb`)

Extensión del modelo ARIMA incorporando variables exógenas.

**Características:**

- Regresores: precio, variables categóricas (supermarket, variant, pack.size, brand)
- Selección de variables mediante **Backward Elimination** (α = 0.05)
- Limpieza de multicolinealidad
- Evaluación en múltiples series temporales

#### 3.3 CatBoost (`5_Catboost.ipynb`)

Modelo de gradient boosting con soporte nativo para variables categóricas.

**Características:**

- **Modelo global**: Un único modelo para todas las series
- **Features utilizadas**:
  - Variables categóricas (brand, supermarket, variant, pack.size)
  - Lags temporales (lag_1, lag_12, etc.)
  - Precio
  - Features temporales (mes, año)
- **Interpretabilidad**: Análisis SHAP completo
  - Bar plot de importancia de variables
  - Beeswarm plot
  - Dependence plots
  - Waterfall plots

---

## Métricas de Evaluación

| Métrica   | Descripción                                                              |
| --------- | ------------------------------------------------------------------------ |
| **MAE**   | Mean Absolute Error - Error absoluto medio                               |
| **RMSE**  | Root Mean Squared Error - Raíz del error cuadrático medio                |
| **MAPE**  | Mean Absolute Percentage Error - Error porcentual absoluto medio         |
| **sMAPE** | Symmetric MAPE - MAPE simétrico                                          |
| **WAPE**  | Weighted Absolute Percentage Error - Error porcentual absoluto ponderado |

---

## Módulo de Utilidades (`utilities.py`)

### Clase `SalesAnalysis`

Módulo principal con todas las funciones auxiliares organizadas en categorías:

#### Inicialización y Limpieza

```python
cleaning_data()              # Limpieza de marcas
convert_weeks_to_months()    # Agregación mensual
add_price_column()           # Cálculo de precio
order_dataset_by_date()      # Ordenación temporal
```

#### Visualización

```python
detail_plot()                      # Gráfico de serie específica
separate_plot_by_flavour()         # Gráficos por variante
plot_everything_in_4_plots()       # Visualización completa 2x2
plot_everything()                  # Gráficos por supermercado y pack
plot_resid_ACF_PACF()              # Análisis de residuos ACF/PACF
analysis_residuals()               # Diagnóstico completo de residuos
```

#### Modelización

```python
modelization()                              # Regresión lineal con interacciones
modelization_with_backward_elimination()    # Selección de variables
regression_with_backward_elimination()      # OLS con backward elimination
ARIMA()                                     # Modelo ARIMA
ARIMAX()                                    # Modelo ARIMAX
```

#### Tests Estadísticos

```python
test_stationarity()           # Test ADF
ADF_KPSS_test()               # Tests de estacionariedad combinados
seasonal_stationarity_test()  # Estacionariedad estacional
residual_white_noise_test()   # Tests de ruido blanco
test_correlation_residues()   # Autocorrelación de residuos
```

#### Preparación de Variables Exógenas

```python
x_train_exog_custom()         # Preparación de variables exógenas train
x_test_exog()                 # Preparación de variables exógenas test
clean_exogenous_variables()   # Limpieza de multicolinealidad
```

---

## Instalación y Configuración

### Requisitos Previos

- Python 3.9 o superior
- pip (gestor de paquetes)

### Instalación

1. **Clonar el repositorio**

```bash
git clone https://github.com/[usuario]/TFG_ADE.git
cd TFG_ADE/code
```

2. **Crear entorno virtual**

```bash
python -m venv venv
source venv/bin/activate  # En macOS/Linux
# venv\Scripts\activate   # En Windows
```

3. **Instalar dependencias**

```bash
pip install -r requirements.txt
```

4. **Colocar los datos**
   - Crear carpeta `data/` en el directorio raíz
   - Colocar el archivo `Datos_Market_copy.xlsx` en dicha carpeta

---

## Dependencias Principales

### Análisis de Datos

- `pandas==2.2.3`
- `numpy==1.26.4`

### Series Temporales

- `statsmodels==0.14.4`
- `pmdarima==2.0.4`
- `statsforecast==2.0.1`
- `skforecast==0.17.0`

### Machine Learning

- `scikit-learn==1.6.1`
- `catboost==1.2.8`
- `xgboost==2.1.4`

### Visualización

- `matplotlib==3.9.2`
- `seaborn==0.13.2`
- `plotly==6.5.0`

### Interpretabilidad

- `shap==0.47.1`

### Otras Utilidades

- `scipy==1.13.1`
- `patsy==1.0.1`
- `openpyxl==3.1.5`

---

## Ejecución

### Orden Recomendado de Ejecución

1. **Preprocesamiento**: `1_preprocessing_data.ipynb`
2. **Análisis Exploratorio**: `2_descriptive_analysis.ipynb`
3. **Modelos** (pueden ejecutarse independientemente):
   - `3_ARIMA.ipynb`
   - `4_ARIMAX.ipynb`
   - `5_Catboost.ipynb`

### Ejecución de Notebooks

```bash
jupyter notebook
# Navegar a src/ y abrir el notebook deseado
```

---

## Resultados Esperados

El proyecto permite:

1. **Comparación de modelos**: Evaluación del rendimiento de diferentes enfoques (estadísticos vs. ML)
2. **Interpretabilidad**: Análisis SHAP para entender qué factores influyen más en las ventas
3. **Predicciones**: Generación de pronósticos de ventas para horizonte de 6-7 meses
4. **Insights de negocio**: Identificación de patrones por marca, supermercado y variante

---

## Características Técnicas Destacadas

- **Enfoque multi-serie**: Manejo de múltiples series temporales simultáneas
- **División temporal rigurosa**: Train/test basado en fechas (no aleatorio)
- **Transformaciones de varianza**: Box-Cox y logarítmicas para estabilizar series
- **Selección automática de variables**: Backward elimination con criterio estadístico
- **Validación estadística**: Tests de estacionariedad y diagnóstico de residuos
- **Código modular**: Funciones reutilizables en `utilities.py`

---

## Contribuciones

Este proyecto fue desarrollado como parte de un Trabajo de Fin de Grado académico. Para cualquier consulta o sugerencia, por favor abrir un issue en el repositorio.

---

## Licencia

Este proyecto tiene fines académicos y educativos. El uso de los datos está sujeto a las restricciones de confidencialidad aplicables.

---

## Autor

**Oscar Jiménez Bou**  
Grado en Administración y Dirección de Empresas  
Trabajo de Fin de Grado - 2024/2025

---

## Referencias

- Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). _Time series analysis: forecasting and control_. John Wiley & Sons.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. _The American Statistician_, 72(1), 37-45.
- Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. _Advances in neural information processing systems_, 31.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. _Advances in neural information processing systems_, 30.
