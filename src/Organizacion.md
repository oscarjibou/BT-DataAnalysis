#### abstract

El presente trabajo responde a una necesidad planteada por una consultora india: crear un sistema de pronóstico de ventas
de condimentos alimenticios en cadenas de supermercados. La base de datos utilizada contiene registros mensuales de las principales
empresas proveedoras de condimentos durante un período de tres años, incluyendo variables como fecha, sabor, tamaño del envase, marca,
cadena de supermercado, precio, unidades vendidas y volumen de venta. Comprender los factores que influyen en el volumen de ventas es
fundamental para evaluar el impacto de las variaciones de precio. Asimismo, el análisis de los patrones temporales mediante modelos
predictivos resulta esencial para realizar pronósticos futuros. El estudio comprende una fase de preprocesamiento y limpieza de datos,
seguida por la implementación de modelos de series temporales donde se usarán ARIMA, ARIMAX o Prophet. Estos modelos se seleccionan según
su precisión predictiva y calidad de ajuste, identificando las variables más influyentes en cada caso. El análisis resultante ofrece
herramientas valiosas para la toma de decisiones y la predicción.

✅ Preprocesamiento de los datos
✅ Análisis descriptivo
✅ ARIMA
✅ ARIMAX
❌ Prophet

## 1. Preprocesamiento de los datos

```python

    cleaning_data(): agrupamos los datos en 4 brands

    convert_weeks_to_months()

    order_dataset_by_date()

    add_price_column()

    data.dropna()  # drop the rows with NaN values
```

> En los cálculos de algunos scripts hemos cogido solo las principales brands para los scripts. Que son las que cogen la mayor número de datos

```python

    Transformación BOX-COX

```

## 2. Análisis descriptivo

FUNCIONES:

- detail_plot
- separate_plot_by_flavour
- plot_everything_in_4_plots
- plot_everything

‼️ Añadir más funciones para explicar mejor el análisis descriptivo

## 3. ARIMA

/Users/oscarjimenezbou/Documents/TFG_ADE/code/other/analisys.ipynb
/Users/oscarjimenezbou/Documents/TFG_ADE/code/other/data_analysis.ipynb

## 4. ARIMAX

## 5. Prophet
