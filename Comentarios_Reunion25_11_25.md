# Pasos

### 1. Procesamos los datos correctamente

### 2. Regresión lineal

---

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>      <td>volume_sales</td>   <th>  R-squared:         </th> <td>   0.833</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.826</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   134.2</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 20 Nov 2025</td> <th>  Prob (F-statistic):</th> <td>4.22e-223</td>
</tr>
<tr>
  <th>Time:</th>                 <td>18:27:48</td>     <th>  Log-Likelihood:    </th> <td> -7343.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   644</td>      <th>  AIC:               </th> <td>1.473e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   620</td>      <th>  BIC:               </th> <td>1.484e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    23</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
                                <td></td>                                   <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>                                                     <td>  1.21e+04</td> <td> 2671.315</td> <td>    4.530</td> <td> 0.000</td> <td> 6855.607</td> <td> 1.73e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-B]</th>                               <td> 2.319e+04</td> <td> 9730.122</td> <td>    2.383</td> <td> 0.017</td> <td> 4078.902</td> <td> 4.23e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-D]</th>                               <td>-1.051e+05</td> <td> 1.42e+04</td> <td>   -7.402</td> <td> 0.000</td> <td>-1.33e+05</td> <td>-7.72e+04</td>
</tr>
<tr>
  <th>C(variant)[T.standard]</th>                                        <td> 2.624e+04</td> <td> 6175.874</td> <td>    4.249</td> <td> 0.000</td> <td> 1.41e+04</td> <td> 3.84e+04</td>
</tr>
<tr>
  <th>C(variant)[T.vegan]</th>                                           <td> 7.521e+04</td> <td> 1.85e+04</td> <td>    4.071</td> <td> 0.000</td> <td> 3.89e+04</td> <td> 1.11e+05</td>
</tr>
<tr>
  <th>C(pack_size)[T.351 - 500 GR]</th>                                  <td>  8.22e+04</td> <td> 5413.805</td> <td>   15.183</td> <td> 0.000</td> <td> 7.16e+04</td> <td> 9.28e+04</td>
</tr>
<tr>
  <th>C(pack_size)[T.501 - 700 GR]</th>                                  <td> 2.179e+04</td> <td> 3537.727</td> <td>    6.160</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>C(pack_size)[T.701 - 1000 GR]</th>                                 <td>-1.993e+04</td> <td> 6573.311</td> <td>   -3.033</td> <td> 0.003</td> <td>-3.28e+04</td> <td>-7025.422</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-B]:C(variant)[T.light]</th>           <td>-1.924e+04</td> <td> 7704.010</td> <td>   -2.497</td> <td> 0.013</td> <td>-3.44e+04</td> <td>-4109.905</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-C]:C(variant)[T.light]</th>           <td> 3.004e+04</td> <td> 6014.788</td> <td>    4.995</td> <td> 0.000</td> <td> 1.82e+04</td> <td> 4.19e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-B]:C(variant)[T.standard]</th>        <td>-5.204e+04</td> <td> 7152.188</td> <td>   -7.276</td> <td> 0.000</td> <td>-6.61e+04</td> <td> -3.8e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-C]:C(variant)[T.standard]</th>        <td>-3.216e+04</td> <td> 5178.112</td> <td>   -6.210</td> <td> 0.000</td> <td>-4.23e+04</td> <td> -2.2e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-D]:C(variant)[T.standard]</th>        <td> 6.913e+04</td> <td> 7708.818</td> <td>    8.968</td> <td> 0.000</td> <td>  5.4e+04</td> <td> 8.43e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-B]:C(variant)[T.vegan]</th>           <td> 7.112e+04</td> <td> 1.43e+04</td> <td>    4.988</td> <td> 0.000</td> <td> 4.31e+04</td> <td> 9.91e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-C]:C(pack_size)[T.351 - 500 GR]</th>  <td>-3.356e+04</td> <td> 5564.661</td> <td>   -6.031</td> <td> 0.000</td> <td>-4.45e+04</td> <td>-2.26e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-D]:C(pack_size)[T.351 - 500 GR]</th>  <td> 3.171e+04</td> <td> 7293.879</td> <td>    4.348</td> <td> 0.000</td> <td> 1.74e+04</td> <td>  4.6e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-D]:C(pack_size)[T.501 - 700 GR]</th>  <td> 2.179e+04</td> <td> 3537.727</td> <td>    6.160</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>C(supermarket)[T.supermarket-B]:C(pack_size)[T.701 - 1000 GR]</th> <td> 5.548e+04</td> <td> 7817.231</td> <td>    7.097</td> <td> 0.000</td> <td> 4.01e+04</td> <td> 7.08e+04</td>
</tr>
<tr>
  <th>C(variant)[T.light]:C(pack_size)[T.351 - 500 GR]</th>              <td> 3.515e+04</td> <td> 9389.189</td> <td>    3.744</td> <td> 0.000</td> <td> 1.67e+04</td> <td> 5.36e+04</td>
</tr>
<tr>
  <th>C(variant)[T.vegan]:C(pack_size)[T.351 - 500 GR]</th>              <td> 9.395e-11</td> <td> 1.29e-11</td> <td>    7.287</td> <td> 0.000</td> <td> 6.86e-11</td> <td> 1.19e-10</td>
</tr>
<tr>
  <th>C(variant)[T.light]:C(pack_size)[T.501 - 700 GR]</th>              <td> 2.179e+04</td> <td> 3537.727</td> <td>    6.160</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 2.87e+04</td>
</tr>
<tr>
  <th>C(variant)[T.light]:C(pack_size)[T.701 - 1000 GR]</th>             <td> 2.423e+04</td> <td> 6821.642</td> <td>    3.552</td> <td> 0.000</td> <td> 1.08e+04</td> <td> 3.76e+04</td>
</tr>
<tr>
  <th>C(variant)[T.standard]:C(pack_size)[T.701 - 1000 GR]</th>          <td>-4.416e+04</td> <td> 7450.494</td> <td>   -5.928</td> <td> 0.000</td> <td>-5.88e+04</td> <td>-2.95e+04</td>
</tr>
<tr>
  <th>price:C(supermarket)[T.supermarket-B]</th>                         <td>-3.204e+04</td> <td> 8711.473</td> <td>   -3.677</td> <td> 0.000</td> <td>-4.91e+04</td> <td>-1.49e+04</td>
</tr>
<tr>
  <th>price:C(supermarket)[T.supermarket-D]</th>                         <td> 1.217e+05</td> <td> 1.45e+04</td> <td>    8.375</td> <td> 0.000</td> <td> 9.32e+04</td> <td>  1.5e+05</td>
</tr>
<tr>
  <th>price:C(variant)[T.light]</th>                                     <td>-6.512e+04</td> <td> 9743.408</td> <td>   -6.684</td> <td> 0.000</td> <td>-8.43e+04</td> <td> -4.6e+04</td>
</tr>
<tr>
  <th>price:C(variant)[T.vegan]</th>                                     <td> -8.96e+04</td> <td>    2e+04</td> <td>   -4.477</td> <td> 0.000</td> <td>-1.29e+05</td> <td>-5.03e+04</td>
</tr>
<tr>
  <th>price:C(pack_size)[T.701 - 1000 GR]</th>                           <td> 4.872e+04</td> <td> 8836.075</td> <td>    5.513</td> <td> 0.000</td> <td> 3.14e+04</td> <td> 6.61e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>108.327</td> <th>  Durbin-Watson:     </th> <td>   1.898</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1530.123</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.176</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td>10.543</td>  <th>  Cond. No.          </th> <td>1.47e+16</td>
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 6.43e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.

---

### 3. Convertimos interacciones significativas del modelo de regresión en variables exógenas

> **ℹ️ Importante:**  
> Recuerda: coincidiendo interacciones significativas = variables exógenas que meteremos al modelo

### 4. Lanzamos modelo auto_arima y nos devuelve:

---

- ARIMA(0,1,0)(0,1,0)[12] : AIC=18700.911, Time=0.38 sec
- ARIMA(0,1,0)(0,1,1)[12] : AIC=inf, Time=23.23 sec
- ARIMA(0,1,0)(1,1,0)[12] : AIC=15113.200, Time=10.78 sec
- ARIMA(0,1,0)(1,1,1)[12] : AIC=inf, Time=28.88 sec
- ARIMA(0,1,1)(0,1,0)[12] : AIC=inf, Time=1.92 sec
- ARIMA(0,1,1)(0,1,1)[12] : AIC=inf, Time=96.38 sec
- ARIMA(0,1,1)(1,1,0)[12] : AIC=inf, Time=88.04 sec
- ARIMA(0,1,1)(1,1,1)[12] : AIC=inf, Time=57.00 sec
- ARIMA(0,1,2)(0,1,0)[12] : AIC=inf, Time=6.35 sec
- ARIMA(0,1,2)(0,1,1)[12] : AIC=inf, Time=78.86 sec
- ARIMA(0,1,2)(1,1,0)[12] : AIC=inf, Time=58.06 sec
- ARIMA(0,1,2)(1,1,1)[12] : AIC=inf, Time=57.61 sec
- ARIMA(0,1,3)(0,1,0)[12] : AIC=inf, Time=12.50 sec
- ARIMA(0,1,3)(0,1,1)[12] : AIC=inf, Time=56.99 sec
- ARIMA(0,1,3)(1,1,0)[12] : AIC=inf, Time=82.80 sec
- ARIMA(0,1,3)(1,1,1)[12] : AIC=inf, Time=70.21 sec
- ARIMA(1,1,0)(0,1,0)[12] : AIC=15113.277, Time=0.36 sec
- ARIMA(1,1,0)(0,1,1)[12] : AIC=inf, Time=29.57 sec
- ARIMA(1,1,0)(1,1,0)[12] : AIC=14983.257, Time=16.95 sec
- ARIMA(1,1,0)(1,1,1)[12] : AIC=inf, Time=34.50 sec
- ARIMA(1,1,1)(0,1,0)[12] : AIC=inf, Time=3.38 sec
- ARIMA(1,1,1)(0,1,1)[12] : AIC=inf, Time=51.97 sec
- ARIMA(1,1,1)(1,1,0)[12] : AIC=inf, Time=93.29 sec
- ARIMA(1,1,1)(1,1,1)[12] : AIC=inf, Time=50.14 sec
- ARIMA(1,1,2)(0,1,0)[12] : AIC=inf, Time=11.54 sec
- ARIMA(1,1,2)(0,1,1)[12] : AIC=inf, Time=138.71 sec
- ARIMA(1,1,2)(1,1,0)[12] : AIC=inf, Time=111.73 sec
- ARIMA(1,1,2)(1,1,1)[12] : AIC=inf, Time=86.58 sec
- ARIMA(1,1,3)(0,1,0)[12] : AIC=inf, Time=15.24 sec
- ARIMA(1,1,3)(0,1,1)[12] : AIC=inf, Time=101.63 sec
- ARIMA(1,1,3)(1,1,0)[12] : AIC=inf, Time=113.73 sec
- ARIMA(2,1,0)(0,1,0)[12] : AIC=15067.640, Time=0.96 sec
- ARIMA(2,1,0)(0,1,1)[12] : AIC=inf, Time=26.75 sec
- ARIMA(2,1,0)(1,1,0)[12] : AIC=14940.126, Time=18.18 sec
- ARIMA(2,1,0)(1,1,1)[12] : AIC=inf, Time=35.02 sec
- ARIMA(2,1,1)(0,1,0)[12] : AIC=inf, Time=6.97 sec
- ARIMA(2,1,1)(0,1,1)[12] : AIC=inf, Time=51.02 sec
- ARIMA(2,1,1)(1,1,0)[12] : AIC=inf, Time=78.25 sec
- ARIMA(2,1,1)(1,1,1)[12] : AIC=inf, Time=62.28 sec
- ARIMA(2,1,2)(0,1,0)[12] : AIC=inf, Time=7.46 sec
- ARIMA(2,1,2)(0,1,1)[12] : AIC=inf, Time=91.79 sec
- ARIMA(2,1,2)(1,1,0)[12] : AIC=inf, Time=108.89 sec
- ARIMA(2,1,3)(0,1,0)[12] : AIC=inf, Time=20.21 sec
- ARIMA(3,1,0)(0,1,0)[12] : AIC=15033.538, Time=2.24 sec
- ARIMA(3,1,0)(0,1,1)[12] : AIC=inf, Time=28.82 sec
- ARIMA(3,1,0)(1,1,0)[12] : AIC=14902.971, Time=21.48 sec
- ARIMA(3,1,0)(1,1,1)[12] : AIC=inf, Time=39.66 sec
- ARIMA(3,1,1)(0,1,0)[12] : AIC=inf, Time=7.83 sec
- ARIMA(3,1,1)(0,1,1)[12] : AIC=inf, Time=73.91 sec
- ARIMA(3,1,1)(1,1,0)[12] : AIC=inf, Time=59.58 sec
- ARIMA(3,1,2)(0,1,0)[12] : AIC=inf, Time=10.01 sec

---

![Resultados Auto ARIMA](images_markdown/image.png)

---

> **ℹ️ Importante:**  
> ‼️ Nos fijamos que muchos modelos no convergen.

### 5. Chequeamos residuos del modelo anterior

**Residues Analysis (White Noise)**

- [Heteroscedasticity Test] ARCH p-value:

  - 4.80189902198319e-11 -- range(> 0.05)

- [Normality Test] Jarque-Bera p-value:
  - 4**.601860723233844e-50 -- range(> 0.05)**
- [Normality Test] Shapiro-Wilk p-value:
  - 7.320723375739127e-12 -- range(> 0.05)
- [Autocorrelation Test] Ljung-Box p-value: lb_stat | lb_pvalue
  - 10 102.058246 2.109089e-17 -- range(> 0.05)
- [Autocorrelation Test first order] Durbin-Watson statistic:
  - 2.1436117794872604 -- range(2.0)

**Residues Analysis (Stationarity)**

Estadístico ADF: -7.853404794535187
Valor p: 5.5237520752949304e-12 -- es estacionaria si p < 0.05
Valores críticos:
1%: -3.440890045708521
5%: -2.8661904001753618
10%: -2.569246579178572

> **ℹ️ Importante:**  
> ‼️ Salen muy mal. Y no nos sale ruido blanco

---

## Mejora del modelo

## Hacemos un análisis y concluimos que:

---

SOLUCIÓN: LIMPIEZA DE VARIABLES EXÓGENAS

1. ELIMINANDO VARIABLES CONSTANTES:
   Eliminando 1 variables constantes:

   - C(variant)[T.vegan]:C(pack_size)[T.351 - 500 GR]

2. ELIMINANDO VARIABLES CON CORRELACIÓN PERFECTA:
   Eliminando 2 variables con correlación perfecta:

   - C(variant)[T.light]:C(pack_size)[T.501 - 700 GR]
   - C(supermarket)[T.supermarket-D]:C(pack_size)[T.501 - 700 GR]

3. ELIMINANDO VARIABLES ALTAMENTE CORRELACIONADAS (|r| > 0.95):
   Eliminando 3 variables altamente correlacionadas:

   - price:C(supermarket)[T.supermarket-D]
   - price:C(variant)[T.vegan]
   - price:C(pack_size)[T.701 - 1000 GR]

4. VERIFICANDO NÚMERO DE CONDICIÓN DESPUÉS DE LA LIMPIEZA:
   Número de condición original: 7.48e+17
   Número de condición después de limpieza: 2.89e+15
   ⚠️ AÚN HAY MULTICOLINEALIDAD EXTREMA
   💡 Considera eliminar más variables o usar regularización

5. RESUMEN:
   - Variables originales: 27
   - Variables después de limpieza: 21
   - Variables eliminadas: 6
   - Reducción: 22.2%

### 6. Volvemos a hacer el modelo auto_arima

<img src="images_markdown/image2.png" alt="auto_arima_clean" style="width:70%; height:auto;"/>

### 7.Residuos del modelo nuevo

**Residues Analysis (White Noise)**

- [Heteroscedasticity Test] ARCH p-value:
  - 0.0003964551491381505 -- range(> 0.05)
- [Normality Test] Jarque-Bera p-value:
  - 2.0787060633904786e-211 -- range(> 0.05)
- [Normality Test] Shapiro-Wilk p-value:
  - 2.1739187861493027e-17 -- range(> 0.05)
- [Autocorrelation Test] Ljung-Box p-value:
  - 0.013733 -- range(> 0.05)
- [Autocorrelation Test first order] Durbin-Watson statistic:
  - 1.9183124656919244 -- range(2.0)

---

#### 8. Gráficos residuos:

![Grafico residuos auto_arima_clean](images_markdown/image3.png)

### 9. Predicciones

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s1.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s2.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s3.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/s4.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s5.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s6.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s7.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/s8.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s9.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s10.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s11.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/s12.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s13.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s14.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s15.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/s16.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s17.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s18.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/s19.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/s20.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/s21.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
</div>

---

### 10. Probamos haciendo transformación variable objetivo

// Analizamos nuevos resultados

### 11. Lanzamos modelo auto_arima y nos devuelve:

- **Performing stepwise search to minimize aic**
- ARIMA(0,0,0)(0,1,0)[12] intercept : AIC=3077.780
- ARIMA(1,0,0)(1,1,0)[12] intercept : AIC=2876.553
- ARIMA(0,0,1)(0,1,1)[12] intercept : AIC=inf
- ARIMA(0,0,0)(0,1,0)[12] : AIC=3075.913
- ARIMA(1,0,0)(0,1,0)[12] intercept : AIC=3079.670
- ARIMA(1,0,0)(1,1,1)[12] intercept : AIC=inf
- ARIMA(1,0,0)(0,1,1)[12] intercept : AIC=inf
- ARIMA(0,0,0)(1,1,0)[12] intercept : AIC=2875.121
- ARIMA(0,0,0)(1,1,1)[12] intercept : AIC=inf
- ARIMA(0,0,0)(0,1,1)[12] intercept : AIC=inf
- ARIMA(0,0,1)(1,1,0)[12] intercept : AIC=2876.552
- ARIMA(1,0,1)(1,1,0)[12] intercept : AIC=2878.554
- ARIMA(0,0,0)(1,1,0)[12] : AIC=2873.403
- ARIMA(0,0,0)(1,1,1)[12] : AIC=inf
- ARIMA(0,0,0)(0,1,1)[12] : AIC=inf
- ARIMA(1,0,0)(1,1,0)[12] : AIC=2874.816
- ARIMA(0,0,1)(1,1,0)[12] : AIC=2874.814
- ARIMA(1,0,1)(1,1,0)[12] : AIC=2876.816
- **Best model:** ARIMA(0,0,0)(1,1,0)[12]
- **Total fit time:** 2871.337 seconds

### 12. Chequeamos residuos del modelo anterior

**Residues Analysis (White Noise)**

- [Heteroscedasticity Test] ARCH p-value:

  - 0.01174091289335583 -- range(> 0.05)

- [Normality Test] Jarque-Bera p-value:
  - 4.829608521628615e-120 -- range(> 0.05)
- [Normality Test] Shapiro-Wilk p-value:
  - 1.8622387300781965e-19 -- range(> 0.05)
- [Autocorrelation Test] Ljung-Box p-value: lb_stat | lb_pvalue
  - 10 117.992236 1.292208e-20 -- range(> 0.05)
- [Autocorrelation Test first order] Durbin-Watson statistic:
  - 1.5081114222820573 -- range(2.0)

![auto_arima_clean](images_markdown/image4.png)

### 13. Predicciones del modelo con la transformación logarítmica.

> **ℹ️ Importante:**  
> ‼️ He realizado la destransformación antes de representar las predicciones. Es decir:
>
> - obtenemos las predicciones en escala logaritmica
> - revertimos la transformación
> - guardamos en un dataframe, que luego representamos en las gráficas.

<div style="page-break-after: always;"></div>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l1.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l2.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l3.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/l4.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l5.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l6.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l7.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/l8.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

<div style="page-break-after: always;"></div>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l9.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l10.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l11.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/l12.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l13.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l14.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l15.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/l16.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

<div style="page-break-after: always;"></div>

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l17.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l18.png" alt="Predicción 2" style="width: 48%; height: auto;"/>
  <img src="images_markdown/l19.png" alt="Predicción 3" style="width: 48%; height: auto; margin-top: 16px;"/>
  <img src="images_markdown/l20.png" alt="Predicción 4" style="width: 48%; height: auto; margin-top: 16px;"/>
</div>

---

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 16px;">
  <img src="images_markdown/l21.png" alt="Predicción 1" style="width: 48%; height: auto;"/>
</div>
