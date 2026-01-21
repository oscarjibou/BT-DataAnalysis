# from utilities import *

# raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

# sa = SalesAnalysis(raw_data)

# data = sa.data[sa.brand35]

# train_data, test_data = sa.divide_data_for_train_and_test(data=data, train_size=0.8)

# sa.excel(train_data, path="data/train_data.xlsx")
# sa.excel(test_data, path="data/test_data.xlsx")

# model_brand35 = sa.modelization_with_backward_elimination(
#     data_filtered_by_brand=train_data
# )


# --------------------------------------
# AutoArima and Forecasting
# --------------------------------------
import warnings
from urllib3.exceptions import NotOpenSSLWarning

# 1) Opcional: silenciar solo ese aviso
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)

# 2) Librerías
import numpy as np
from pmdarima.arima import auto_arima


# 3) Datos ficticios
np.random.seed(42)
y = np.random.randn(120).cumsum()

# 4) Ajuste
model = auto_arima(
    y,
    seasonal=False,  # datos simulados sin estacionalidad
    stepwise=True,
    suppress_warnings=True,
)

# 5) Resultados
print(model.summary())  # tabla con los parámetros
print("\nPronóstico próximo mes:")
print(model.predict(n_periods=4))
