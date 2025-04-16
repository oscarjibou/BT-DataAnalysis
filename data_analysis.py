from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data[sa.brand35]

train_data, test_data = sa.divide_data_for_train_and_test(data=data, train_size=0.8)

sa.excel(train_data, path="data/train_data.xlsx")
sa.excel(test_data, path="data/test_data.xlsx")

model_brand35 = sa.modelization_with_backward_elimination(
    data_filtered_by_brand=train_data
)

possible_models = {
    "1": (0, 0, 5),
    "2": (5, 0, 0),
    "3": (5, 0, 5),
}

model_arima_brand35_selected = sa.ARIMA(
    residues=model_brand35.resid,
    model_chosen=(possible_models["1"]),
    diff_need_for_residues=False,
)

# --------------------------------------
# AutoArima and Forecasting
# --------------------------------------


# TODO: understang the forecasting output (for now I don't) --> see chat: todo tfg
# Forecasting for residuals
forecasting = model_arima_brand35_selected.forecast(steps=len(test_data))

breakpoint()

# Forecasting for volume.sales
# test_data_volum = test_data["volume.sales"]

futures_data = model_brand35.predict(test_data)

# FIXME:
"""
Expect more variables.

-  parece que el modelo espera 29 variables (como indica el tamaño de params), pero estás proporcionando solo 9 variables en test_data.

variables used in prediction are: model_brand35.model.exog_names (are 29)
"""


breakpoint()
futures_data = model_arima_brand35_selected.forecast(steps=len(test_data_volum))

breakpoint()

final_forest = forecasting + futures_data

breakpoint()

# TODO: uso this function to calculate the metrics for the study
# rmse = np.sqrt(mean_squared_error(test_data, forecasting))
# mae = mean_absolute_error(test_data, forecasting)

# print(f"RMSE: {rmse}")
# print(f"MAE: {mae}")
##
