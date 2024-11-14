from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data[sa.brand35]

train_data, test_data = sa.divide_data_for_train_and_test(data=data, train_size=0.8)

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
breakpoint()
##### FORECASTING

# TODO: understang the forecasting output (for now I don't) --> see chat: todo tfg
# Forecasting for the next 12 months
forecasting = model_arima_brand35_selected.forecast(steps=len(test_data))

# get the real values
real_values = test_data.values

# FIXME: doesn't work for now
# rmse = np.sqrt(mean_squared_error(test_data, forecasting))
# mae = mean_absolute_error(test_data, forecasting)

# print(f"RMSE: {rmse}")
# print(f"MAE: {mae}")
