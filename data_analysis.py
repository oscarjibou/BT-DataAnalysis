from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

model_brand35 = sa.modelization(
    data[sa.brand35], sa.__interactions_delected_brand35__()
)

possible_models = {
    "1": (0, 1, 5),
    "2": (0, 1, 4),
    "3": (0, 1, 3),
    "4": (0, 1, 2),
    "5": (0, 1, 1),
    "6": (4, 1, 0),
    "7": (3, 1, 0),
    "8": (2, 1, 0),
    "9": (1, 1, 0),
    "10": (1, 1, 1),
}

# model_auto_arima = sa.autoArima(data["volume.sales"], data["value.sales"])
# print(model_auto_arima.summary())
