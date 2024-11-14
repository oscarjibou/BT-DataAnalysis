from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

model_brand35 = sa.modelization_with_backward_elimination(data[sa.brand35])

possible_models = {
    "1": (0, 0, 5),
    "2": (5, 0, 0),
    "3": (5, 0, 5),
}

for modelo in possible_models:
    model_brand35_arima = sa.ARIMA(
        residues=model_brand35.resid,
        model_chosen=(possible_models[modelo]),
        diff_need_for_residues=False,
    )

    print("################### Residues Analysis (White Noise) ####################")
    sa.residual_white_noise_test(model_brand35_arima.resid)
    print("#######################################################################")
