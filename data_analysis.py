from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

(model_brand35, data_model) = sa.modelization_with_backward_elimination(
    data[sa.brand35]
)

# breakpoint()

print(model_brand35.summary())
