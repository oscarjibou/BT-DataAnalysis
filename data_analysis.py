from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

modelo_brand35 = sa.modelization(
    data[sa.brand35], sa.__interactions_delected_brand35__()
)

print(modelo_brand35.summary())
