from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

# data_dummies, model = sa.modelization(data[sa.brand35], fix_significance=True)

# print(model.summary())

# data_dummies, modelo = sa.modelization_with_statsmodels(data[sa.brand35])

# # save in a excel
# data_dummies.to_excel("data/data_dummies.xlsx")
# # modelo.summary()

# print(modelo.summary())

# modelo = sa.modelization_selector(data[sa.brand35])
# data, modelo2 = sa.modelization_with_statsmodels(data[sa.brand35])
modelo3 = sa.modelization_selector2(
    data[sa.brand35], sa.__interactions_delected_brand35__()
)


# print(modelo.summary())
print(modelo3.summary())

"""
                    "pack_size_701___1000_GR:variant_vegan",
                    "value_sales:pack_size_501___700_GR",
                    "value_sales:supermarket_supermarket_D",
                    "supermarket_supermarket_C:supermarket_supermarket_D",
                    "pack_size_501___700_GR:variant_standard",
"""
