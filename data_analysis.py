from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

# data_dummies, model = sa.modelization(data[sa.brand35], fix_significance=True)

# print(model.summary())

data_dummies, model = sa.modelization_with_statsmodels(data[sa.brand35])
print(model.summary())
