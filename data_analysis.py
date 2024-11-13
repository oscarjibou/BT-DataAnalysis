from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data


# #########################################
# data_brand35 = data[sa.brand35]

# correlation_matrix = data_brand35.corr().abs()

# upper_triangle = correlation_matrix.where(
#     np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
# )
# high_correlation = [
#     column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)
# ]

# print(f"Variables con alta correlación a eliminar: {high_correlation}")

# # Paso 3: Eliminar las variables con alta correlación del DataFrame
# data_reduced = data.drop(columns=high_correlation)


# # Paso 4: Crear un nuevo modelo con el conjunto reducido de variables
# model_reduced = smf.ols(formula=formula, data=data_reduced).fit()

# # Paso 5: Evaluar el nuevo modelo
# print(model_reduced.summary())


# breakpoint()
#########################################

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


###################################

print(model_brand35.summary())
breakpoint()

model_initial = model_brand35  # Ya hemos ajustado este modelo
df_dummies = (
    model_initial.model.exog
)  # Obtiene las variables independientes (matriz X del modelo)
df_dummies_df = pd.DataFrame(df_dummies, columns=model_initial.model.exog_names)

# Paso 2: Calcular la matriz de correlación y encontrar variables altamente correlacionadas
correlation_matrix = df_dummies_df.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)
high_correlation = [
    column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)
]

# Imprimir las variables altamente correlacionadas identificadas para análisis
print(high_correlation)
###################################
