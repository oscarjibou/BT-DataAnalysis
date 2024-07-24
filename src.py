import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


file = "Tractores_Matriculados.csv"

# load raw_data
raw_data = pd.read_excel("Datos_Market_copy.xlsx")
print(raw_data.head())

# print(raw_data.dtypes)

# selected the first 100 rows
selected_data = raw_data.head(100)

plt.figure(figsize=(12, 6))
plt.plot(
    selected_data["value.sales"],
    selected_data["unit.sales"],
    marker="o",
    linestyle="--",
)
plt.xlabel("Price")
plt.ylabel("Units")
plt.title("Price vs Units")
plt.grid(True)
plt.show()
