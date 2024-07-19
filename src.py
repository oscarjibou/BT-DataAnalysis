import pandas as pd
import matplotlib.pyplot as plt

# load raw_data
raw_data = pd.read_csv("Tractores_Matriculados.csv")
raw_data.head()

# Filter data by year
start_year = "1985"
end_year = "1996"

data = raw_data[(raw_data["Fecha"] >= start_year) & (raw_data["Fecha"] <= end_year)]

data.Tractores.plot(
    x="Fecha", y="Tractores", kind="line", title="Tractores matriculados por año"
)
plt.show()
