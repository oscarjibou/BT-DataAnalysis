import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from composition_stats import ilr, ilr_inv, closure
from statsmodels.tsa.seasonal import seasonal_decompose

raw_data = pd.read_excel("Datos_Market_copy.xlsx")

data = raw_data.copy()
data.set_index("date", inplace=True)

# Unify the brand names to 4 categories (brand-15, brand-14, brand-35, other)
for i in data["brand"]:
    if i != "brand-15" and i != "brand-14" and i != "brand-35":
        # change the name
        # test_data["brand"].replace(i, "other", inplace=True)
        data["brand"].replace({i: "other"}, inplace=True)


sales = ["volume.sales", "value.sales", "unit.sales"]

pack_size350 = data["pack.size"].isin(["0 - 350 GR"])
pack_size500 = data["pack.size"].isin(["351 - 500 GR"])
pack_size1000 = data["pack.size"].isin(["701 - 1000 GR"])
pack_size = data["pack.size"].isin(["0 - 350 GR", "351 - 500 GR"])

brand35 = data["brand"].isin(["brand-35"])
brand15 = data["brand"].isin(["brand-15"])
brand14 = data["brand"].isin(["brand-14"])
brandOther = data["brand"].isin(["other"])

variantF = data["variant"].isin(["flavoured"])
variantL = data["variant"].isin(["light"])
variantS = data["variant"].isin(["standard"])
variantV = data["variant"].isin(["vegan"])

supermarketA = data["supermarket"].isin(["supermarket-A"])
supermarketB = data["supermarket"].isin(["supermarket-B"])
supermarketC = data["supermarket"].isin(["supermarket-C"])
supermarketD = data["supermarket"].isin(["supermarket-D"])


def plot_data(brand, supermarket, pack_size, sales):
    fig, ax = plt.subplots(2, 2, figsize=(15, 7))

    ax[0, 0].plot(data[(brand) & (variantF) & (supermarket) & (pack_size)][sales])
    ax[0, 0].set_title("Flavoured")
    ax[0, 0].set_ylabel(sales)
    ax[0, 0].grid(True)

    ax[0, 1].plot(data[(brand) & (variantS) & (supermarket) & (pack_size)][sales])
    ax[0, 1].set_title("Standard")
    ax[0, 1].set_ylabel(sales)
    ax[0, 1].grid(True)

    ax[1, 0].plot(data[(brand) & (variantL) & (supermarket) & (pack_size)][sales])
    ax[1, 0].set_title("Light")
    ax[1, 0].set_ylabel(sales)
    ax[1, 0].grid(True)

    ax[1, 1].plot(data[(brand) & (variantV) & (supermarket) & (pack_size)][sales])
    ax[1, 1].set_title("Vegan")
    ax[1, 1].set_ylabel(sales)
    ax[1, 1].grid(True)

    plt.suptitle(f"{data[brand]['brand'].unique()} - {sales}")
    plt.tight_layout()


plot_data(brand35, pack_size350, supermarketA, "volume.sales")
plot_data(brand35, pack_size350, supermarketB, "volume.sales")
plot_data(brand35, pack_size350, supermarketC, "volume.sales")
plot_data(brand35, pack_size350, supermarketD, "volume.sales")

plt.show()
