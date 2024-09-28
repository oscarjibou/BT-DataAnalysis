import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SalesAnalysis:
    def __init__(self, raw_data):

        # Asignamos los datos originales a un atributo de la clase
        self.raw_data = raw_data

        # Limpiamos los datos usando el método cleaning_data de la clase
        self.data = self.cleaning_data()

        # Definimos las condiciones para los supermercados
        self.supermarketA = self.data["supermarket"] == "supermarket-A"
        self.supermarketB = self.data["supermarket"] == "supermarket-B"
        self.supermarketC = self.data["supermarket"] == "supermarket-C"
        self.supermarketD = self.data["supermarket"] == "supermarket-D"

        # Definimos las condiciones para las variantes
        self.variantF = self.data["variant"] == "flavoured"
        self.variantS = self.data["variant"] == "standard"
        self.variantL = self.data["variant"] == "light"
        self.variantV = self.data["variant"] == "vegan"

        self.pack350 = self.data["pack.size"] == "0 - 350 GR"
        self.pack500 = self.data["pack.size"] == "351 - 500 GR"
        self.pack600 = self.data["pack.size"] == "450 - 600GR"
        self.pack700 = self.data["pack.size"] == "501 - 700 GR"
        self.pack1000 = self.data["pack.size"] == "701 - 1000 GR"

        self.brand35 = self.data["brand"] == "brand-35"
        self.brand14 = self.data["brand"] == "brand-14"
        self.brand15 = self.data["brand"] == "brand-15"
        self.brancOther = self.data["brand"] == "other"

    def cleaning_data(self):

        data = self.raw_data.copy()
        # data.set_index("date", inplace=True)

        for i in data["brand"]:
            if i != "brand-15" and i != "brand-14" and i != "brand-35":
                # change the name
                # test_data["brand"].replace(i, "other", inplace=True)
                data["brand"].replace({i: "other"}, inplace=True)

        return data

    def convert_weeks_to_months(self):

        monthly_data = pd.DataFrame()

        for supermarket in [
            self.supermarketA,
            self.supermarketB,
            self.supermarketC,
            self.supermarketD,
        ]:
            for variant in [self.variantF, self.variantS, self.variantL, self.variantV]:
                for pack_size in [
                    self.pack350,
                    self.pack500,
                    self.pack600,
                    self.pack700,
                    self.pack1000,
                ]:
                    for brand in [
                        self.brand35,
                        self.brand14,
                        self.brand15,
                        self.brancOther,
                    ]:
                        filtered_data = self.data[
                            supermarket & variant & pack_size & brand
                        ]

                        filtered_data = filtered_data.groupby(
                            pd.Grouper(key="date", freq="M")
                        ).sum()

                        filtered_data.reset_index(inplace=True)

                        monthly_data = pd.concat([monthly_data, filtered_data])

        return monthly_data

    # Actualización del código para incluir todos los pack sizes
    def plot_all_separate_flavour(self, brand, sales="volume.sales"):

        fig, ax = plt.subplots(2, 2, figsize=(15, 7), sharex=True)

        # Obtener todos los tamaños de empaque disponibles
        pack_sizes = self.data["pack.size"].unique()

        def plot__(flavour, num1, num2):
            for pack_size in pack_sizes:
                for supermarket, label in zip(
                    [
                        self.supermarketA,
                        self.supermarketB,
                        self.supermarketC,
                        self.supermarketD,
                    ],
                    [
                        "supermercado A",
                        "supermercado B",
                        "supermercado C",
                        "supermercado D",
                    ],
                ):
                    filtered_data = self.data[
                        (self.data["brand"] == brand)
                        & (self.data["pack.size"] == pack_size)
                        & flavour
                        & supermarket
                    ]
                    ax[num1, num2].plot(
                        filtered_data["date"],
                        filtered_data[sales],
                        label=f"{label} - {pack_size}",
                    )

            ax[num1, num2].set_title(
                f"Ventas de {flavour.name} por supermercado y pack.size"
            )
            ax[num1, num2].set_ylabel(sales)
            ax[num1, num2].set_xlabel("Fecha")
            ax[1, 1].legend()
            ax[num1, num2].grid(True)

        # Graficar para cada variante
        plot__(self.variantF, 0, 0)
        plot__(self.variantS, 0, 1)
        plot__(self.variantL, 1, 0)
        plot__(self.variantV, 1, 1)

        # Ajustar el diseño y mostrar la gráfica
        plt.tight_layout()
        plt.show()

    def plot_detail_graph(self, brand, flavour, sales="volume.sales"):

        plt.figure(figsize=(10, 6))

        pack_sizes = self.data["pack.size"].unique()

        for pack_size in pack_sizes:
            for supermarket, label in zip(
                [
                    self.supermarketA,
                    self.supermarketB,
                    self.supermarketC,
                    self.supermarketD,
                ][
                    "supermercado A",
                    "supermercado B",
                    "supermercado C",
                    "supermercado D",
                ]
            ):

                filtered_data = self.data[
                    (self.data["brand"] == brand)
                    & (self.data["pack.size"] == pack_size)
                    & flavour
                    & supermarket
                ]

                plt.plot(filtered_data["date"], filtered_data[sales])
                plt.show()
