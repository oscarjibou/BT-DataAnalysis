import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class SalesAnalysis:
    def __init__(self, raw_data):

        # Assign the original data to a class attribute
        self.raw_data = raw_data

        # CleaningData
        self.data = self.cleaning_data()

        # Define the conditions for the supermarkets
        self.supermarketA = self.data["supermarket"] == "supermarket-A"
        self.supermarketB = self.data["supermarket"] == "supermarket-B"
        self.supermarketC = self.data["supermarket"] == "supermarket-C"
        self.supermarketD = self.data["supermarket"] == "supermarket-D"

        # Define the conditions for the variants
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

        # Convert the weeks into months using the convert_weeks_to_months method
        self.data = self.convert_weeks_to_months()

    def cleaning_data(self):
        """
        Cleans the raw data by replacing certain brand names with 'other'.

        This method performs the following steps:
        1. Creates a copy of the raw data.
        2. Iterates through the 'brand' column of the data.
        3. Replaces any brand name that is not 'brand-15', 'brand-14', or 'brand-35' with 'other'.

        Returns:
            pandas.DataFrame: The cleaned data with specified brand names replaced.
        """

        data = self.raw_data.copy()
        # data.set_index("date", inplace=True)

        for i in data["brand"]:
            if i != "brand-15" and i != "brand-14" and i != "brand-35":
                # change the name
                # test_data["brand"].replace(i, "other", inplace=True)
                data["brand"].replace({i: "other"}, inplace=True)

        return data

    def convert_weeks_to_months(self):
        """
        Converts weekly sales data to monthly aggregated data.

        This method processes the sales data for different supermarkets, variants, pack sizes, and brands,
        aggregating the data on a monthly basis. The aggregation includes summing up the sales volumes, unit sales,
        and value sales, while keeping the first occurrence of supermarket, variant, pack size, and brand for each month.

        Returns:
            pd.DataFrame: A DataFrame containing the monthly aggregated sales data.
        """

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

                        # filtered_data = filtered_data.groupby(
                        #     pd.Grouper(key="date", freq="M")
                        # ).sum()

                        filtered_data = filtered_data.groupby(
                            pd.Grouper(
                                key="date",
                                freq="M",
                            )
                        ).agg(
                            {
                                "volume.sales": "sum",
                                "unit.sales": "sum",
                                "value.sales": "sum",
                                "supermarket": "first",
                                "variant": "first",
                                "pack.size": "first",
                                "brand": "first",
                            }
                        )

                        filtered_data.reset_index(inplace=True)

                        monthly_data = pd.concat([monthly_data, filtered_data])

        return monthly_data

    def plot_all_separate_flavour(self, brand, sales="volume.sales"):
        """
        Plots sales data for different flavours of a given brand across multiple supermarkets and pack sizes.

        Parameters:
        -----------
        brand : str
            The brand for which the sales data is to be plotted.
        sales : str, optional
            The column name in the data representing the sales metric to be plotted (default is "volume.sales").

        Returns:
        --------
        None
            This function does not return any value. It generates and displays a plot.

        Notes:
        ------
        - The function creates a 2x2 subplot grid where each subplot represents sales data for a different flavour variant.
        - The sales data is filtered by brand, pack size, and supermarket before plotting.
        - Each subplot contains sales data for different pack sizes and supermarkets.
        - The function assumes that `self.data` contains the sales data and `self.variantF`, `self.variantS`, `self.variantL`, and `self.variantV` are the flavour variants to be plotted.
        - The function also assumes that `self.supermarketA`, `self.supermarketB`, `self.supermarketC`, and `self.supermarketD` are the supermarket filters.
        """

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
        """
        Plots a detailed graph of sales data for a specific brand and flavour across different supermarkets.

        Parameters:
        brand (str): The brand of the product to filter the data.
        flavour (str): The flavour of the product to filter the data.
        sales (str, optional): The sales metric to plot. Defaults to "volume.sales".

        Returns:
        None
        """

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
