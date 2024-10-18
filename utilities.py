import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import PolynomialFeatures


class SalesAnalysis:
    def __init__(self, raw_data: pd.DataFrame):

        # Assign the original data to a class attribute
        self.raw_data = raw_data

        # CleaningData
        self.data = self.cleaning_data()

        self.__variables__()

        # Convert the weeks into months using the convert_weeks_to_months method
        self.data = self.convert_weeks_to_months()

        self.__variables__()  # update the variables after the conversion

        # # Convert the 'date' column to datetime format wihout the time only the date
        self.data["date"] = self.data["date"].dt.date

    def __variables__(self):
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
        self.brandOther = self.data["brand"] == "other"

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
                        self.brandOther,
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

    def detail_plot(
        self,
        brand: pd.Series,
        supermarket: pd.Series,
        pack_size: pd.Series,
        variant: pd.Series,
        sales: str = "volume.sales",
        plot: bool = True,
    ) -> None:

        filtered_data = self.data[brand & supermarket & pack_size & variant]

        plt.figure(figsize=(10, 6))
        plt.plot(filtered_data["date"], filtered_data[sales])
        plt.xlabel("Date")
        plt.ylabel(sales)
        plt.grid(True)
        if plot:
            plt.show()

    def separate_plot_by_flavour(
        self, brand: pd.Series, pack_size: pd.Series, sales: str = "volume.sales"
    ) -> None:
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
                filtered_data = self.data[brand & pack_size & flavour & supermarket]
                ax[num1, num2].plot(
                    filtered_data["date"],
                    filtered_data[sales],
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

    def plot_everything_in_4_plots(
        self,
        brand: pd.Series,
        sales: str = "volume.sales",
        keepAxisX: bool = True,
        keepAxisY: bool = False,
    ) -> None:
        """
        Plots sales data for different product variants across multiple supermarkets and package sizes in a 2x2 grid of subplots.

        Parameters:
        -----------
        self : object
            The instance of the class containing the data and attributes required for plotting.
        brand : pandas.Series
            A boolean series used to filter the data for the specified brand.
        sales : str, optional
            The column name in the data representing sales figures to be plotted. Default is "volume.sales".
        keepAxisX : bool, optional
            If True, the x-axis will be shared among subplots. Default is True.
        keepAxisY : bool, optional
            If True, the y-axis will be shared among subplots. Default is False.

        Returns:
        --------
        None
            This function does not return any value. It generates and displays a plot.

        Notes:
        ------
        - The function iterates over predefined supermarkets and package sizes to filter and plot the data.
        - Each subplot represents sales data for a specific product variant (e.g., Flavoured, Standard, Light, Vegan).
        - The function ensures that only non-empty data with non-zero sales are plotted.
        - Legends, titles, and grid lines are added to each subplot for better readability.
        - The layout of the plots is adjusted using `plt.tight_layout()` for a cleaner appearance.
        """

        fig, ax = plt.subplots(
            2, 2, figsize=(15, 10), sharex=keepAxisX, sharey=keepAxisY
        )

        def plot_variant(ax, variant, title, row, col):
            # Iterar por cada supermercado
            for supermarket, supermarket_label in zip(
                [
                    self.supermarketA,
                    self.supermarketB,
                    self.supermarketC,
                    self.supermarketD,
                ],
                ["Supermarket A", "Supermarket B", "Supermarket C", "Supermarket D"],
            ):
                # Iterar por cada tamaño de paquete
                for pack_size, pack_label in zip(
                    [
                        self.pack350,
                        self.pack500,
                        self.pack600,
                        self.pack700,
                        self.pack1000,
                    ],
                    [
                        "0 - 350 GR",
                        "351 - 500 GR",
                        "450 - 600 GR",
                        "501 - 700 GR",
                        "701 - 1000 GR",
                    ],
                ):
                    filtered_data = self.data[brand & variant & supermarket & pack_size]
                    # ax[row, col].plot(
                    #     filtered_data["date"],
                    #     filtered_data[sales],
                    #     label=f"{supermarket_label} - {pack_label}",
                    # )

                    # Verificar si los datos no están vacíos ni tienen solo ceros en las ventas
                    if not filtered_data.empty and filtered_data[sales].sum() > 0:
                        ax[row, col].plot(
                            filtered_data["date"],
                            filtered_data[sales],
                            label=f"{supermarket_label} - {pack_label}",
                        )

            ax[row, col].set_title(f"{title} - {sales}")
            ax[row, col].set_ylabel(sales)
            ax[row, col].legend(loc="upper left", fontsize="small")
            ax[row, col].grid(True)

        # Graficar para cada variante (sabor)
        plot_variant(ax, self.variantF, "Flavoured", 0, 0)
        plot_variant(ax, self.variantS, "Standard", 0, 1)
        plot_variant(ax, self.variantL, "Light", 1, 0)
        plot_variant(ax, self.variantV, "Vegan", 1, 1)

        # Ajustar el diseño
        plt.tight_layout()
        plt.show()

    def plot_everything(
        self,
        brand: pd.Series,
        variant: pd.Series,
        sales: str = "volume.sales",
        title: str = "sales",
        plot: bool = True,
    ) -> None:
        """
        Plots sales data for different supermarkets and package sizes.

        Parameters:
        self : object
            The instance of the class containing the data and attributes.
        brand : pd.Series
            A boolean series to filter the data by brand.
        variant : pd.Series
            A boolean series to filter the data by variant.
        sales : str, optional
            The column name of the sales data to plot (default is "volume.sales").
        title : str, optional
            The title of the plot (default is "sales").
        plot : bool, optional
            If True, the plot will be displayed (default is True).

        Returns:
        None
        """
        plt.figure(figsize=(15, 10))

        for supermarket, supermarket_label in zip(
            [
                self.supermarketA,
                self.supermarketB,
                self.supermarketC,
                self.supermarketD,
            ],
            ["Supermarket A", "Supermarket B", "Supermarket C", "Supermarket D"],
        ):
            # Iterar por cada tamaño de paquete
            for pack_size, pack_label in zip(
                [
                    self.pack350,
                    self.pack500,
                    self.pack600,
                    self.pack700,
                    self.pack1000,
                ],
                [
                    "0 - 350 GR",
                    "351 - 500 GR",
                    "450 - 600 GR",
                    "501 - 700 GR",
                    "701 - 1000 GR",
                ],
            ):
                filtered_data = self.data[brand & variant & supermarket & pack_size]

                if not filtered_data.empty and filtered_data[sales].sum() > 0:
                    plt.plot(
                        filtered_data["date"],
                        filtered_data[sales],
                        label=f"{supermarket_label} - {pack_label}",
                    )
                plt.title(f"{title} - {sales}")
                plt.xlabel("date")
                plt.ylabel(sales)
                plt.legend()
                plt.grid(True)

        if plot:

            plt.show()

    def modelization(
        self,
        data_filtered_by_brand: pd.DataFrame,
        fix_significance: bool = False,
        interactions: int = 1,
    ) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:

        # --> Añadirle la libreria (from sklearn.preprocessing import PolynomialFeatures) para hacer las interraciones entre las variables y poder hacer el modelo polinomico
        """
        example: data_dummies, model = sa.modelization(data[sa.brand35])

        This function creates dummy variables for the 'supermarket', 'variant', and 'pack.size' columns,
        converts specific columns to integer type, and then fits an Ordinary Least Squares (OLS) regression model
        using the specified features to predict 'volume.sales'.

        Parameters:
        -----------
        data_filtered_by_brand : pandas.DataFrame
            The input DataFrame containing the filtered data by brand.

        Returns:
        --------
        data_dummies : pandas.DataFrame
            The DataFrame with dummy variables created and specific columns converted to integer type.
        model : statsmodels.regression.linear_model.RegressionResultsWrapper
            The fitted OLS regression model.
        """

        # Create dummy variables for the supermarket column
        data_dummies = pd.get_dummies(
            data_filtered_by_brand,
            columns=["supermarket", "variant", "pack.size"],
            drop_first=True,
        )

        for col in [
            "supermarket_supermarket-B",
            "supermarket_supermarket-C",
            "supermarket_supermarket-D",
            "variant_light",
            "variant_standard",
            "variant_vegan",
            "pack.size_351 - 500 GR",
            "pack.size_501 - 700 GR",
            "pack.size_701 - 1000 GR",
        ]:
            data_dummies[col] = data_dummies[col].astype(int)

        X = data_dummies[
            [
                "unit.sales",
                "value.sales",
                "supermarket_supermarket-B",
                "supermarket_supermarket-C",
                "supermarket_supermarket-D",
                "variant_light",
                "variant_standard",
                "variant_vegan",
                "pack.size_351 - 500 GR",
                "pack.size_501 - 700 GR",
                "pack.size_701 - 1000 GR",
            ]
        ]

        y = data_dummies["volume.sales"]

        # Crear interacciones usando PolynomialFeatures (grado 2 para crear términos de interacción)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X)  # Generar las interacciones

        # Agregar una constante al modelo (intercepto)
        X_poly = sm.add_constant(X_poly)

        # Ajustar el modelo de regresión
        model = sm.OLS(y, X_poly).fit()

        if fix_significance:

            for i in range(interactions):
                p_values = model.pvalues
                significant_vars = p_values[
                    p_values < 0.05
                ].index  ############################### que quite solo una variable, no todas

                # Seleccionar las columnas significativas en X_poly
                X_poly_significant = X_poly[
                    :, [i for i in range(len(p_values)) if p_values[i] < 0.05]
                ]

                # Reajustar el modelo con las variables significativas
                model = sm.OLS(y, X_poly_significant).fit()

                model_summary_significant = model.summary()

        # Mostrar el resumen del modelo

        model_summary = model.summary()

        return data_dummies, model

    def modelization2(
        self,
        data_filtered_by_brand: pd.DataFrame,
        fix_significance: bool = False,
        interactions: int = 1,
    ) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:

        # Crear variables dummy para las columnas especificadas
        data_dummies = pd.get_dummies(
            data_filtered_by_brand,
            columns=["supermarket", "variant", "pack.size"],
            drop_first=True,
        )

        for col in [
            "supermarket_supermarket-B",
            "supermarket_supermarket-C",
            "supermarket_supermarket-D",
            "variant_light",
            "variant_standard",
            "variant_vegan",
            "pack.size_351 - 500 GR",
            "pack.size_501 - 700 GR",
            "pack.size_701 - 1000 GR",
        ]:
            data_dummies[col] = data_dummies[col].astype(int)

        X = data_dummies[
            [
                "unit.sales",
                "value.sales",
                "supermarket_supermarket-B",
                "supermarket_supermarket-C",
                "supermarket_supermarket-D",
                "variant_light",
                "variant_standard",
                "variant_vegan",
                "pack.size_351 - 500 GR",
                "pack.size_501 - 700 GR",
                "pack.size_701 - 1000 GR",
            ]
        ]

        y = data_dummies["volume.sales"]

        # Crear interacciones usando PolynomialFeatures (grado 2 para crear términos de interacción)
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X)  # Generar las interacciones

        # Obtener los nombres originales y las interacciones
        feature_names = poly.get_feature_names_out(X.columns)

        # Crear un DataFrame con los nombres de las variables
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names)

        # Agregar una constante al modelo (intercepto)
        X_poly_df = sm.add_constant(X_poly_df)

        # Ajustar el modelo de regresión
        model = sm.OLS(y, X_poly_df).fit()

        if fix_significance:
            for i in range(interactions):
                p_values = model.pvalues
                significant_vars = p_values[p_values < 0.05].index

                # Seleccionar las columnas significativas en X_poly_df
                X_poly_significant = X_poly_df[significant_vars]

                # Reajustar el modelo con las variables significativas
                model = sm.OLS(y, X_poly_significant).fit()

        # Mostrar el resumen del modelo
        model_summary = model.summary()

        return data_dummies, model

    def plot_resid_ACF_PACF(
        self, model: sm.regression.linear_model.RegressionResultsWrapper, lags: int = 40
    ) -> None:
        """
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the residuals of a given model.

        Parameters:
        model : statsmodels object
            The fitted model from which residuals are to be plotted.
        lags : int, optional
            The number of lags to include in the ACF and PACF plots (default is 40).

        Returns:
        None
        """
        residuals = model.resid

        plt.figure(figsize=(10, 6))
        plot_acf(residuals, lags=lags)
        plt.title("ACF residuals")
        plt.show()

        plt.figure(figsize=(10, 6))
        plot_pacf(residuals, lags=lags)
        plt.title("PACF residuals")
        plt.show()

    def test_stationarity(
        self, data_dummies: pd.DataFrame, sales: str = "volume.sales"
    ) -> None:
        """
        Tests the stationarity of a time series using the Augmented Dickey-Fuller (ADF) test.
        Parameters:
        -----------
        data_dummies : pandas.DataFrame
            The dataframe containing the time series data.
        sales : str, optional
            The column name of the time series to be tested. Default is "volume.sales".
        Returns:
        --------
        None
            Prints the ADF statistic, p-value, and critical values.
        Notes:
        ------
        - The ADF statistic indicates if the series is stationary. If the value is very negative and less than the critical values, the series is likely stationary.
        - The p-value: If it is less than 0.05, you can reject the null hypothesis of non-stationarity, indicating that the series is stationary.
        """

        # Dickie-Fuller's test
        adf_result = adfuller(data_dummies[sales])

        # Mostramos los resultados
        print(f"Estadístico ADF: {adf_result[0]}")
        print(f"Valor p: {adf_result[1]}")
        print("Valores críticos:")
        for key, value in adf_result[4].items():
            print(f"{key}: {value}")

        """ El estadístico ADF te dice si la serie es estacionaria. Si el valor es muy negativo y menor que los valores críticos, entonces es probable que la serie sea estacionaria.
        El valor p: Si es menor a 0.05, puedes rechazar la hipótesis nula de no estacionariedad, lo que indica que la serie es estacionaria."""
