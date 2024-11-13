import pandas as pd
import numpy as np
import patsy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

# from pmdarima.arima import auto_arima


#################################### SALES ANALYSIS CLASS ####################################
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

    #################################### DESCRIPTIVE ANALYSIS (GRAPHS) ####################################

    def detail_plot(
        self,
        brand: pd.Series,
        supermarket: pd.Series,
        pack_size: pd.Series,
        variant: pd.Series,
        sales: str = "volume.sales",
        plot: bool = True,
    ) -> None:
        """
        Applied: sa.detail_plot(sa.brand35, sa.supermarketA, sa.pack350, sa.variantF)
        """

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
        --------
        Applied: sa.separate_plot_by_flavour(sa.brand35, sa.pack350)
        --------
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
        --------
        Applied: sa.plot_everything_in_4_plots(sa.brand35)
        --------
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
        --------
        Applied: sa.plot_everything(sa.brand35, sa.variantF)
        --------
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

    def plot_resid_ACF_PACF(self, residues: pd.Series, lags: int = 40) -> None:
        """
        Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) of the residuals of a given model.

        Parameters:
        model : statsmodels object
            The fitted model from which residuals are to be plotted.
        lags : int, optional
            The number of lags to include in the ACF and PACF plots (default is 40).
        --------
        Returns:
        None
        --------
        Applied:
        sa.plot_resid_ACF_PACF(model)
        """
        ax, fig = plt.subplots(2, 1, figsize=(10, 12))

        plot_acf(residues, lags=lags, ax=fig[0])
        fig[0].set_title("ACF residuals")

        plot_pacf(residues, lags=lags, ax=fig[1])
        fig[1].set_title("PACF residuals")

        plt.show()

    #################################### MODELIZATION ####################################

    def modelization(
        self,
        data_filtered_by_brand: pd.DataFrame,
        interactions_deleted: list = [],
    ) -> sm.regression.linear_model.RegressionResultsWrapper:
        """
        Selects and fits a linear regression model on the provided data, creating interaction terms
        between numeric and categorical variables, as well as between categorical variables.
        Parameters:
        -----------
        data_filtered_by_brand : pd.DataFrame
            The input DataFrame filtered by brand, containing sales data and other features.
        interactions_deleted : list, optional
            A list of interaction terms to exclude from the model. Default is an empty list.
        Returns:
        --------
        sm.regression.linear_model.RegressionResultsWrapper
            The fitted linear regression model.
        --------
        Applied:
        sa.modelization(data[sa.brand35], sa.__interactions_delected_brand35__())
        --------
        Notes:
        ------
        - The function renames certain columns to remove problematic characters.
        - Dummy variables are created for categorical columns, and interaction terms are generated.
        - The formula for the regression model includes all variables except 'volume_sales' and
          those specified in `interactions_deleted`.
        """

        data_filtered_by_brand.rename(
            columns={
                "value.sales": "value_sales",
                "unit.sales": "unit_sales",
                "volume.sales": "volume_sales",
                "pack.size": "pack_size",
            },
            inplace=True,
        )

        df_dummies = pd.get_dummies(
            data_filtered_by_brand.drop(
                columns=["date", "brand"]
            ),  # exclude the 'date' and 'brand' columns
            columns=["supermarket", "variant", "pack_size"],
            drop_first=True,  # drop the first dummy variable to avoid multicollinearity (only k-1 dummies needed)
        )

        # Rename columns to remove spaces and hyphens for compatibility with patsy
        df_dummies.columns = df_dummies.columns.str.replace(" ", "_").str.replace(
            "-", "_"
        )

        ###INTERACTIONS
        # between numeric and dummy variables
        dummy_vars = [
            col
            for col in df_dummies.columns
            if col.startswith(("supermarket", "variant", "pack_size"))
        ]

        numeric_vars = ["unit_sales", "value_sales"]

        for num_var, dummy_var in product(numeric_vars, dummy_vars):
            interaction_name = f"{num_var}:{dummy_var}"
            df_dummies[interaction_name] = df_dummies[num_var] * df_dummies[dummy_var]

        # between dummy variables
        for dummy_var1, dummy_var2 in product(dummy_vars, repeat=2):
            if dummy_var1 < dummy_var2:  # Para evitar duplicar interacciones
                interaction_name = f"{dummy_var1}:{dummy_var2}"
                df_dummies[interaction_name] = (
                    df_dummies[dummy_var1] * df_dummies[dummy_var2]
                )

        ### INTERACTIONS  and FORMULA
        independent_vars = " + ".join(
            [
                col
                for col in df_dummies.columns
                if col
                not in [
                    "volume_sales",
                    *interactions_deleted,  # add * unpacks the tuple interactions_deleted to the list of elements with volume_sales
                ]
            ]
        )

        formula = f"volume_sales ~ {independent_vars} "

        ###MODEL
        model = smf.ols(formula=formula, data=df_dummies).fit()

        return model

    def modelization_with_backward_elimination(
        self, data_filtered_by_brand: pd.DataFrame
    ) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:

        data_filtered_by_brand.rename(
            columns={
                "value.sales": "value_sales",
                "unit.sales": "unit_sales",
                "volume.sales": "volume_sales",
                "pack.size": "pack_size",
            },
            inplace=True,
        )
        formule = "volume_sales ~ (unit_sales + value_sales + C(supermarket) + C(variant) + C(pack_size)) ** 2"

        y, X = patsy.dmatrices(
            formule, data=data_filtered_by_brand, return_type="dataframe"
        )

        # modelo = smf.ols(formula=formule, data=data_filtered_by_brand).fit()

        final_model, selected_columns = self.backward_elimination(X, y)

        return final_model, selected_columns

    def modelization_draw1(
        self,
        data_filtered_by_brand: pd.DataFrame,
        fix_significance: bool = False,
        interactions: int = 1,
    ) -> tuple[pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:

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

    def modelization_draw3(
        self, data_filtered_by_brand: pd.DataFrame
    ) -> sm.regression.linear_model.RegressionResultsWrapper:

        data_filtered_by_brand.rename(
            columns={
                "value.sales": "value_sales",
                "unit.sales": "unit_sales",
                "volume.sales": "volume_sales",
                "pack.size": "pack_size",
            },
            inplace=True,
        )

        formula = "volume_sales ~ (unit_sales + value_sales + C(supermarket) + C(variant) + C(pack_size)) ** 2"

        # Paso 2: Usa patsy para crear la matriz de diseño (X) y el vector de salida (y)
        y, X = patsy.dmatrices(
            formula, data=data_filtered_by_brand, return_type="dataframe"
        )

        # Paso 3: Inicializa un modelo de regresión lineal de sklearn
        model = LinearRegression()

        # Paso 4: Configura RFE para seleccionar el número deseado de características
        # En este caso, ajustaremos el modelo para seleccionar, por ejemplo, las 10 características más importantes
        selector = RFE(model, n_features_to_select=30, step=1)
        selector = selector.fit(X, y.values.ravel())

        # Paso 5: Filtrar las características seleccionadas
        selected_features = X.columns[selector.support_]

        # Paso 6: Crear un nuevo modelo en statsmodels solo con las características seleccionadas
        X_selected = X[selected_features]
        final_model = sm.OLS(y, X_selected).fit()

        return final_model

    def backward_elimination(self, X, y, alpha=0.05):
        # Agrega la constante para el intercepto
        X = sm.add_constant(X, has_constant="add")

        numVars = X.shape[1]
        for i in range(numVars):
            model = sm.OLS(y, X).fit()
            # Excluir el p-valor de la constante para no eliminarla
            pvalues = model.pvalues.drop("const", errors="ignore")
            maxPVal = pvalues.max()

            # Elimina la variable con el p-valor más alto, pero no la constante
            if maxPVal > alpha:
                for j in range(numVars - i):
                    if (
                        model.pvalues.index[j] != "const"
                        and model.pvalues[j] == maxPVal
                    ):
                        X = X.drop(X.columns[j], axis=1)
            else:
                break

        return model, X.columns

    def ARIMA(
        self,
        residues: pd.Series,
        model_chosen: tuple = (1, 1, 0),
        diff_need_for_residues: bool = False,
    ):
        """
        Input: sa.ARIMA(residues, model_chosen=(1, 1, 0))
        """

        if diff_need_for_residues:
            residues = residues.diff().dropna()

        model_arima = ARIMA(residues, order=model_chosen).fit()

        return model_arima

    def ARIMAX(
        self,
        endog: pd.Series = None,
        exog: pd.Series = None,
        model_chosen: tuple = (1, 1, 0),
    ):
        """
        Input: sa.ARIMAX(data_dummies["volume.sales"], data_dummies["value.sales"])
        """

        if exog is None:
            exog = self.data["value.sales"]

        if endog is None:
            endog = self.data["volume.sales"]

        model_arimax = ARIMA(endog, exog=exog, order=model_chosen).fit()

        return model_arimax

    # def autoArima(
    #     self,
    #     endog: pd.Series = None,
    #     exog: pd.Series = None,
    #     seasonal: bool = False,
    # ):

    #     if exog is None:
    #         exog = self.data["value.sales"]

    #     if endog is None:
    #         endog = self.data["volume.sales"]

    #     model_auto_arima = auto_arima(
    #         endog,
    #         exog,
    #         seasonal=seasonal,
    #         stepwise=True,
    #         trace=True,
    #         error_action="ignore",
    #         suppress_warnings=True,
    #     )

    #     return model_auto_arima

    #################################### TESTS ####################################

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
        --------
        Applied: sa.test_stationarity(data)
        --------
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

    def test_correlation_residues(
        self, residues: pd.Series, lags_ACF_PACF: int = 40, lags_ljungbox: int = 10
    ) -> None:
        """
        Check the correlation between the residues of the model seeing the ACF and PACF plots. Anyways the Ljung-Box test is a  way to check the correlation between the residues. If the p-value is less than 0.05, then the residues are correlated and the model is not good.
        """

        self.plot_resid_ACF_PACF(residues, lags=lags_ACF_PACF)
        ljung_box_test = acorr_ljungbox(residues, lags=[lags_ljungbox])

        return ljung_box_test

    #################################### UTILITIES ####################################

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

    def __interactions_delected_brand35__(self) -> list:
        """
        This method returns a list of interactions that have been deleted for brand 35.

        Returns:
            list: interactions deleted for brand 35 in the modelization_selector2 method
        """

        interactions_deleted = [
            "pack_size_701___1000_GR:variant_vegan",
            "value_sales:pack_size_501___700_GR",
            "value_sales:supermarket_supermarket_D",
            "supermarket_supermarket_C:supermarket_supermarket_D",
            "pack_size_501___700_GR:variant_standard",
            "supermarket_supermarket_C:variant_vegan",
            "value_sales:pack_size_351___500_GR",
            "supermarket_supermarket_D:variant_standard",
            "pack_size_701___1000_GR",
            "pack_size_351___500_GR:supermarket_supermarket_B",
            "pack_size_351___500_GR:pack_size_701___1000_GR",
            "pack_size_351___500_GR:pack_size_501___700_GR",
            "pack_size_501___700_GR:pack_size_701___1000_GR",
            "supermarket_supermarket_B:supermarket_supermarket_C",
            "pack_size_501___700_GR:supermarket_supermarket_C",
            "pack_size_701___1000_GR:variant_standard",
            "pack_size_701___1000_GR:supermarket_supermarket_C",
            "variant_standard:variant_vegan",
            "variant_vegan",
            "pack_size_501___700_GR:supermarket_supermarket_B",
            "pack_size_701___1000_GR:supermarket_supermarket_D",
            "pack_size_501___700_GR:variant_vegan",
            "supermarket_supermarket_B:variant_vegan",
            "variant_light",
            "pack_size_351___500_GR:variant_vegan",
            "supermarket_supermarket_B:supermarket_supermarket_D",
            "supermarket_supermarket_B:variant_light",
            "pack_size_701___1000_GR:variant_light",
            "pack_size_351___500_GR:supermarket_supermarket_C",
            "supermarket_supermarket_B:variant_standard",
            "pack_size_701___1000_GR:supermarket_supermarket_B",
            "supermarket_supermarket_C:variant_light",
            "pack_size_351___500_GR:variant_light",
            "supermarket_supermarket_D:variant_vegan",
            "supermarket_supermarket_D",
            "supermarket_supermarket_D:variant_light",
            "pack_size_501___700_GR:variant_light",
            "pack_size_501___700_GR:supermarket_supermarket_D",
            "pack_size_501___700_GR",
            "variant_light:variant_vegan",
            "pack_size_351___500_GR:supermarket_supermarket_D",
            "variant_light:variant_standard",
            "unit_sales:variant_standard",
            "supermarket_supermarket_B",
            "pack_size_351___500_GR",
            "pack_size_351___500_GR:variant_standard",
        ]

        return interactions_deleted
