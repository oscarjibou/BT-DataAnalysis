import os
import pandas as pd
import numpy as np
import patsy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from itertools import product
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from scipy.stats import jarque_bera
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from urllib3.exceptions import NotOpenSSLWarning
from pmdarima.arima import auto_arima
from scipy import stats
from prophet import Prophet

warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


class SalesAnalysis:  # TODO: add a class for descriptive analysis
    def __init__(self, raw_data: pd.DataFrame):

        # Assign the original data to a class attribute
        self.raw_data = raw_data

        # CleaningData
        self.data = self.cleaning_data()

        self.__variables__()

        # Convert the weeks into months using the convert_weeks_to_months method
        self.data = self.convert_weeks_to_months()

        self.data = self.order_dataset_by_date(self.data)

        self.data = self.add_price_column(self.data)

        self.data = self.data.dropna()  # drop the rows with NaN values

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

    def analysis_residuals(self, residues: pd.Series, fitted_values: pd.Series):
        """
        Analiza los residuos de un modelo ARIMAX.
        Parameters:
        -----------
        residues: pd.Series
            Los residuos del modelo ARIMAX.
        fitted_values: pd.Series
            Los valores ajustados del modelo ARIMAX.
        """
        # Crear figura con 6 subplots (3 filas x 2 columnas)
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(
            "Análisis de Residuos - Diagnóstico del Modelo ARIMAX",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Residuos vs. Tiempo (arriba-izquierda)
        axes[0, 0].plot(residues, color="blue", linewidth=0.8)
        axes[0, 0].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0, 0].set_title("Residuos vs. Tiempo", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Index")
        axes[0, 0].set_ylabel("Residuos")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Residuos vs. Valores Ajustados (arriba-derecha)
        axes[0, 1].scatter(fitted_values, residues, alpha=0.5, s=10, color="blue")
        axes[0, 1].axhline(y=0, color="red", linestyle="--", linewidth=1)
        axes[0, 1].set_title(
            "Residuos vs. Valores Ajustados", fontsize=12, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Fitted Values")
        axes[0, 1].set_ylabel("Residuos")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot (medio-izquierda)
        stats.probplot(residues, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot (Normalidad)", fontsize=12, fontweight="bold")
        axes[1, 0].grid(True, alpha=0.3)

        # 4. ACF (Autocorrelación) (medio-derecha)
        plot_acf(residues, lags=12, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title("Autocorrelación (ACF)", fontsize=12, fontweight="bold")
        axes[1, 1].grid(True, alpha=0.3)

        # 5. PACF (Autocorrelación Parcial) (abajo-izquierda)
        plot_pacf(residues, lags=12, ax=axes[2, 0], alpha=0.05, method="ywm")
        axes[2, 0].set_title(
            "Autocorrelación Parcial (PACF)", fontsize=12, fontweight="bold"
        )
        axes[2, 0].grid(True, alpha=0.3)

        # 6. Histograma de Residuos (abajo-derecha)
        axes[2, 1].hist(residues, bins=30, edgecolor="black", alpha=0.7, color="blue")
        axes[2, 1].set_title("Histograma de Residuos", fontsize=12, fontweight="bold")
        axes[2, 1].set_xlabel("Residuos")
        axes[2, 1].set_ylabel("Frequency")
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
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
        """
        Modelization with backward elimination.
        Parameters:
        data_filtered_by_brand: pd.DataFrame
            The data filtered by brand.
        Returns:
        - model, which is the final model after backward elimination.
        - design_info, which is the design info of the model. It contains the design matrix and the design formula.
        - selected_columns, which are the columns that were selected by the backward elimination.
        """

        data_filtered_by_brand.rename(  # renaming columns for compatibility with patsy
            columns={
                "value.sales": "value_sales",
                "unit.sales": "unit_sales",
                "volume.sales": "volume_sales",
                "pack.size": "pack_size",
            },
            inplace=True,
        )
        formule = (
            "volume_sales ~ (price + C(supermarket) + C(variant) + C(pack_size)) ** 2"
        )

        y, X = patsy.dmatrices(
            formule, data=data_filtered_by_brand, return_type="dataframe"
        )
        design_info = X.design_info

        # modelo = smf.ols(formula=formule, data=data_filtered_by_brand).fit()

        final_model, selected_columns = self.backward_elimination_old(X, y, alpha=0.05)

        data_filtered_by_brand.rename(
            columns={
                "value_sales": "value.sales",
                "unit_sales": "unit.sales",
                "volume_sales": "volume.sales",
                "pack_size": "pack.size",
            },
            inplace=True,
        )

        return final_model, design_info, selected_columns

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

    def backward_elimination_old(self, X, y, alpha=0.05):
        # Agrega la constante para el intercepto
        X = sm.add_constant(X, has_constant="skip")

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

    def backward_elimination(self, X: pd.DataFrame, y, alpha: float = 0.05):
        """
        X debe venir de patsy.dmatrices(..., return_type='dataframe')
        y es un pandas Series/DataFrame de dmatrices (columna única).
        """
        # Patsy ya incluye 'Intercept' en X. Aseguramos no añadir otra.
        X_be = X.copy()

        # nombres de constante posibles
        CONST_CANDIDATES = [c for c in ["const", "Intercept"] if c in X_be.columns]

        while True:
            model = sm.OLS(y, X_be).fit()
            # quitamos p-valor de la constante (si existe) para no eliminarla
            pvals = model.pvalues.copy()
            for c in CONST_CANDIDATES:
                if c in pvals.index:
                    pvals = pvals.drop(c)
            if pvals.empty:
                break

            max_p = pvals.max()
            if max_p > alpha:
                worst_var = pvals.idxmax()  # nombre de la peor variable
                # eliminar por nombre (no por posición)
                X_be = X_be.drop(columns=[worst_var])
            else:
                break

        return model, X_be.columns.tolist()

    def regression_with_backward_elimination(
        self,
        data: pd.DataFrame,
        target: str = "volume.sales",
        alpha: float = 0.05,
        verbose: bool = False,
        aggregation_formule: str = None,
    ) -> tuple:
        """
        Realiza regresión OLS con backward elimination automática.

        Elimina automáticamente las variables no significativas (p-valor > alpha)
        del modelo de regresión.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame con los datos. Debe contener las columnas: price, supermarket,
            variant, pack.size, brand, y la columna target.
        target : str, default="volume.sales"
            Nombre de la columna objetivo (variable dependiente).
        alpha : float, default=0.05
            Nivel de significancia para eliminar variables. Variables con p-valor > alpha
            serán eliminadas.
        verbose : bool, default=False
            Si True, muestra el progreso de eliminación de variables.
        agrregation_formule : str, default=None
            Fórmula de agregación para agregar al modelo.
            Example: "Q('dummy_outlier_2022_07_31')"
        Returns:
        --------
        tuple:
            - model: Modelo final de statsmodels después de backward elimination
            - selected_vars: Lista de variables seleccionadas (nombres de columnas)
            - eliminated_vars: Lista de variables eliminadas durante el proceso
        """
        # Hacer una copia para no modificar el DataFrame original
        data_work = data.copy()

        # Renombrar columnas para compatibilidad con patsy (puntos → guiones bajos)
        #########################################################################################
        rename_dict = {
            "value.sales": "value_sales",
            "unit.sales": "unit_sales",
            "volume.sales": "volume_sales",
            "pack.size": "pack_size",
        }
        rename_dict = {k: v for k, v in rename_dict.items() if k in data_work.columns}
        data_work.rename(columns=rename_dict, inplace=True)

        # Actualizar el nombre del target si fue renombrado
        target_renamed = rename_dict.get(target, target)

        # Crear fórmula base
        formule = (
            f"{target_renamed} ~ price + C(supermarket) + C(variant) + C(pack_size) + "
            f"C(brand) + (price + C(brand)) ** 2"
        )

        if aggregation_formule is not None:
            formule += f" + {aggregation_formule}"

        if verbose:
            print(f"Fórmula del modelo:")
            print(formule)
            if aggregation_formule is not None:
                print(f"\n✅ {aggregation_formule} agregada al modelo")

        # Crear matrices de diseño usando patsy
        y, X = patsy.dmatrices(formule, data=data_work, return_type="dataframe")

        # Guardar todas las variables iniciales
        initial_vars = set(X.columns.tolist())

        # Nombres posibles de constante/intercepto
        CONST_CANDIDATES = [c for c in ["const", "Intercept"] if c in X.columns]

        # Lista para rastrear variables eliminadas
        eliminated_vars = []

        # Backward elimination
        iteration = 0
        while True:
            iteration += 1

            # Ajustar el modelo OLS
            model = sm.OLS(y, X).fit()

            # Obtener p-valores excluyendo la constante/intercepto
            pvals = model.pvalues.copy()
            for c in CONST_CANDIDATES:
                if c in pvals.index:
                    pvals = pvals.drop(c)

            # Si no hay más variables (solo queda la constante), terminar
            if pvals.empty:
                if verbose:
                    print("No quedan variables para eliminar (solo constante).")
                break

            # Encontrar el p-valor máximo
            max_p = pvals.max()

            # Si el p-valor máximo es mayor que alpha, eliminar esa variable
            if max_p > alpha:
                worst_var = pvals.idxmax()

                if verbose:
                    print(
                        f"Iteración {iteration}: Eliminando '{worst_var}' (p-valor = {max_p:.4f})"
                    )

                # Eliminar la variable por nombre
                X = X.drop(columns=[worst_var])
                eliminated_vars.append(worst_var)
            else:
                if verbose:
                    print(
                        f"Iteración {iteration}: Todas las variables restantes son significativas (p-valor ≤ {alpha})"
                    )
                break

        # Ajustar el modelo final con las variables seleccionadas
        final_model = sm.OLS(y, X).fit()

        # Obtener las variables seleccionadas
        selected_vars = X.columns.tolist()

        if verbose:
            print(f"\nResumen:")
            print(f"  Variables iniciales: {len(initial_vars)}")
            print(f"  Variables seleccionadas: {len(selected_vars)}")
            print(f"  Variables eliminadas: {len(eliminated_vars)}")
            print(f"  R² ajustado: {final_model.rsquared_adj:.4f}")

        return final_model, selected_vars, eliminated_vars
        pass

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

    def ARIMAX(  # FIXME: si no lo uso, borrarlo. Creo que el Sarimax es lo mismo que el ARIMAX
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

    def x_train_exog_custom(
        self,
        train_data,
        selected_columns,
        model,
        target="volume.sales",
        aggregation_formule=None,
    ):
        """
        Prepara las variables exógenas usando la misma fórmula que regression_with_backward_elimination.

        Parameters:
        -----------
        train_data : pd.DataFrame
            Datos de entrenamiento
        selected_columns : list
            Lista de columnas seleccionadas por backward elimination
        model : statsmodels model
            Modelo ajustado
        target : str
            Nombre de la columna objetivo

        Returns:
        --------
        X_train_exog : pd.DataFrame
            DataFrame con las variables exógenas seleccionadas
        """
        train_data_for_patsy = train_data.copy()

        # Renombrar columnas para compatibilidad con patsy
        rename_dict = {
            "value.sales": "value_sales",
            "unit.sales": "unit_sales",
            "volume.sales": "volume_sales",
            "pack.size": "pack_size",
        }
        rename_dict = {
            k: v for k, v in rename_dict.items() if k in train_data_for_patsy.columns
        }
        train_data_for_patsy.rename(columns=rename_dict, inplace=True)

        # Actualizar el nombre del target si fue renombrado
        target_renamed = rename_dict.get(target, target)

        # Usar la MISMA fórmula que en regression_with_backward_elimination
        formula = (
            "volume_sales ~ price + C(supermarket) + C(variant) + C(pack_size) + "
            "C(brand) + (price + C(brand)) ** 2"
        )
        if aggregation_formule is not None:
            formula += f" + {aggregation_formule}"

        # Crear la matriz de diseño
        y_design, X_design = patsy.dmatrices(
            formula, data=train_data_for_patsy, return_type="dataframe"
        )

        # Filtrar solo las columnas seleccionadas (excluyendo Intercept)
        selected_columns_no_intercept = [
            col for col in selected_columns if col != "Intercept" and col != "const"
        ]

        # Verificar que las columnas existen
        missing_cols = [
            col for col in selected_columns_no_intercept if col not in X_design.columns
        ]
        if missing_cols:
            print(
                f"⚠️ Advertencia: Las siguientes columnas no se encontraron en X_design: {missing_cols}"
            )
            print(f"Columnas disponibles en X_design: {list(X_design.columns)}")
            # Filtrar solo las que existen
            selected_columns_no_intercept = [
                col for col in selected_columns_no_intercept if col in X_design.columns
            ]

        X_train_exog = X_design[selected_columns_no_intercept]

        # Verificar que coinciden con el modelo
        model_features = list(model.params.index)
        exog_features = list(X_train_exog.columns)

        model_features_no_intercept = [
            f for f in model_features if f not in ["Intercept", "const"]
        ]
        if set(model_features_no_intercept) == set(exog_features):
            print("✅ YES - All features match perfectly!")
        else:
            print("❌ NO - Features don't match")
            print(
                f"Model features (sin intercept): {sorted(model_features_no_intercept)}"
            )
            print(f"Exog features: {sorted(exog_features)}")
            missing_in_exog = set(model_features_no_intercept) - set(exog_features)
            missing_in_model = set(exog_features) - set(model_features_no_intercept)
            if missing_in_exog:
                print(f"Missing in exog: {missing_in_exog}")
            if missing_in_model:
                print(f"Missing in model: {missing_in_model}")

        return X_train_exog

    def x_test_exog(self, test_data, selected_columns, design_info):
        """
        Prepare exogenous variables for test data using the SAME transformation
        as training data (same formula, same selected columns, same design_info).

        Parameters:
        -----------
        test_data : pd.DataFrame
            The test dataset containing the same columns as training data.
        selected_columns : list
            The list of columns selected by backward elimination (from modelization_with_backward_elimination).
        design_info : patsy.DesignInfo
            The design info from the training data to ensure identical column creation.

        Returns:
        --------
        pd.DataFrame
            Test exogenous variables with the same columns as X_train_exog (without Intercept).

        Notes:
        ------
        - Uses the design_info from training to create identical columns
        - Handles missing categories in test data by using training's design matrix
        - Filters to use only the selected_columns from backward elimination
        - Removes Intercept column as it's not needed for exogenous variables
        """
        test_data_for_patsy = test_data.copy()

        # Rename columns to match Patsy naming convention
        test_data_for_patsy.rename(
            columns={
                "value.sales": "value_sales",
                "unit.sales": "unit_sales",
                "volume.sales": "volume_sales",
                "pack.size": "pack_size",
            },
            inplace=True,
        )

        # Use the design_info from training to create the SAME design matrix
        # This ensures that even if some categories are missing in test data,
        # the same columns will be created (with zeros for missing categories)
        X_design = patsy.build_design_matrices(
            [design_info], test_data_for_patsy, return_type="dataframe"
        )[0]

        # Use the SAME selected columns (without Intercept)
        selected_columns_no_intercept = [
            col for col in selected_columns if col != "Intercept"
        ]

        X_test_exog = X_design[selected_columns_no_intercept]

        return X_test_exog

    def clean_exogenous_variables(self, X_exog, corr_threshold=0.95, verbose=False):
        """
        Limpia variables exógenas eliminando:
        1. Variables constantes (rango = 0)
        2. Variables con correlación perfecta (|r| = 1.0)
        3. Variables altamente correlacionadas (|r| > corr_threshold)

        Parameters:
        -----------
        X_exog : pd.DataFrame
            Variables exógenas a limpiar
        corr_threshold : float, default=0.95
            Umbral de correlación para eliminar variables (0.95 = 95%)
        verbose : bool, default=False
            Si True, imprime información sobre el proceso

        Returns:
        --------
        X_clean : pd.DataFrame
            Variables exógenas limpias
        removed_vars : list
            Lista de variables eliminadas
        cond_number_before : float
            Número de condición antes de la limpieza
        cond_number_after : float
            Número de condición después de la limpieza
        """

        X_clean = X_exog.copy()
        removed_vars = []

        # Calcular número de condición inicial
        try:
            X_matrix = X_clean.values
            X_with_const = np.column_stack([np.ones(len(X_matrix)), X_matrix])
            cond_number_before = np.linalg.cond(X_with_const)
        except:
            cond_number_before = None

        # 1. Eliminar variables constantes
        ranges = X_clean.max() - X_clean.min()
        constant_vars = ranges[ranges == 0].index.tolist()
        if constant_vars:
            removed_vars.extend(constant_vars)
            X_clean = X_clean.drop(columns=constant_vars)

        # 2. Eliminar variables con correlación perfecta
        corr_matrix = X_clean.corr()
        perfect_corr_vars = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= 1.0 - 1e-10:
                    var2 = corr_matrix.columns[j]
                    if var2 not in perfect_corr_vars:
                        perfect_corr_vars.add(var2)

        if perfect_corr_vars:
            removed_vars.extend(list(perfect_corr_vars))
            X_clean = X_clean.drop(columns=list(perfect_corr_vars))

        # 3. Eliminar variables altamente correlacionadas
        corr_matrix = X_clean.corr()
        high_corr_vars = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    var2 = corr_matrix.columns[j]
                    if var2 not in high_corr_vars:
                        high_corr_vars.add(var2)

        if high_corr_vars:
            removed_vars.extend(list(high_corr_vars))
            X_clean = X_clean.drop(columns=list(high_corr_vars))

        # Calcular número de condición final
        try:
            X_matrix_clean = X_clean.values
            X_with_const_clean = np.column_stack(
                [np.ones(len(X_matrix_clean)), X_matrix_clean]
            )
            cond_number_after = np.linalg.cond(X_with_const_clean)
        except:
            cond_number_after = None

        return X_clean, removed_vars, cond_number_before, cond_number_after

    #################################### TESTS ####################################

    def residual_white_noise_test(
        self, residues: pd.Series, lags_ljungbox: int = 10
    ) -> None:

        ######## Residues Analysis (White Noise) ########
        """
        # 1. Mean value is zero
        # 2. Constant variance
            - Arch Test: p-value > 0.05 (no autocorrelation) variance is constant
        # 3. Covaariance between two observations is only dependent on the lag between them
        # 4 Normal distribution
            - Jarque-Bera Test: p-value > 0.05 (data is normally distributed)
            - Shapiro-Wilk Test: p-value > 0.05 (data is normally distributed)
        ---------------------------------------------
        Lfung-Box Test: p-value > 0.05 (data is white noise)
        Durbin-Watson Test: 2.0 (no autocorrelation)
        """

        # ARCH test
        arch_test = het_arch(residues)
        print(
            f"[Heteroscedasticity Test] ARCH p-value: {arch_test[1]} -- range(> 0.05)"
        )

        # Jarque-Bera test
        jb_stat, jb_p_value = jarque_bera(residues)
        print(f"[Normality Test] Jarque-Bera p-value: {jb_p_value} -- range(> 0.05)")

        # Shapiro-Wilk test
        sw_stat, sw_p_value = shapiro(residues)
        print(f"[Normality Test] Shapiro-Wilk p-value: {sw_p_value} -- range(> 0.05)")

        # Ljung-Box test
        ljung_box_test = acorr_ljungbox(residues, lags=[lags_ljungbox])
        print(
            f"[Autocorrelation Test] Ljung-Box p-value:\n {ljung_box_test} -- range(> 0.05)"
        )

        # Durbin-Watson test
        dw_stat = durbin_watson(residues)
        print(
            f"[Autocorrelation Test first order] Durbin-Watson statistic: {dw_stat} -- range(2.0)"
        )

        return None

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
        print(f"Valor p: {adf_result[1]} -- es estacionaria si p < 0.05")
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

    def ADF_KPSS_test(self, series: pd.Series, series_name: str = "Serie"):
        """
        Tests de estacionariedad ADF y KPSS sobre una serie temporal.

        Parameters:
        -----------
        series : pd.Series
            La serie temporal a analizar (debe ser una pd.Series, no un DataFrame)
        series_name : str
            Nombre descriptivo de la serie para los prints

        Returns:
        --------
        dict : Diccionario con los resultados de los tests
        """
        print("=" * 100)
        print(f"ANÁLISIS DE ESTACIONARIEDAD: {series_name}")
        print("=" * 100)

        # Diferenciación regular (d=1)
        series_diff = series.diff().dropna()
        # Segunda diferenciación regular (d=2) - CORREGIDO: era diff(periods=2)
        series_diff2 = series.diff().diff().dropna()

        results = {}

        # Test serie original (d=0)
        print("\n📊 Test estacionariedad serie ORIGINAL (d=0)")
        print("-" * 50)
        adfuller_result = adfuller(series)
        kpss_result = kpss(series, regression="c")
        print(
            f"ADF Statistic: {adfuller_result[0]:.4f}, p-value: {adfuller_result[1]:.4f}"
        )
        print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
        adf_stationary = adfuller_result[1] < 0.05
        kpss_stationary = kpss_result[1] > 0.05
        print(
            f"   → ADF: {'Estacionaria ✓' if adf_stationary else 'No estacionaria ✗'}"
        )
        print(
            f"   → KPSS: {'Estacionaria ✓' if kpss_stationary else 'No estacionaria ✗'}"
        )
        results["d0"] = {
            "adf_pvalue": adfuller_result[1],
            "kpss_pvalue": kpss_result[1],
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
        }

        # Test serie diferenciada una vez (d=1)
        print("\n📊 Test estacionariedad serie DIFERENCIADA (d=1)")
        print("-" * 50)
        adfuller_result_diff = adfuller(series_diff)
        kpss_result_diff = kpss(series_diff, regression="c")
        print(
            f"ADF Statistic: {adfuller_result_diff[0]:.4f}, p-value: {adfuller_result_diff[1]:.4f}"
        )
        print(
            f"KPSS Statistic: {kpss_result_diff[0]:.4f}, p-value: {kpss_result_diff[1]:.4f}"
        )
        adf_stationary = adfuller_result_diff[1] < 0.05
        kpss_stationary = kpss_result_diff[1] > 0.05
        print(
            f"   → ADF: {'Estacionaria ✓' if adf_stationary else 'No estacionaria ✗'}"
        )
        print(
            f"   → KPSS: {'Estacionaria ✓' if kpss_stationary else 'No estacionaria ✗'}"
        )
        results["d1"] = {
            "adf_pvalue": adfuller_result_diff[1],
            "kpss_pvalue": kpss_result_diff[1],
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
        }

        # Test serie diferenciada dos veces (d=2)
        print("\n📊 Test estacionariedad serie SEGUNDA DIFERENCIACIÓN (d=2)")
        print("-" * 50)
        adfuller_result_diff2 = adfuller(series_diff2)
        kpss_result_diff2 = kpss(series_diff2, regression="c")
        print(
            f"ADF Statistic: {adfuller_result_diff2[0]:.4f}, p-value: {adfuller_result_diff2[1]:.4f}"
        )
        print(
            f"KPSS Statistic: {kpss_result_diff2[0]:.4f}, p-value: {kpss_result_diff2[1]:.4f}"
        )
        adf_stationary = adfuller_result_diff2[1] < 0.05
        kpss_stationary = kpss_result_diff2[1] > 0.05
        print(
            f"   → ADF: {'Estacionaria ✓' if adf_stationary else 'No estacionaria ✗'}"
        )
        print(
            f"   → KPSS: {'Estacionaria ✓' if kpss_stationary else 'No estacionaria ✗'}"
        )
        results["d2"] = {
            "adf_pvalue": adfuller_result_diff2[1],
            "kpss_pvalue": kpss_result_diff2[1],
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
        }

        # Recomendación de d
        print("\n" + "=" * 50)
        print("RECOMENDACIÓN PARA d:")
        if results["d0"]["adf_stationary"] and results["d0"]["kpss_stationary"]:
            print("   → d=0 (serie ya es estacionaria)")
            results["recommended_d"] = 0
        elif results["d1"]["adf_stationary"] and results["d1"]["kpss_stationary"]:
            print("   → d=1 (necesita una diferenciación)")
            results["recommended_d"] = 1
        else:
            print("   → d=2 (necesita dos diferenciaciones)")
            results["recommended_d"] = 2
        print("=" * 50)

        # Gráficos
        warnings.filterwarnings("ignore")
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(12, 10), sharex=True)
        series.plot(ax=axs[0], title="Serie original (d=0)")
        series_diff.plot(ax=axs[1], title="Diferenciación orden 1 (d=1)")
        series_diff2.plot(ax=axs[2], title="Diferenciación orden 2 (d=2)")
        plt.tight_layout()
        plt.show()

        return results

    def seasonal_stationarity_test(
        self, series: pd.Series, m: int = 12, series_name: str = "Serie"
    ):
        """
        Tests de estacionariedad estacional para determinar D.

        Parameters:
        -----------
        series : pd.Series
            La serie temporal a analizar
        m : int
            Periodicidad estacional (12 para datos mensuales, 4 para trimestrales, etc.)
        series_name : str
            Nombre descriptivo de la serie

        Returns:
        --------
        dict : Diccionario con los resultados de los tests
        """
        print("=" * 100)
        print(f"ANÁLISIS DE ESTACIONARIEDAD ESTACIONAL: {series_name} (m={m})")
        print("=" * 100)

        results = {}

        # Diferenciación estacional (D=1)
        series_seasonal_diff = series.diff(m).dropna()

        # Test serie original para estacionalidad
        print("\n📊 Test estacionariedad serie ORIGINAL (D=0)")
        print("-" * 50)
        adfuller_result = adfuller(series)
        kpss_result = kpss(series, regression="c")
        print(
            f"ADF Statistic: {adfuller_result[0]:.4f}, p-value: {adfuller_result[1]:.4f}"
        )
        print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")
        adf_stationary = adfuller_result[1] < 0.05
        kpss_stationary = kpss_result[1] > 0.05
        print(
            f"   → ADF: {'Estacionaria ✓' if adf_stationary else 'No estacionaria ✗'}"
        )
        print(
            f"   → KPSS: {'Estacionaria ✓' if kpss_stationary else 'No estacionaria ✗'}"
        )
        results["D0"] = {
            "adf_pvalue": adfuller_result[1],
            "kpss_pvalue": kpss_result[1],
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
        }

        # Test serie con diferenciación estacional (D=1)
        print(
            f"\n📊 Test estacionariedad serie con DIFERENCIACIÓN ESTACIONAL (D=1, lag={m})"
        )
        print("-" * 50)
        adfuller_result_s = adfuller(series_seasonal_diff)
        kpss_result_s = kpss(series_seasonal_diff, regression="c")
        print(
            f"ADF Statistic: {adfuller_result_s[0]:.4f}, p-value: {adfuller_result_s[1]:.4f}"
        )
        print(
            f"KPSS Statistic: {kpss_result_s[0]:.4f}, p-value: {kpss_result_s[1]:.4f}"
        )
        adf_stationary = adfuller_result_s[1] < 0.05
        kpss_stationary = kpss_result_s[1] > 0.05
        print(
            f"   → ADF: {'Estacionaria ✓' if adf_stationary else 'No estacionaria ✗'}"
        )
        print(
            f"   → KPSS: {'Estacionaria ✓' if kpss_stationary else 'No estacionaria ✗'}"
        )
        results["D1"] = {
            "adf_pvalue": adfuller_result_s[1],
            "kpss_pvalue": kpss_result_s[1],
            "adf_stationary": adf_stationary,
            "kpss_stationary": kpss_stationary,
        }

        # Análisis de ACF para detectar estacionalidad
        print(
            f"\n📊 Análisis ACF para detectar patrón estacional (lags múltiplos de {m})"
        )
        print("-" * 50)
        from statsmodels.tsa.stattools import acf

        # Limitar nlags al máximo permitido por la longitud de la serie
        max_lag = min(3 * m, len(series) - 1)
        acf_values = acf(series, nlags=max_lag)

        # Solo incluir lags estacionales que estén disponibles
        seasonal_lags = [lag for lag in [m, 2 * m, 3 * m] if lag < len(acf_values)]

        print("Autocorrelaciones en lags estacionales:")
        significant_seasonal = False
        significance = 1.96 / np.sqrt(len(series))

        if len(seasonal_lags) == 0:
            print(
                f"   ⚠️ Serie muy corta ({len(series)} obs). No se pueden analizar lags estacionales."
            )
        else:
            for lag in seasonal_lags:
                acf_val = acf_values[lag]
                is_significant = abs(acf_val) > significance
                if is_significant:
                    significant_seasonal = True
                print(
                    f"   Lag {lag}: ACF = {acf_val:.4f} {'(significativo ✗ - hay estacionalidad)' if is_significant else '(no significativo ✓)'}"
                )

        results["significant_seasonal_acf"] = significant_seasonal

        # Recomendación de D
        print("\n" + "=" * 50)
        print("RECOMENDACIÓN PARA D:")
        if significant_seasonal:
            print(f"   → D=1 (se detecta patrón estacional significativo en ACF)")
            results["recommended_D"] = 1
        elif (
            not results["D0"]["adf_stationary"] or not results["D0"]["kpss_stationary"]
        ):
            # Si hay correlaciones estacionales significativas, probablemente necesita D=1
            print(f"   → D=1 (la serie original no es completamente estacionaria)")
            results["recommended_D"] = 1
        else:
            print(f"   → D=0 (no se detecta patrón estacional fuerte)")
            results["recommended_D"] = 0
        print("=" * 50)

        # Gráficos
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

        # Serie original y diferenciada estacionalmente
        series.plot(ax=axs[0, 0], title="Serie original (D=0)")
        series_seasonal_diff.plot(
            ax=axs[0, 1], title=f"Serie con diferenciación estacional (D=1, lag={m})"
        )

        # ACF de ambas - limitar lags según longitud de serie
        acf_lags_original = min(3 * m, len(series) - 2)
        acf_lags_diff = min(3 * m, len(series_seasonal_diff) - 2)

        plot_acf(
            series, lags=acf_lags_original, ax=axs[1, 0], title="ACF Serie Original"
        )
        plot_acf(
            series_seasonal_diff,
            lags=acf_lags_diff,
            ax=axs[1, 1],
            title=f"ACF Serie Diferenciada Estacionalmente",
        )

        plt.tight_layout()
        plt.show()

        return results

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

    def add_price_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a 'price' column to the data by dividing the 'value.sales' column by the 'unit.sales' column.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing the sales data.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with an additional 'price' column.
        """

        data["price"] = data["value.sales"] / data["unit.sales"]

        return data

    def divide_data_for_train_and_test(
        self, data: pd.DataFrame, train_size: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # FIXME: esta funcion no es correcta, hay que cambiarla para que se pueda dividir el data en train y test segun las fechas que se le pasen.
        """
        La forma correcta es:

        date_min = data["date"].min()
        date_max = data["date"].max()
        date_cutoff = pd.Timestamp('2023-06-30')

        train_data_ = data[data['date'] <= date_cutoff].copy()
        test_data_ = data[(data['date'] >= date_cutoff + pd.Timedelta(days=1)) & (data['date'] <= date_max)].copy()

        """

        """
        Divides the data into training and testing sets based on the specified training size.

        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame containing the sales data.
        train_size : float, optional
            The proportion of the data to be used for training (default is 0.8).

        Returns:
        --------
        tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the training and testing DataFrames.
        """

        train_size = int(len(data) * train_size)

        train_data = data[:train_size]
        test_data = data[train_size:]

        return train_data, test_data

    def excel(self, data, path: str) -> None:
        """
        Converts the data to an Excel file and saves it to the specified path.
        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame to be converted and saved.
        path : str
            The path where the Excel file will be saved.
        --------
        Applied: sa.convert_excel(data, "data.xlsx")
        --------
        Returns:
        --------
        """
        data.to_excel(path, index=False)

    def order_dataset_by_date(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Orders the dataset by date in ascending order.
        Parameters:
        -----------
        data : pd.DataFrame
            The input DataFrame to be ordered by date.
        Returns:
        --------
        pd.DataFrame
            The DataFrame ordered by date in ascending order.
        """
        data = data.sort_values(by="date", ascending=True)
        return data

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

    def __clean__(self):
        """clean the console screen"""
        os.system("clear")
        return None
