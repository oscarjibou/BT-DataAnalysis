import sys

sys.path.append("..")

from utilities import *

warnings.filterwarnings("ignore")

raw_data = pd.read_excel("../data/Datos_Market_copy.xlsx")

sa = SalesAnalysis(raw_data)

data = sa.data

data["date"] = pd.to_datetime(data["date"])

data = data[data["brand"].isin(["brand-35", "brand-14", "brand-15"])]

if False:
    data["series_id"] = (
        data["brand"].astype(str)
        + "_"
        + data["supermarket"].astype(str)
        + "_"
        + data["variant"].astype(str)
        + "_"
        + data["pack.size"].astype(str)
    )

    # Ordenar por series_id y date
    data = data.sort_values(["series_id", "date"]).reset_index(drop=True)

    print(f"Número de series únicas: {data['series_id'].nunique()}")
    print(f"Series ID ejemplo: {data['series_id'].iloc[0]}")
    print(f"Rango de fechas: {data['date'].min()} a {data['date'].max()}")


def plot_every_series(data, plot_all=False, filter_brand="brand-14", **kwargs):
    """
    Plots every series in the data.
    If plot_all is True, it will plot all series.
    If plot_all is False, it will plot the series filtered by brand.
    If data['series_id'] is not a column, it will create it.
    """
    # ============================================================================
    # PREPARACIÓN DE LOS DATOS
    # ============================================================================
    # Crear series_id si no existe o si tiene valores NaN
    if "series_id" not in data.columns or data["series_id"].isna().any():

        data["series_id"] = (
            data["brand"].astype(str)
            + "_"
            + data["supermarket"].astype(str)
            + "_"
            + data["variant"].astype(str)
            + "_"
            + data["pack.size"].astype(str)
        )

        # Ordenar por series_id y date
        data = data.sort_values(["series_id", "date"]).reset_index(drop=True)

    # ============================================================================
    # CONFIGURACIÓN DE FILTROS - Modifica estas variables según necesites
    # ============================================================================

    # Opción 2: Filtrar por marca específica (usar si plot_all = False)
    # Puedes poner: "brand-35", "brand-14", "brand-15", o None para no filtrar
    # filter_brand = kwargs.get('filter_brand', None)  # Cambia a None o a otra marca según necesites
    filter_brand = filter_brand  # Cambia a None o a otra marca según necesites

    # Puedes poner valores específicos o None para no filtrar
    filter_supermarket = kwargs.get(
        "filter_supermarket", None
    )  # Ejemplo: "supermarket-A", "supermarket-B", etc.
    filter_variant = kwargs.get(
        "filter_variant", None
    )  # Ejemplo: "flavoured", "standard", "light", "vegan"
    filter_pack_size = kwargs.get(
        "filter_pack_size", None
    )  # Ejemplo: "0 - 350 GR", "351 - 500 GR", etc.

    # ============================================================================
    # CÓDIGO DE VISUALIZACIÓN
    # ============================================================================

    # Aplicar filtros según la configuración
    data_to_plot = data.copy()

    if not plot_all:
        # Aplicar filtros si están definidos
        if filter_brand is not None:
            data_to_plot = data_to_plot[data_to_plot["brand"] == filter_brand]
            print(f"Filtrado por marca: {filter_brand}")

        if filter_supermarket is not None:
            data_to_plot = data_to_plot[
                data_to_plot["supermarket"] == filter_supermarket
            ]
            print(f"Filtrado por supermercado: {filter_supermarket}")

        if filter_variant is not None:
            data_to_plot = data_to_plot[data_to_plot["variant"] == filter_variant]
            print(f"Filtrado por variante: {filter_variant}")

        if filter_pack_size is not None:
            data_to_plot = data_to_plot[data_to_plot["pack.size"] == filter_pack_size]
            print(f"Filtrado por tamaño: {filter_pack_size}")

    # Obtener las series únicas después de aplicar filtros
    series_ids = data_to_plot["series_id"].unique()
    n_series = len(series_ids)

    print(f"\nTotal de series a plotear: {n_series}")

    if n_series == 0:
        print("¡No hay series que cumplan los filtros seleccionados!")
    else:
        # Obtener el rango de fechas completo para el eje x (mismo para todas)
        date_min = data_to_plot["date"].min()
        date_max = data_to_plot["date"].max()

        # Crear un gráfico separado para cada serie
        for idx, series_id in enumerate(series_ids):

            print(f"Gráfico {idx+1}/{n_series} completado: {series_id}")

            # Crear una nueva figura para cada serie
            fig, ax = plt.subplots(figsize=(12, 6))

            # Filtrar datos para esta serie
            series_data = data_to_plot[
                data_to_plot["series_id"] == series_id
            ].sort_values("date")

            # Graficar la serie temporal
            ax.plot(
                series_data["date"],
                series_data["volume.sales"],
                linewidth=2,
                marker="o",
                markersize=4,
            )

            # Configurar el eje x con el mismo rango para todas las series
            ax.set_xlim([date_min, date_max])

            # Configurar título y etiquetas
            ax.set_title(f"Serie: {series_id}", fontsize=14, fontweight="bold")
            ax.set_xlabel("Fecha", fontsize=12)
            ax.set_ylabel("Volume Sales", fontsize=12)
            ax.grid(True, alpha=0.3)

            # Rotar las etiquetas del eje x
            plt.xticks(rotation=45, ha="right")

            # Ajustar el layout
            plt.tight_layout()

            # Mostrar el gráfico
            plt.show()

        print(f"\n✅ Se han generado {n_series} gráficos individuales")


def series_less_than_36(data: pd.DataFrame) -> pd.DataFrame:
    """
    Comprueba que series no tienen datos de los 3 años

    Parameters:
    data (pd.DataFrame): Dataframe con los datos

    Returns (pd.DataFrame):
    pd.DataFrame: Dataframe con las series con menos de 36 datos ordenadas por número de datos
    """
    date_min = data["date"].min()
    date_max = data["date"].max()

    # Comprabar que series no tienen datos de los 3 años
    result_df = pd.DataFrame(columns=["series_id", "length"])
    for series_id in data["series_id"].unique():
        series_data = data[data["series_id"] == series_id]
        if len(series_data) < 36:
            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        {"series_id": [series_id], "length": [len(series_data)]}
                    ),
                ],
                ignore_index=True,
            )
    return result_df.sort_values(by="length").reset_index(drop=True)
