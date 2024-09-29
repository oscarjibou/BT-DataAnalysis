from utilities import *

raw_data = pd.read_excel("data/Datos_Market_copy.xlsx")

sales_analysis = SalesAnalysis(raw_data)

data = sales_analysis.data

# sales_analysis.plot_all_separate_flavour("brand-35")

sales_analysis.plot_flavours_by_brand("brand-35")
