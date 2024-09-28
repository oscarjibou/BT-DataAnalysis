from utilities import *

(raw_data, data) = cleaning_data("data/Datos_Market_copy.xlsx")

variantF = data["variant"] == "flavoured"
variantS = data["variant"] == "standard"
variantL = data["variant"] == "light"
variantV = data["variant"] == "vegan"

supermarketA = data["supermarket"] == "supermarket-A"
supermarketB = data["supermarket"] == "supermarket-B"
supermarketC = data["supermarket"] == "supermarket-C"
supermarketD = data["supermarket"] == "supermarket-D"
