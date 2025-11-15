from pathlib import Path
import pandas as pd

# Path to this file
FILE_DIR = Path(__file__).resolve().parent

# Project root = go up two levels (src/data → src → ENERGYFORECASTING)
ROOT = FILE_DIR.parents[1]

# Path to Excel file
DATA_FILE = ROOT / "Data" / "20251111_JUNCTION_training.xlsx"
FORTUM_DATA_FILE = ROOT / "Data" / "20251111_JUNCTION_training.xlsx"
OULU_DATA_FILE  = ROOT / "Data" / "Oulu Oulunsalo Pellonpää- 1.1.2021 - 30.9.2024_eec95dce-6cb6-4b17-a1c3-383716a90676.xlsx"
POHJOISPOHJANMAA_DATA_FILE_1 = ROOT / "Data" / "Pohjois-Pohjanmaa_Others Pyhäjärvi Ojakylä- 1.1.2021 - 30.9.2024_dcde386e-2473-430e-a8d5-f30bb9f2f1b7.xlsx"
POHJOISPOHJANMAA_DATA_FILE_2 = ROOT / "Data" / "Pohjois-Pohjanmaa_others Taivalkoski kirkonkylä- 1.1.2021 - 30.9.2024_f5f624f3-de95-4591-b6c3-49a9bc894411.xlsx"
ROVANIEMI_DATA_FILE = ROOT / "Data" / "Rovaniemi rautatieasema- 1.1.2021 - 30.9.2024_c48659cf-1f90-4e7e-be36-72d4cf19fcaa.xlsx"
LAPPI_DATA_FILE1 = ROOT / "Data" / "Lappi Inari Ivalo lentoasema- 1.1.2021 - 30.9.2024_7471801b-137f-45bf-86b9-292905334ccf.xlsx"
LAPPI_DATA_FILE_2 = ROOT / "Data" / "Lappi Sodankylä Tähtelä- 1.1.2021 - 30.9.2024_b060c208-3e7b-4006-861f-78647f07b66f.xlsx"
ESPOO_TAPIOLA_DATA_FILE = ROOT / "Data" / "Espoo Tapiola_ 1.1.2021 - 30.9.2024_bc039f9b-3b44-40e0-a7fe-9aeb036afb8b.xlsx"
MIKKELI_DATA_FILE = ROOT / "Data" / "Etelä-Savo Mikkeli lentoasema_ 1.1.2021 - 30.9.2024_c9deb697-1706-4204-be28-478feaecd54e.xlsx"
HELSINKI_KAISANIEMI_DATA_FILE = ROOT / "Data" / "Helsinki Kaisaniemi_ 1.1.2021 - 30.9.2024_47f8e46c-0d9c-4d3a-8618-94bb4d2c342c.xlsx"
JYVASKYLA_DATA_FILE = ROOT / "Data" / "Jyväskylä lentoasema_ 1.1.2021 - 30.9.2024_ede2b8a6-ec86-4ed8-8881-f2ae5097ae15.xlsx"
HAMEENLINNA_DATA_FILE = ROOT / "Data" / "Kanta-Häme Hämeenlinna Katinen_ 1.1.2021 - 30.9.2024_8ed2ac91-0020-402c-884f-b35326d3f431.xlsx"
LAHTI_DATA_FILE = ROOT / "Data" / "Lahti Sopenkorpi_ 1.1.2021 - 30.9.2024_b3600d58-f275-4722-ac12-bb4a2fc9411f.xlsx"
LAPPEENRANTA_DATA_FILE = ROOT / "Data" / "Lappeenranta lentoasema_ 1.1.2021 - 30.9.2024_84f4a144-0b6f-4ca7-8aea-4edd901d7f64.xlsx"
JOENSUU_DATA_FILE = ROOT / "Data" / "Liperi Joensuu lentoasema_ 1.1.2021 - 30.9.2024_67d8f935-5e17-4bbe-86b9-f71fc84e1b78.xlsx"
PIRKANMAA_DATA_FILE = ROOT / "Data" / "Pirkanmaa_Others Pirkkala Tampere lentoasema_ 1.1.2021 - 30.9.2024_0981d871-fcce-4256-b628-a5ae1a36c364.xlsx"
POHJANMAA_DATA_FILE = ROOT / "Data" / "Pohjanmaa Vaasa lentoasema_ 1.1.2021 - 30.9.2024_011c3848-c16d-466d-b925-c3bac452631b.xlsx"
POHJOISSAVO_DATA_FILE = ROOT / "Data" / "Pohjois-Savo Siilinjärvi Kuopio lentoasema_ 1.1.2021 - 30.9.2024_8d598bc1-14fd-424f-9913-622db6422968.xlsx"
PORI_DATA_FILE = ROOT / "Data" / "Pori rautatieasema_ 1.1.2021 - 30.9.2024_bab55314-eced-43bc-9460-e10fce05a118.xlsx"
TAMPERE_DATA_FILE = ROOT / "Data" / "Tampere Härmälä_ 1.1.2021 - 30.9.2024_5976ebcf-ef72-4cd8-ba24-7e9d31476fd4.xlsx"
VANTAA_DATA_FILE = ROOT / "Data" / "Vantaa Helsinki-Vantaan lentoasema_ 1.1.2021 - 30.9.2024_9fd70b98-5481-451c-9172-0d2531ddfab8.xlsx"
print("Loading:", DATA_FILE)

xls = pd.ExcelFile(DATA_FILE)

cons   = pd.read_excel(xls, "training_consumption")
groups = pd.read_excel(xls, "groups")
prices = pd.read_excel(xls, "training_prices")

print(cons.head())
print(groups.head())
print(prices.head())
