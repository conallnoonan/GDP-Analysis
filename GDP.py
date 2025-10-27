import pandas as pd # NOQA F401

# Read both CSV files in
df_gdp = pd.read_csv(
    "/Users/conallnoonan/Documents/GDP_Analysis/GDP_Predictions_2026.csv"
    )
df_pop = pd.read_csv("/Users/conallnoonan/Documents/GDP_Analysis/Pop_2024.csv")

# Merge on 'Country'
df = pd.merge(df_gdp, df_pop, on="Country", how="inner")
print(df.head())
print(df.ndim)
print(df.shape)
