import pandas as pd
df_original = pd.read_csv("data/raw/profiles.csv")
print(f"Unique companies : {df_original['current_company'].nunique()}")
print(df_original['current_company'].value_counts().head(20).to_string())