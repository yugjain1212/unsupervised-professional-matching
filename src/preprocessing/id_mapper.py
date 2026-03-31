"""
Purpose of this file 
the dataset contains real looking identifiers like profile_id and personal info
if these reach the model it could cluster people by identity  instead of professional  attributes 
which is exactly what we want to avoid

this file does 2 things:
   1. Create a mapping table : profile_id -> anonymous row index (0,1,2,...)
   and saves it to a csv file name data/anon_id_map.csv so we can trace results
   back to real profiles after  the model runs 
   2. Drops all PII (Personally identifiable info) columns from the dataframe so the model never sees them 

"""

import pandas as pd
import os

from src.config import MAPPINGS_PATH, PII_COLS

def build_anon_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    After clustering we need to show real profiles details to the user but duing 
    modelling the model must be compeletly blind to identity the mapping csv acts as lookup
    table that lives outside the ml pipeline - so we use it display time 

    How it works :
       df.reset_index(drop ==True) gives every row a clean integer index starting from
       0 that integer becomes the anonymous ID
       we store : anon_id (the new integer index)-> profile_id (the real integer) 

    """

    df =  df.reset_index(drop=True)
    if "profile_id" not in df.columns:
        raise ValueError("Expected 'profile_id' column not found in DataFrame.")
    
    mapping_df = pd.DataFrame({
        "anon_id" : df.index,               #integer row posi
        "profile_id" : df["profile_id"],    #original profile id
    })

    mapping_df.to_csv(MAPPINGS_PATH,index = False)

    print(f"[id_mapper] Mapping saved -> {MAPPINGS_PATH}")
    print(f"[id_mapper] Total profiles Mapped : {len(mapping_df):,}")
    print(f"[id_mapper] sample(first 5 Rows)")
    print(mapping_df.head().to_string(index=False))
    return df


def Drop_pii(df: pd.DataFrame)->pd.DataFrame:
    """
    Drops all columns that are consider PII from the dataFrame 

    PII_COLS is defined in cofig.py as:
       ['name','email','profile_id']
    
    we import it from config so if the list ever changes this file automatically picks up the change
    """
    cols_to_drop = [col for col in PII_COlS if col in df.columns]
    
    missing = set(PII_COLS) - set(cols_to_drop)
    if missing:
        print("[id_mapper] Warning - these PII cols not found in (skipped): {missing}")

    df = df.drop(columns= cols_to_drop)
    
    print(f"\n[id_mapper] Dropped PII cols dropped : {cols_to_drop}")
    print(f"[id_mapper] Remaining cols : {df.shape[1]}")
    print(f"[id mapper] remaining cols : {df.columns.tolist()}")

    return df

def run(df: pd.DataFrame)->pd.DataFrame:

    print("*"*60)
    print("[id_mapper] Starting ID Mapping and PII Dropping")
    print("*"*60)

    df=build_anon_mapping(df)
    df= Drop_pii(df)

    print(f"\n[id_mapper] Completed ID Mapping and PII Dropping")
    print(f"[id_mapper] Final DataFrame shape : {df.shape}")

    print("*"*60)
    return df


if __name__ == "__main__":
    from src.config import RAW_DATA_PATH
    df_raw= pd.read_csv(RAW_DATA_PATH)
    print(f"[id_mapper] Loaded : {df_raw.shape[0]:,} rows and {df_raw.shape[1]} cols")

    df_clean= run(df_raw)

    print(f"[id_mapper] final dataframe head")
    print(df_clean.head().to_string(index=False))
    print("\n[id_mapper] Test complete. Check data/mappings/anon_id_map.csv")