import pandas as pd
from src.config import MAPPINGS_PATH


def create_id_map(df_original: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping between profile_id and anonymous integer row index.
    
    WHY: The model never sees profile_id (it is a PII identifier).
    Instead it works with integer row indices 0..N-1.
    This mapping file is the only bridge between a model output
    (a row index) and a displayable profile (full details from df_original).
    
    Saves the mapping to data/mappings/anon_id_map.csv.
    Returns the mapping dataframe.
    """
    id_map = pd.DataFrame({
        "anon_index": range(len(df_original)),
        "profile_id": df_original["profile_id"].values
    })
    
    id_map.to_csv(MAPPINGS_PATH, index=False)
    print(f"✓ ID map saved → {MAPPINGS_PATH}")
    print(f"  {len(id_map):,} profiles mapped")
    return id_map


def lookup_profile(anon_index: int,
                   df_original: pd.DataFrame) -> pd.Series:
    """
    Given an anonymous row index returned by the model,
    return the full profile row from df_original for display.
    
    This is the function called in notebook 03 when showing
    recommendation results to the user.
    """
    return df_original.iloc[anon_index]