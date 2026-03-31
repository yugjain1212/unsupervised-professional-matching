import pandas as pd
from src.config import PII_COLS, DROP_COLS


def drop_pii_and_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all PII columns and columns flagged for dropping in config.
    
    PII_COLS  : name, email, profile_id — identifiers, no semantic value
    DROP_COLS : current_company, source, headline, about — 
                high cardinality / zero variance / low signal
    
    This function is the formal, pipeline-level drop.
    The EDA notebook only masked these columns temporarily in memory.
    
    Returns a copy — never modifies the input dataframe.
    """
    cols_to_drop = [
        c for c in PII_COLS + DROP_COLS
        if c in df.columns
    ]
    
    print(f"Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
    return df.drop(columns=cols_to_drop).copy()