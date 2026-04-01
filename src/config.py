import os

# ── Project root ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DATA_PATH = os.path.join(ROOT, "data", "raw", "profiles.csv")
PROCESSED_PATH = os.path.join(ROOT, "data", "processed", "X_processed.npy")
MAPPINGS_PATH = os.path.join(ROOT, "data", "mappings", "anon_id_map.csv")
PLOTS_DIR = os.path.join(ROOT, "outputs", "plots")

# ── Create directories ─────────────────────────────────────────────────────
for _path in [
    PLOTS_DIR,
    os.path.join(ROOT, "data", "processed"),
    os.path.join(ROOT, "data", "mappings"),
    os.path.join(ROOT, "outputs", "models"),
    os.path.join(ROOT, "outputs", "clusters"),
]:
    os.makedirs(_path, exist_ok=True)

# ── Columns ────────────────────────────────────────────────────────────────
PII_COLS = ["name", "email", "profile_id"]

DROP_COLS = [
    "source",  # constant value ("synthetic") — zero variance
    "headline",  # free text, redundant with role + skills
    "about",  # generic boilerplate, low signal
]

# Stored as Python literal strings — use ast.literal_eval, NOT json.loads
AST_COLS = ["skills", "experience", "education", "goals", "needs", "can_offer"]

# ── Categorical encodings ──────────────────────────────────────────────────
SENIORITY_ORDER = ["entry", "mid", "senior", "executive"]
SENIORITY_MAP = {level: i for i, level in enumerate(SENIORITY_ORDER)}
# → {"entry": 0, "mid": 1, "senior": 2, "executive": 3}

REMOTE_MAP = {"onsite": 0, "hybrid": 1, "remote": 2}

# current_role — only 24 unique values in this dataset (synthetic vocabulary)
# Direct one-hot encoding is applied — no bucketing required
# ROLE_BUCKETS removed after EDA confirmed low cardinality

# ── Feature engineering constants ──────────────────────────────────────────
N_SKILLS = 100  # top N skills for multi-hot encoding
N_TFIDF_TERMS = 50  # top N terms from TF-IDF on intent fields
PCA_COMPONENTS = 30  # increased to retain more variance (target ~35-40%)
K_RANGE = range(2, 31)  # test K from 2 to 20
