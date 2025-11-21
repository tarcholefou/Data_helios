import os
import re
from io import BytesIO
from datetime import datetime

import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

DATA_DIR = "data"
HISTORY_TVA_FILE = os.path.join(DATA_DIR, "history_tva.csv")   # ventes issues des PDF
HISTORY_ABOS_FILE = os.path.join(DATA_DIR, "history_abos.csv") # abonnements / cartes issus des CSV
os.makedirs(DATA_DIR, exist_ok=True)

CATEGORIES_TVA = [
    "Abonnements / cartes",
    "Boissons & compléments alimentaires",
    "Vestimentaire & accessoires sport",
    "AUTRE",
]

MOIS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}


# =========================
# UTILS GÉNÉRAUX
# =========================

def to_float(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0.0
    s = str(x)
    s = s.replace("€", "").replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def to_int(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    try:
        return int(float(str(x).replace(",", ".").replace(" ", "")))
    except ValueError:
        return 0


def extract_period_from_text(text: str):
    """Extrait la période '01-10-2025 - 31-10-2025' si présente."""
    m = re.search(r"(\d{2}-\d{2}-\d{4})\s*-\s*(\d{2}-\d{2}-\d{4})", text)
    if not m:
        return None, None, None
    d1 = datetime.strptime(m.group(1), "%d-%m-%Y")
    d2 = datetime.strptime(m.group(2), "%d-%m-%Y")
    mois = f"{d1.year}-{d1.month:02d}"
    return mois, d1.date().isoformat(), d2.date().isoformat()


def sort_months(month_iterable):
    return sorted(month_iterable, key=lambda x: datetime.strptime(x, "%Y-%m"))


def format_mois_label(mois: str) -> str:
    """Transforme '2025-10' en 'Octobre 2025'."""
    dt = datetime.strptime(mois, "%Y-%m")
    return f"{MOIS_FR[dt.month]} {dt.year}"


def style_delta(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: green; font-weight: bold;"
    if val < 0:
        return "color: red; font-weight: bold;"
    return ""


# =========================
# CATEGORISATION TVA
# =========================

def categorize_product_tva(name: str):
    if not isinstance(name, str):
        name = str(name)
    n = name.lower()

    # Abonnements / cartes
    if "abonn" in n:
        return "Abonnements / cartes", "Abonnement"
    if "carte" in n or "prépayée" in n or "prepayee" in n:
        return "Abonnements / cartes", "Carte"
    if "drop in" in n or "drop-in" in n or "drop" in n or "open gym" in n:
        return "Abonnements / cartes", "Drop-in / visite"

    # Boissons & compléments
    patterns_boissons = [
        "nocco", "barebells", "fitaid", "vitamin well", "vitaminwell",
        "hydrate", "reload", "anti oxydant", "antioxydant",
        "omega", "oméga",
        "collagène", "collagene",
        "créatine", "creatine",
        "whey",
        "magnésium", "magnesium",
        "multi vitamines", "multivitamine"
    ]
    if any(p in n for p in patterns_boissons):
        return "Boissons & compléments alimentaires", "Boisson / complément"

    # Vestimentaire & accessoires sport
    patterns_vetements = [
        "t shirt", "t-shirt", "tee shirt", "tee-shirt",
        "genouillère", "genouillere",
        "ceinture",
        "bande de poignets", "bande", "bandes de force",
        "maniques", "manique"
    ]
    if any(p in n for p in patterns_vetements):
        return "Vestimentaire & accessoires sport", "Textile / accessoires"

    return "AUTRE", "AUTRE"


# =========================
# EXTRACTION PDF TVA
# =========================

def extract_sales_tables_from_pdf(file_obj: BytesIO, forced_month: str = None) -> pd.DataFrame:
    rows = []
    periode_debut = None
    periode_fin = None

    with pdfplumber.open(file_obj) as pdf:
        first_text = pdf.pages[0].extract_text() or ""
        mois_detecte, periode_debut, periode_fin = extract_period_from_text(first_text)

        # On utilise en priorité le mois choisi dans l'UI
        periode_mois = forced_month or mois_detecte
        if periode_mois is None:
            today = datetime.today()
            periode_mois = f"{today.year}-{today.month:02d}"

        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                if not t or len(t) < 2:
                    continue

                header = [c.strip() if c else "" for c in t[0]]
                header_lower = [h.lower() for h in header]

                if not any("désignation" in h or "designation" in h for h in header_lower):
                    continue
                if not any("quantité" in h or "quantite" in h for h in header_lower):
                    continue

                data_rows = t[1:]
                df = pd.DataFrame(data_rows, columns=header)

                # mapping colonnes
                colmap = {}
                for col in df.columns:
                    col_norm = col.lower().strip()
                    if "désignation" in col_norm or "designation" in col_norm:
                        colmap[col] = "designation"
                    elif "quantité" in col_norm or "quantite" in col_norm:
                        colmap[col] = "quantite"
                    elif "total ttc" in col_norm:
                        colmap[col] = "total_ttc"
                    elif "tva (%)" in col_norm or "tva%" in col_norm:
                        colmap[col] = "tva_pct"
                    elif "total tva" in col_norm:
                        colmap[col] = "total_tva"
                    elif "total ht" in col_norm:
                        colmap[col] = "total_ht"
                    else:
                        colmap[col] = col_norm

                df = df.rename(columns=colmap)

                if "quantite" in df.columns:
                    df["quantite"] = df["quantite"].apply(to_int)
                else:
                    df["quantite"] = 0

                for c in ["total_ttc", "total_tva", "total_ht"]:
                    if c in df.columns:
                        df[c] = df[c].apply(to_float)
                    else:
                        df[c] = 0.0

                if "tva_pct" in df.columns:
                    df["tva_pct"] = df["tva_pct"].apply(to_float)
                else:
                    df["tva_pct"] = 0.0

                df["mois"] = periode_mois
                df["periode_debut"] = periode_debut
                df["periode_fin"] = periode_fin

                rows.append(df)

    if not rows:
        return pd.DataFrame()

    full_df = pd.concat(rows, ignore_index=True)
    full_df = full_df[full_df["designation"].notna()]
    full_df = full_df[full_df["designation"].str.strip() != ""]

    cat_main = []
    cat_sub = []
    for name in full_df["designation"]:
        cmain, csub = categorize_product_tva(name)
        cat_main.append(cmain)
        cat_sub.append(csub)
    full_df["categorie"] = cat_main
    full_df["sous_categorie"] = cat_sub

    return full_df


# =========================
# IMPORT / HISTORIQUE TVA
# =========================

def load_history_tva() -> pd.DataFrame:
    cols = [
        "mois", "periode_debut", "periode_fin",
        "designation", "quantite",
        "total_ttc", "total_tva", "total_ht", "tva_pct",
        "categorie", "sous_categorie",
    ]
    if os.path.exists(HISTORY_TVA_FILE):
        try:
            df = pd.read_csv(HISTORY_TVA_FILE)
            if df.empty:
                return pd.DataFrame(columns=cols)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=cols)
    else:
        return pd.DataFrame(columns=cols)


def save_history_tva(df_history: pd.DataFrame):
    df_history.to_csv(HISTORY_TVA_FILE, index=False)


def build_month_summary_tva(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist.copy()
    res = df.groupby("mois").agg(
        CA_total=("total_ttc", "sum"),
        Qt_total=("quantite", "sum"),
    ).reset_index()

    for cat in CATEGORIES_TVA:
        col_name = f"CA_{cat}"
        tmp = (
            df[df["categorie"] == cat]
            .groupby("mois")["total_ttc"]
            .sum()
            .rename(col_name)
        )
        res = res.merge(tmp, on="mois", how="left")

    res = res.sort_values("mois", key=lambda s: s.map(la_
