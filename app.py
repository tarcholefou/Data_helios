import os
import re
from io import BytesIO
from datetime import datetime, date

import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt
import altair as alt


# =========================
# CONFIG
# =========================

DATA_DIR = "data"
HISTORY_TVA_FILE = os.path.join(DATA_DIR, "history_tva.csv")   # ventes issues des PDF TVA
HISTORY_ABOS_FILE = os.path.join(DATA_DIR, "history_abos.csv") # abonnements / cartes issus du CSV
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
    """Extrait la période '01-10-2025 - 31-10-2025' si présente dans le PDF TVA."""
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
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: green; font-weight: bold;"
    if v < 0:
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
    """
    Lit un PDF Helios CrossFit - Rapport TVA et renvoie un DataFrame
    avec toutes les lignes de ventes (OFFRES + PRODUITS), catégorisées.
    """
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
# HISTORIQUE TVA
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

    res = res.sort_values("mois", key=lambda s: s.map(lambda x: datetime.strptime(x, "%Y-%m")))
    res = res.fillna(0.0)
    return res


# =========================
# ABONNEMENTS / CARTES (CSV)
# =========================

def parse_date_creation(raw):
    """
    Convertit les dates du CSV en date Python.
    Exemples gérés :
    - '29/09/25 à 21:37'
    - '10/11/2025 17:33'
    - '2025-11-10'
    """
    if pd.isna(raw):
        return None

    s = str(raw).strip()

    if "à" in s:
        s = s.split("à")[0].strip()
    elif " " in s:
        s = s.split(" ")[0].strip()

    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            d = datetime.strptime(s, fmt)
            return d.date()
        except ValueError:
            continue

    return None


def classify_contrat(offre: str):
    """
    Retourne (type_contrat, sous_type)
    type_contrat ∈ {ABONNEMENT, CARTE_10, EVENT, EXCLU}
    """
    if not isinstance(offre, str):
        offre = str(offre)

    s = offre.lower().strip()

    # Événements à exclure
    if any(kw in s for kw in ["soirée", "soiree", "inauguration", "raclette", "event"]):
        return ("EVENT", offre)

    # Drop in = séance ponctuelle, on exclut
    if "drop" in s:
        return ("EXCLU", offre)

    # Liberté = carnet 10 séances
    if "liberté" in s or "liberte" in s:
        return ("CARTE_10", "Carnet 10 séances")

    # Abonnements
    abo_keywords = [
        "essentiel",
        "evolution",
        "premium",
        "hyrox",
        "1x semaine",
        "1 x semaine",
        "1xsemaine",
        "ascension",
    ]
    if any(k in s for k in abo_keywords):
        return ("ABONNEMENT", offre)

    # Par défaut : exclu
    return ("EXCLU", offre)


def extract_abos_from_csv(file_obj: BytesIO) -> pd.DataFrame:
    # Lecture brute
    df_raw = pd.read_csv(file_obj)

    # Nettoyage des noms de colonnes
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # Détection souple de la colonne date de création
    date_col = None
    for c in df_raw.columns:
        lc = c.lower()
        if "date" in lc and ("cré" in lc or "crea" in lc or "crÃ©" in lc):
            date_col = c
            break
    if date_col is None:
        for c in df_raw.columns:
            if "date" in c.lower():
                date_col = c
                break
    if date_col is None:
        date_col = df_raw.columns[0]

    # Mapping des colonnes utiles
    colmap = {
        "Prénom": "prenom",
        "Nom": "nom",
        "Email": "email",
        "Téléphone": "telephone",
        "Offre": "offre",
        "Date de début": "date_debut",
        "Date de fin": "date_fin",
        "Statut": "statut",
        "Méthode de paiement": "methode_paiement",
        "Prix de l'offre": "prix_offre",
        "Prix personnalisé": "prix_perso",
        "Reconduction": "reconduction",
        "Paiement comptant": "paiement_comptant",
        "Entrées restantes": "entrees_restantes",
        "Entrées max": "entrees_max",
    }

    df = df_raw.rename(columns={k: v for k, v in colmap.items() if k in df_raw.columns})

    # Date de création
    df["date_creation"] = df_raw[date_col].apply(parse_date_creation)
    df = df[~df["date_creation"].isna()]

    if df.empty:
        return df

    df["mois_creation"] = df["date_creation"].apply(lambda d: f"{d.year}-{d.month:02d}")

    # Classification contrat
    df["offre"] = df["offre"].astype(str)
    types = df["offre"].apply(classify_contrat)
    df["type_contrat"] = types.apply(lambda x: x[0])
    df["sous_type"] = types.apply(lambda x: x[1])

    # Prix
    df["prix_offre"] = df.get("prix_offre", 0).apply(to_float)
    df["prix_perso"] = df.get("prix_perso", 0).apply(to_float)
    df["prix_effectif"] = df.apply(
        lambda r: r["prix_perso"] if r["prix_perso"] > 0 else r["prix_offre"],
        axis=1,
    )

    # Entrées
    df["entrees_restantes"] = df.get("entrees_restantes", 0).apply(to_int)
    df["entrees_max"] = df.get("entrees_max", 0).apply(to_int)

    return df


def load_history_abos() -> pd.DataFrame:
    if os.path.exists(HISTORY_ABOS_FILE):
        try:
            df = pd.read_csv(HISTORY_ABOS_FILE, parse_dates=["date_creation"])
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def save_history_abos(df: pd.DataFrame):
    df.to_csv(HISTORY_ABOS_FILE, index=False)


# =========================
# CALCULS DÉRIVÉS
# =========================

def add_deltas(df: pd.DataFrame, col_base: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{col_base}_prec"] = df[col_base].shift(1)
    df[f"Delta_{col_base}"] = df[col_base] - df[f"{col_base}_prec"]
    df[f"Delta_%_{col_base}"] = (df[f"Delta_{col_base}"] / df[f"{col_base}_prec"] * 100)
    df[f"Delta_%_{col_base}"] = df[f"Delta_%_{col_base}"].replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    return df


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(page_title="Helios – Reporting CA", layout="wide")
st.title("Helios CrossFit – Outil de reporting CA")


# ---------- IMPORTS ----------

st.markdown(
    """
## Import de données

### 1) Import CA (rapports TVA – PDF)

1. Exporter le **rapport TVA PDF** du mois depuis ton logiciel.  
2. Choisir le **mois concerné** (année + mois).  
3. Uploader le PDF.  
4. Cliquer sur **Importer / remplacer ce mois (TVA)**.  

👉 Le mois sélectionné est **remplacé** dans l'historique TVA, les autres mois ne bougent pas.
"""
)

annee_courante = datetime.today().year
annees = list(range(2022, annee_courante + 1))

col_a, col_m = st.columns(2)
with col_a:
    annee_select = st.selectbox(
        "Année du rapport TVA à importer",
        options=annees,
        index=len(annees) - 1,
        key="import_tva_year",
    )
with col_m:
    mois_num = st.selectbox(
        "Mois du rapport TVA à importer",
        options=list(MOIS_FR.keys()),
        format_func=lambda x: MOIS_FR[x],
        key="import_tva_month",
    )

mois_import_tva = f"{annee_select}-{mois_num:02d}"

uploaded_pdf = st.file_uploader("Uploader le rapport TVA (PDF)", type=["pdf"], key="pdf_uploader")
import_tva_clicked = st.button("Importer / remplacer ce mois (TVA)", key="import_tva_button")

if import_tva_clicked:
    if uploaded_pdf is None:
        st.error("Merci de choisir d'abord un fichier PDF.")
    else:
        with st.spinner("Extraction des données du PDF..."):
            df_new_tva = extract_sales_tables_from_pdf(BytesIO(uploaded_pdf.read()), forced_month=mois_import_tva)

        if df_new_tva.empty:
            st.error("Impossible d'extraire des ventes depuis ce PDF. Vérifie le format.")
        else:
            df_hist_tva_old = load_history_tva()
            nb_ancien = len(df_hist_tva_old[df_hist_tva_old["mois"] == mois_import_tva])
            df_autres = df_hist_tva_old[df_hist_tva_old["mois"] != mois_import_tva]

            df_hist_new = pd.concat([df_autres, df_new_tva], ignore_index=True)
            save_history_tva(df_hist_new)

            ca_new = df_new_tva["total_ttc"].sum()

            st.success(
                f"{len(df_new_tva)} lignes importées pour {format_mois_label(mois_import_tva)} "
                f"(remplace {nb_ancien} lignes précédentes). CA : {ca_new:.2f} €."
            )
            st.dataframe(df_new_tva.head(50))

st.markdown(
    """
### 2) Import abonnements / cartes (CSV)

1. Exporter le fichier **inscriptions (.csv)** depuis ton logiciel.  
2. Uploader le fichier.  
3. Cliquer sur **Importer / remplacer l'historique inscriptions**.  

👉 L’historique abonnements/cartes est **entièrement reconstruit** à partir de ce fichier.
"""
)

csv_file = st.file_uploader("Uploader le fichier d’inscriptions (CSV)", type=["csv"], key="csv_abos_uploader")
import_abos_clicked = st.button("Importer / remplacer l’historique inscriptions", key="import_abos_button")

if import_abos_clicked:
    if csv_file is None:
        st.error("Merci de choisir d'abord un fichier CSV.")
    else:
        with st.spinner("Traitement du CSV..."):
            df_abos_new = extract_abos_from_csv(BytesIO(csv_file.read()))

        if df_abos_new.empty:
            st.error("Aucune inscription exploitable trouvée dans ce fichier.")
        else:
            save_history_abos(df_abos_new)

            nb_rows = len(df_abos_new)
            mois_couverts = sorted(df_abos_new["mois_creation"].unique())
            st.success(
                f"{nb_rows} inscriptions importées. Période couverte : "
                f"{format_mois_label(mois_couverts[0])} → {format_mois_label(mois_couverts[-1])}."
            )
            st.dataframe(df_abos_new.head(50))

st.markdown("---")

# ---------- CHARGEMENT HISTORIQUES ----------

df_hist_tva = load_history_tva()
if df_hist_tva.empty:
    st.warning("Aucune donnée TVA (PDF) n’a encore été importée.")
    st.stop()

df_hist_tva["mois"] = df_hist_tva["mois"].astype(str)
df_hist_tva["total_ttc"] = df_hist_tva["total_ttc"].astype(float)
df_hist_tva["quantite"] = df_hist_tva["quantite"].astype(int)
mois_dispo_tva = sort_months(df_hist_tva["mois"].unique())
summary_tva = build_month_summary_tva(df_hist_tva)

df_abos = load_history_abos()
has_abos = not df_abos.empty
if has_abos:
    df_abos["mois_creation"] = df_abos["mois_creation"].astype(str)


# =========================
# TABS
# =========================

tab_dash, tab_mensuel, tab_comp, tab_detail = st.tabs(
    ["📊 Dashboard Direction", "📅 Vue mensuelle", "📈 Comparaison mensuelle", "🔍 Détail produits / abonnements"]
)

# -------------------------------------------------------------------
# TAB 0 : DASHBOARD DIRECTION
# -------------------------------------------------------------------
with tab_dash:
    st.subheader("Synthèse globale – Dashboard Direction")

    # Mois de référence = dernier mois dispo par défaut
    mois_ref = st.selectbox(
        "Mois analysé",
        options=mois_dispo_tva,
        index=len(mois_dispo_tva) - 1,
        format_func=format_mois_label,
        key="dash_mois_ref",
    )

    idx_ref = mois_dispo_tva.index(mois_ref)
    mois_prev = mois_dispo_tva[idx_ref - 1] if idx_ref > 0 else None

    row_ref = summary_tva[summary_tva["mois"] == mois_ref].iloc[0]
    row_prev = summary_tva[summary_tva["mois"] == mois_prev].iloc[0] if mois_prev else None

    ca_ref = row_ref["CA_total"]
    qt_ref = row_ref["Qt_total"]
    ca_prev = row_prev["CA_total"] if row_prev is not None else None

    delta_ca_abs = None
    delta_ca_pct = None
    if ca_prev and ca_prev != 0:
        delta_ca_abs = ca_ref - ca_prev
        delta_ca_pct = (delta_ca_abs / ca_prev) * 100

    # CA par catégorie TVA
    ca_abos_cartes_tva = row_ref.get("CA_Abonnements / cartes", 0.0)
    ca_boissons = row_ref.get("CA_Boissons & compléments alimentaires", 0.0)
    ca_vesti = row_ref.get("CA_Vestimentaire & accessoires sport", 0.0)
    ca_autre_tva = row_ref.get("CA_AUTRE", 0.0)
    ca_boutique = ca_boissons + ca_vesti + ca_autre_tva

    # Abonnements / cartes (CSV) sur mois_ref
    if has_abos:
        df_mois_abos = df_abos[df_abos["mois_creation"] == mois_ref]
        df_abos_only = df_mois_abos[df_mois_abos["type_contrat"] == "ABONNEMENT"]
        df_cartes10 = df_mois_abos[df_mois_abos["type_contrat"] == "CARTE_10"]
        nb_abos = len(df_abos_only)
        nb_cartes10 = len(df_cartes10)
        ca_abos_csv = df_abos_only["prix_effectif"].sum()
        ca_cartes10_csv = df_cartes10["prix_effectif"].sum()
    else:
        df_mois_abos = pd.DataFrame()
        df_abos_only = pd.DataFrame()
        df_cartes10 = pd.DataFrame()
        nb_abos = nb_cartes10 = 0
        ca_abos_csv = ca_cartes10_csv = 0.0

    # KPI haut de page
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "CA total (TVA)",
        f"{ca_ref:,.2f} €".replace(",", " "),
        None if delta_ca_abs is None else f"{delta_ca_abs:+.0f} € ({delta_ca_pct:+.1f} %)",
    )
    col2.metric("Quantités vendues (TVA)", int(qt_ref))
    col3.metric("CA abonnements (CSV)", f"{ca_abos_csv:,.2f} €".replace(",", " "))
    col4.metric("CA carnets 10 (CSV)", f"{ca_cartes10_csv:,.2f} €".replace(",", " "))

    st.markdown("---")

    # GRAPH 1 – CA total vs mois précédent
    st.markdown("### CA total – comparaison rapide")

    if mois_prev is None:
        st.info("Un seul mois de données disponible pour l’instant.")
    else:
        df_ca_comp = pd.DataFrame({
            "mois": [mois_prev, mois_ref],
            "mois_label": [format_mois_label(mois_prev), format_mois_label(mois_ref)],
            "CA": [ca_prev, ca_ref],
        })
        chart_ca = (
            alt.Chart(df_ca_comp)
            .mark_bar()
            .encode(
                x=alt.X("mois_label:N", title=None),
                y=alt.Y("CA:Q", title="CA total (€)"),
                tooltip=[alt.Tooltip("CA:Q", format=".2f"), "mois_label"],
            )
            .properties(height=260)
        )
        st.altair_chart(chart_ca, use_container_width=True)

    # GRAPH 2 – Structure du CA TVA (catégories)
    st.markdown("### Structure du CA (TVA)")

    df_pie = pd.DataFrame({
        "categorie": ["Abonnements / cartes", "Boissons & compléments", "Textile & autres"],
        "CA": [ca_abos_cartes_tva, ca_boissons, ca_vesti + ca_autre_tva],
    })

    col_pie, col_tab = st.columns((1, 1))
    with col_pie:
        fig, ax = plt.subplots(figsize=(4, 4))
        if df_pie["CA"].sum() > 0:
            ax.pie(df_pie["CA"], labels=df_pie["categorie"], autopct="%1.1f%%")
        ax.set_title(f"Répartition CA – {format_mois_label(mois_ref)}")
        st.pyplot(fig)
    with col_tab:
        if df_pie["CA"].sum() > 0:
            df_pie["% CA"] = (df_pie["CA"] / df_pie["CA"].sum() * 100).round(1)
        else:
            df_pie["% CA"] = 0.0
        st.dataframe(df_pie)

    st.markdown("---")

    # GRAPH 3 – Abonnements vs carnets 10 (CSV) – nombre et CA
    st.markdown("### Abonnements vs carnets 10 (CSV)")

    if not has_abos or df_mois_abos.empty:
        st.info("Aucune donnée CSV d’inscriptions pour ce mois.")
    else:
        # Nb contrats
        df_nb = pd.DataFrame({
            "type": ["Abonnements", "Carnets 10"],
            "Nombre": [nb_abos, nb_cartes10],
        })
        chart_nb = (
            alt.Chart(df_nb)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title=None),
                y=alt.Y("Nombre:Q", title="Nombre de contrats"),
                tooltip=["type", "Nombre"],
            )
            .properties(height=220)
        )

        # CA contrats
        df_ca = pd.DataFrame({
            "type": ["Abonnements", "Carnets 10"],
            "CA": [ca_abos_csv, ca_cartes10_csv],
        })
        chart_ca2 = (
            alt.Chart(df_ca)
            .mark_bar()
            .encode(
                x=alt.X("type:N", title=None),
                y=alt.Y("CA:Q", title="CA (€)"),
                tooltip=[alt.Tooltip("CA:Q", format=".2f"), "type"],
            )
            .properties(height=220)
        )

        col_nb, col_ca = st.columns(2)
        with col_nb:
            st.altair_chart(chart_nb, use_container_width=True)
        with col_ca:
            st.altair_chart(chart_ca2, use_container_width=True)

    st.markdown("---")

    # Analyse automatique simple
    st.markdown("### Analyse automatique (lecture rapide)")

    bullet_points = []

    if delta_ca_abs is not None:
        if delta_ca_abs > 0:
            bullet_points.append(
                f"- CA total en **hausse** de **{delta_ca_abs:.0f} €** ({delta_ca_pct:+.1f} %) vs mois précédent."
            )
        elif delta_ca_abs < 0:
            bullet_points.append(
                f"- CA total en **baisse** de **{abs(delta_ca_abs):.0f} €** ({delta_ca_pct:+.1f} %) vs mois précédent."
            )
        else:
            bullet_points.append("- CA total **stable** vs mois précédent.")

    # Catégorie la plus forte
    cat_values = {
        "Abonnements / cartes": ca_abos_cartes_tva,
        "Boissons & compléments": ca_boissons,
        "Textile & autres": ca_vesti + ca_autre_tva,
    }
    cat_sorted = sorted(cat_values.items(), key=lambda x: x[1], reverse=True)
    if cat_sorted and cat_sorted[0][1] > 0:
        bullet_points.append(
            f"- Catégorie principale : **{cat_sorted[0][0]}** ({cat_sorted[0][1]:.0f} €)."
        )

    if has_abos and not df_mois_abos.empty:
        total_contrats = nb_abos + nb_cartes10
        if total_contrats > 0:
            part_abos = nb_abos / total_contrats * 100
            bullet_points.append(
                f"- Répartition contrats : **{part_abos:.0f} % abonnements** / {100 - part_abos:.0f} % carnets 10."
            )

    if not bullet_points:
        st.write("Pas encore assez de données pour une analyse automatique.")
    else:
        for b in bullet_points:
            st.markdown(b)

# -------------------------------------------------------------------
# TAB 1 : VUE MENSUELLE
# -------------------------------------------------------------------
with tab_mensuel:
    st.subheader("Analyse d’un mois")

    mois_focus = st.selectbox(
        "Mois à analyser",
        options=mois_dispo_tva,
        index=len(mois_dispo_tva) - 1,
        format_func=format_mois_label,
        key="view_month_focus",
    )

    df_mois_tva = df_hist_tva[df_hist_tva["mois"] == mois_focus]

    # Mois précédent
    mois_sorted = mois_dispo_tva
    idx = mois_sorted.index(mois_focus)
    df_prev_tva = None
    if idx > 0:
        mois_prev2 = mois_sorted[idx - 1]
        df_prev_tva = df_hist_tva[df_hist_tva["mois"] == mois_prev2]

    ca_mois = df_mois_tva["total_ttc"].sum()
    qte_mois = df_mois_tva["quantite"].sum()

    ca_prev2 = df_prev_tva["total_ttc"].sum() if df_prev_tva is not None else None
    delta_ca_abs2 = None
    delta_ca_pct2 = None
    if ca_prev2 and ca_prev2 != 0:
        delta_ca_abs2 = ca_mois - ca_prev2
        delta_ca_pct2 = (delta_ca_abs2 / ca_prev2) * 100

    ca_cat_mois = (
        df_mois_tva.groupby("categorie", as_index=False)
        .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
        .sort_values("CA", ascending=False)
    )

    st.markdown(f"### Synthèse CA – {format_mois_label(mois_focus)} (TVA)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CA total", f"{ca_mois:,.2f} €".replace(",", " "))
    c2.metric("Quantités vendues", int(qte_mois))
    if delta_ca_abs2 is not None:
        c3.metric("Δ CA vs mois précédent", f"{delta_ca_abs2:+.0f} €", f"{delta_ca_pct2:+.1f} %")
    else:
        c3.metric("Δ CA vs mois précédent", "N/A", "N/A")
    c4.metric("Nb lignes (ventes)", len(df_mois_tva))

    st.markdown("#### Répartition du CA par catégorie (TVA)")

    col_pie, col_tab = st.columns((1, 1))
    with col_pie:
        fig, ax = plt.subplots(figsize=(4, 4))
        if not ca_cat_mois.empty:
            ax.pie(ca_cat_mois["CA"], labels=ca_cat_mois["categorie"], autopct="%1.1f%%")
        ax.set_title("CA par catégorie")
        st.pyplot(fig)
    with col_tab:
        if ca_mois > 0:
            ca_cat_mois["% CA"] = (ca_cat_mois["CA"] / ca_mois * 100).round(1)
        else:
            ca_cat_mois["% CA"] = 0.0
        st.dataframe(ca_cat_mois)

    st.markdown("---")

    # Abonnements / cartes pour ce mois (CSV)
    st.markdown(f"### Abonnements & carnets 10 – {format_mois_label(mois_focus)} (CSV)")

    if not has_abos:
        st.info("Aucune donnée CSV d’inscriptions importée pour l’instant.")
    else:
        df_mois_abos2 = df_abos[df_abos["mois_creation"] == mois_focus]

        if df_mois_abos2.empty:
            st.info("Aucune inscription pour ce mois dans le CSV.")
        else:
            df_abos_only2 = df_mois_abos2[df_mois_abos2["type_contrat"] == "ABONNEMENT"]
            df_cartes102 = df_mois_abos2[df_mois_abos2["type_contrat"] == "CARTE_10"]
            df_events2 = df_mois_abos2[df_mois_abos2["type_contrat"].isin(["EVENT", "EXCLU"])]

            nb_abos2 = len(df_abos_only2)
            nb_cartes102 = len(df_cartes102)
            ca_abos2 = df_abos_only2["prix_effectif"].sum()
            ca_cartes102 = df_cartes102["prix_effectif"].sum()

            cA, cB, cC, cD = st.columns(4)
            cA.metric("Abonnements vendus", nb_abos2)
            cB.metric("CA abonnements", f"{ca_abos2:,.2f} €".replace(",", " "))
            cC.metric("Carnets 10 vendus", nb_cartes102)
            cD.metric("CA carnets 10", f"{ca_cartes102:,.2f} €".replace(",", " "))

            # Répartition par type d'abo
            if not df_abos_only2.empty:
                st.markdown("#### Répartition des abonnements par type (ventes)")

                abo_type = (
                    df_abos_only2.groupby("sous_type", as_index=False)
                    .agg(
                        Nb=("offre", "count"),
                        CA=("prix_effectif", "sum"),
                    )
                    .sort_values("CA", ascending=False)
                )
                abo_type["% des abos"] = (abo_type["Nb"] / abo_type["Nb"].sum() * 100).round(1)

                col_bar_abo, col_tab_abo = st.columns((1.2, 1))
                with col_bar_abo:
                    chart = (
                        alt.Chart(abo_type)
                        .mark_bar()
                        .encode(
                            x=alt.X("sous_type:N", sort="-y", title="Type d’abonnement"),
                            y=alt.Y("Nb:Q", title="Nb d’abonnements"),
                            tooltip=["sous_type", "Nb", alt.Tooltip("CA:Q", format=".2f")]
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart, use_container_width=True)

                with col_tab_abo:
                    st.dataframe(abo_type)

            if not df_events2.empty:
                st.caption(
                    f"{len(df_events2)} inscription(s) marquée(s) comme EVENT / EXCLU "
                    "(soirées, drop-in, etc.) – exclues des stats abos/cartes."
                )

    st.markdown("---")

    # Top produits TVA par catégorie
    st.markdown("#### Top produits par catégorie (TVA)")

    cat_focus = st.selectbox(
        "Catégorie TVA (top produits)",
        options=CATEGORIES_TVA,
        key="month_cat_focus",
    )
    df_cat_focus = df_mois_tva[df_mois_tva["categorie"] == cat_focus]

    if df_cat_focus.empty:
        st.info("Aucun produit pour cette catégorie ce mois-ci.")
    else:
        top_prod = (
            df_cat_focus.groupby("designation", as_index=False)
            .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
            .sort_values("CA", ascending=False)
        )

        st.dataframe(top_prod)

        # Barres plus compactes (Altair)
        top10 = top_prod.head(10)
        chart_top = (
            alt.Chart(top10)
            .mark_bar()
            .encode(
                x=alt.X("CA:Q", title="CA (€)"),
                y=alt.Y("designation:N", sort="-x", title=None),
                tooltip=["designation", alt.Tooltip("CA:Q", format=".2f"), "Quantites"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart_top, use_container_width=True)


# -------------------------------------------------------------------
# TAB 2 : COMPARAISON
# -------------------------------------------------------------------
with tab_comp:
    st.subheader("Comparaison mois par mois")

    col_deb, col_fin = st.columns(2)
    with col_deb:
        mois_deb = st.selectbox(
            "Mois de début (TVA)",
            options=mois_dispo_tva,
            index=0,
            format_func=format_mois_label,
            key="comp_month_start",
        )
    with col_fin:
        mois_fin = st.selectbox(
            "Mois de fin (TVA)",
            options=mois_dispo_tva,
            index=len(mois_dispo_tva) - 1,
            format_func=format_mois_label,
            key="comp_month_end",
        )

    idx_deb = mois_dispo_tva.index(mois_deb)
    idx_fin = mois_dispo_tva.index(mois_fin)
    if idx_deb > idx_fin:
        st.error("Le mois de début doit être antérieur ou égal au mois de fin.")
    else:
        mois_range = mois_dispo_tva[idx_deb:idx_fin + 1]
        df_range_tva = df_hist_tva[df_hist_tva["mois"].isin(mois_range)]
        summary_range = build_month_summary_tva(df_range_tva)
        summary_range = add_deltas(summary_range, "CA_total")
        summary_range["mois_label"] = summary_range["mois"].apply(format_mois_label)

        # Vue globale CA total (TVA) – BARRES + tableau
        st.markdown("### Vue globale – CA total par mois (TVA)")

        col_chart, col_table = st.columns((1.3, 1))
        with col_chart:
            chart_global = (
                alt.Chart(summary_range)
                .mark_bar()
                .encode(
                    x=alt.X("mois_label:N", sort=list(summary_range["mois_label"]), title=None),
                    y=alt.Y("CA_total:Q", title="CA total (€)"),
                    tooltip=[alt.Tooltip("CA_total:Q", format=".2f"), "mois_label"]
                )
                .properties(height=280)
            )
            st.altair_chart(chart_global, use_container_width=True)

        with col_table:
            df_display = summary_range[["mois", "CA_total", "Delta_CA_total", "Delta_%_CA_total"]].copy()
            df_display["mois"] = df_display["mois"].apply(format_mois_label)

            styled = df_display.style.format({
                "CA_total": "{:,.2f} €".format,
                "Delta_CA_total": lambda x: "—" if pd.isna(x) else f"{x:+.0f} €",
                "Delta_%_CA_total": lambda x: "—" if pd.isna(x) else f"{x:+.1f} %",
            }).applymap(style_delta, subset=["Delta_CA_total", "Delta_%_CA_total"])

            st.dataframe(styled)

        st.markdown("---")

        # Comparaison par catégorie TVA (CA mensuel)
        st.markdown("### Comparaison par catégorie (CA mensuel – TVA)")

        cat_choice = st.selectbox(
            "Catégorie TVA à comparer",
            options=CATEGORIES_TVA,
            key="comp_cat_choice",
        )

        col_name = f"CA_{cat_choice}"
        df_cat = summary_range[["mois", "mois_label", col_name]].copy().rename(columns={col_name: "CA_cat"})
        df_cat = add_deltas(df_cat, "CA_cat")

        col_chart_cat, col_table_cat = st.columns((1.3, 1))

        with col_chart_cat:
            chart_cat = (
                alt.Chart(df_cat)
                .mark_bar()
                .encode(
                    x=alt.X("mois_label:N", sort=list(df_cat["mois_label"]), title=None),
                    y=alt.Y("CA_cat:Q", title=f"CA – {cat_choice} (€)"),
                    tooltip=[alt.Tooltip("CA_cat:Q", format=".2f"), "mois_label"]
                )
                .properties(height=260)
            )
            st.altair_chart(chart_cat, use_container_width=True)

        with col_table_cat:
            df_cat_display = df_cat[["mois", "CA_cat", "Delta_CA_cat", "Delta_%_CA_cat"]].copy()
            df_cat_display["mois"] = df_cat_display["mois"].apply(format_mois_label)

            styled_cat = df_cat_display.style.format({
                "CA_cat": "{:,.2f} €".format,
                "Delta_CA_cat": lambda x: "—" if pd.isna(x) else f"{x:+.0f} €",
                "Delta_%_CA_cat": lambda x: "—" if pd.isna(x) else f"{x:+.1f} %",
            }).applymap(style_delta, subset=["Delta_CA_cat", "Delta_%_CA_cat"])

            st.dataframe(styled_cat)

        st.markdown("---")

        # Comparaison abonnements / cartes (CSV)
        st.markdown("### Abonnements & carnets 10 – comparaison par mois (CSV)")

        if not has_abos:
            st.info("Aucune donnée CSV d’inscriptions importée – pas de comparaison possible.")
        else:
            df_abos_range = df_abos[df_abos["mois_creation"].isin(mois_range)]

            if df_abos_range.empty:
                st.info("Aucune inscription sur cette plage de mois.")
            else:
                abo_month = (
                    df_abos_range[df_abos_range["type_contrat"] == "ABONNEMENT"]
                    .groupby("mois_creation", as_index=False)
                    .agg(
                        Nb_abos=("offre", "count"),
                        CA_abos=("prix_effectif", "sum"),
                    )
                )

                cartes_month = (
                    df_abos_range[df_abos_range["type_contrat"] == "CARTE_10"]
                    .groupby("mois_creation", as_index=False)
                    .agg(
                        Nb_cartes=("offre", "count"),
                        CA_cartes=("prix_effectif", "sum"),
                    )
                )

                abo_cartes = pd.merge(
                    abo_month,
                    cartes_month,
                    on="mois_creation",
                    how="outer",
                ).fillna(0.0)
                abo_cartes["mois"] = abo_cartes["mois_creation"]
                abo_cartes = abo_cartes.sort_values("mois", key=lambda s: s.map(lambda x: datetime.strptime(x, "%Y-%m")))
                abo_cartes["mois_label"] = abo_cartes["mois"].apply(format_mois_label)

                # Nb abos / cartes – barres côte à côte (2 graphiques colonnes)
                col_chart_abos, col_tab_abos = st.columns((1.3, 1))

                with col_chart_abos:
                    df_long = abo_cartes.melt(
                        id_vars=["mois", "mois_label"],
                        value_vars=["Nb_abos", "Nb_cartes"],
                        var_name="type",
                        value_name="Nombre",
                    )
                    df_long["type"] = df_long["type"].map({"Nb_abos": "Abonnements", "Nb_cartes": "Carnets 10"})

                    chart_nb = (
                        alt.Chart(df_long)
                        .mark_bar()
                        .encode(
                            x=alt.X("mois_label:N", sort=list(abo_cartes["mois_label"]), title=None),
                            y=alt.Y("Nombre:Q", title="Nombre de contrats"),
                            color=alt.Color("type:N", title=""),
                            column=alt.Column("type:N", title=None),
                        )
                        .properties(height=260)
                    )
                    st.altair_chart(chart_nb, use_container_width=True)

                with col_tab_abos:
                    disp = abo_cartes[["mois", "Nb_abos", "CA_abos", "Nb_cartes", "CA_cartes"]].copy()
                    disp["mois"] = disp["mois"].apply(format_mois_label)
                    st.dataframe(disp)

                # CA abos vs carnets – barres groupées par type
                st.markdown("#### CA abonnements vs carnets 10 (CSV)")

                df_ca_long = abo_cartes.melt(
                    id_vars=["mois", "mois_label"],
                    value_vars=["CA_abos", "CA_cartes"],
                    var_name="type",
                    value_name="CA",
                )
                df_ca_long["type"] = df_ca_long["type"].map({"CA_abos": "Abonnements", "CA_cartes": "Carnets 10"})

                chart_ca = (
                    alt.Chart(df_ca_long)
                    .mark_bar()
                    .encode(
                        x=alt.X("mois_label:N", sort=list(abo_cartes["mois_label"]), title=None),
                        y=alt.Y("CA:Q", title="CA (€)"),
                        color=alt.Color("type:N", title=""),
                        column=alt.Column("type:N", title=None),
                    )
                    .properties(height=260)
                )
                st.altair_chart(chart_ca, use_container_width=True)

                st.caption("Affichage côte à côte pour lire rapidement la dynamique abos vs carnets.")


# -------------------------------------------------------------------
# TAB 3 : DÉTAIL PRODUITS / ABOS
# -------------------------------------------------------------------
with tab_detail:
    st.subheader("Détail produits (TVA) et abonnements/cartes (CSV)")

    col_cat_d, col_mois_d = st.columns(2)
    with col_cat_d:
        cat_det = st.selectbox(
            "Catégorie TVA (détail produits)",
            options=CATEGORIES_TVA,
            key="detail_cat_tva",
        )
    with col_mois_d:
        mois_det = st.selectbox(
            "Mois (détail produits TVA)",
            options=mois_dispo_tva,
            index=len(mois_dispo_tva) - 1,
            format_func=format_mois_label,
            key="detail_month_tva",
        )

    df_det = df_hist_tva[(df_hist_tva["categorie"] == cat_det) & (df_hist_tva["mois"] == mois_det)]

    if df_det.empty:
        st.info("Aucune donnée TVA pour cette combinaison.")
    else:
        top_prod = (
            df_det.groupby("designation", as_index=False)
            .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
            .sort_values("CA", ascending=False)
        )

        st.markdown(f"Top produits TVA – **{cat_det}** – {format_mois_label(mois_det)}")
        st.dataframe(top_prod)

        top10 = top_prod.head(10)
        chart_top = (
            alt.Chart(top10)
            .mark_bar()
            .encode(
                x=alt.X("CA:Q", title="CA (€)"),
                y=alt.Y("designation:N", sort="-x", title=None),
                tooltip=["designation", alt.Tooltip("CA:Q", format=".2f"), "Quantites"]
            )
            .properties(height=280)
        )
        st.altair_chart(chart_top, use_container_width=True)

    st.markdown("---")

    st.subheader("Abonnements / cartes – détail (CSV)")

    if not has_abos:
        st.info("Aucune donnée CSV d’inscriptions importée.")
    else:
        col_cat_a, col_mois_a = st.columns(2)
        with col_cat_a:
            type_filter = st.selectbox(
                "Type de contrat",
                options=["ABONNEMENT", "CARTE_10", "EVENT", "EXCLU", "TOUS"],
                key="detail_type_abos",
            )
        with col_mois_a:
            mois_abos = st.selectbox(
                "Mois (détail inscriptions)",
                options=sort_months(df_abos["mois_creation"].unique()),
                index=len(sort_months(df_abos["mois_creation"].unique())) - 1,
                format_func=format_mois_label,
                key="detail_month_abos",
            )

        df_det_abos = df_abos[df_abos["mois_creation"] == mois_abos]
        if type_filter != "TOUS":
            df_det_abos = df_det_abos[df_det_abos["type_contrat"] == type_filter]

        if df_det_abos.empty:
            st.info("Aucune inscription pour cette combinaison.")
        else:
            agg = (
                df_det_abos.groupby(["type_contrat", "sous_type"], as_index=False)
                .agg(
                    Nb=("offre", "count"),
                    CA=("prix_effectif", "sum"),
                )
                .sort_values("CA", ascending=False)
            )
            st.markdown(f"Détail contrats – {format_mois_label(mois_abos)}")
            st.dataframe(agg)

    st.markdown("---")

    st.subheader("Produits TVA classés en AUTRE (à recatégoriser)")

    df_autre = (
        df_hist_tva[df_hist_tva["categorie"] == "AUTRE"]
        .groupby("designation", as_index=False)
        .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
        .sort_values("CA", ascending=False)
    )

    if df_autre.empty:
        st.info("Aucun produit TVA en catégorie AUTRE.")
    else:
        st.dataframe(df_autre)
