import os
import re
from io import BytesIO
from datetime import datetime, date

import streamlit as st
import pandas as pd
import pdfplumber
import matplotlib.pyplot as plt


# =========================
# CONFIG & CONSTANTES
# =========================

DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# =========================
# FONCTIONS UTILITAIRES
# =========================

def to_float(x):
    """Convertit une chaîne avec virgule/€/espaces en float."""
    if x is None:
        return 0.0
    s = str(x)
    s = s.replace("€", "").replace(" ", "").replace("\u00a0", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


def to_int(x):
    """Convertit en int, tolérant les virgules/espaces."""
    try:
        return int(float(str(x).replace(",", ".").replace(" ", "")))
    except ValueError:
        return 0


def extract_period_from_text(text: str):
    """
    Cherche dans le texte une période du type '01-10-2025 - 31-10-2025'.
    Retourne (mois 'AAAA-MM', date_debut, date_fin).
    """
    match = re.search(r"(\d{2}-\d{2}-\d{4})\s*-\s*(\d{2}-\d{2}-\d{4})", text)
    if not match:
        return None, None, None

    d1 = datetime.strptime(match.group(1), "%d-%m-%Y")
    d2 = datetime.strptime(match.group(2), "%d-%m-%Y")
    mois = f"{d1.year}-{d1.month:02d}"
    return mois, d1.date().isoformat(), d2.date().isoformat()


# =========================
# CATEGORISATION PRODUITS
# =========================

def categorize_product(name: str):
    """
    Renvoie (categorie_principale, sous_categorie) à partir de la désignation.
    Catégories principales imposées :
    - 'Abonnements / cartes'
    - 'Boissons & compléments alimentaires'
    - 'Vestimentaire & accessoires sport'
    - 'AUTRE'
    """
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
# EXTRACTION PDF
# =========================

def extract_sales_tables_from_pdf(file_obj: BytesIO, forced_month: str = None) -> pd.DataFrame:
    """
    Lit un PDF Helios CrossFit - Rapport TVA et retourne un DataFrame
    avec toutes les lignes de ventes (OFFRES + PRODUITS).

    Si forced_month est fourni (format 'AAAA-MM'), ce mois est utilisé,
    sinon on essaie de le déduire du PDF.
    """
    rows = []
    periode_mois = None
    periode_debut = None
    periode_fin = None

    with pdfplumber.open(file_obj) as pdf:
        # Récupération de la période depuis la première page
        first_page_text = pdf.pages[0].extract_text() or ""
        periode_mois_detecte, periode_debut, periode_fin = extract_period_from_text(first_page_text)

        if forced_month is not None:
            periode_mois = forced_month
        else:
            periode_mois = periode_mois_detecte

        # Fallback si rien trouvé
        if periode_mois is None:
            today = datetime.today()
            periode_mois = f"{today.year}-{today.month:02d}"

        # Parcours des tables
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                if not t or len(t) < 2:
                    continue

                # Première ligne = header
                header = [c.strip() if c else "" for c in t[0]]
                header_lower = [h.lower() for h in header]

                # On ne garde que les tables ayant Désignation + Quantité
                if not any("désignation" in h or "designation" in h for h in header_lower):
                    continue
                if not any("quantité" in h or "quantite" in h for h in header_lower):
                    continue

                data_rows = t[1:]
                df = pd.DataFrame(data_rows, columns=header)

                # Normalisation des noms de colonnes
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

                # Nettoyage numérique
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

                # Meta période
                df["mois"] = periode_mois
                df["periode_debut"] = periode_debut
                df["periode_fin"] = periode_fin

                rows.append(df)

    if not rows:
        return pd.DataFrame()

    full_df = pd.concat(rows, ignore_index=True)

    # Suppression lignes vides
    full_df = full_df[full_df["designation"].notna()]
    full_df = full_df[full_df["designation"].str.strip() != ""]

    # Catégorisation
    cat_main = []
    cat_sub = []
    for name in full_df["designation"]:
        cmain, csub = categorize_product(name)
        cat_main.append(cmain)
        cat_sub.append(csub)
    full_df["categorie"] = cat_main
    full_df["sous_categorie"] = cat_sub

    # ID de ligne pour éviter les doublons
    full_df["id_ligne"] = (
        full_df["mois"].astype(str)
        + "_"
        + full_df["designation"].astype(str)
        + "_"
        + full_df["quantite"].astype(str)
        + "_"
        + full_df["total_ttc"].astype(str)
    )

    return full_df


# =========================
# HISTORIQUE
# =========================

def load_history() -> pd.DataFrame:
    """Charge l'historique, en gérant le cas fichier vide/corrompu."""
    cols = [
        "mois",
        "periode_debut",
        "periode_fin",
        "designation",
        "quantite",
        "total_ttc",
        "total_tva",
        "total_ht",
        "tva_pct",
        "categorie",
        "sous_categorie",
        "id_ligne",
    ]

    if os.path.exists(HISTORY_FILE):
        try:
            df = pd.read_csv(HISTORY_FILE)
            if df.empty:
                return pd.DataFrame(columns=cols)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=cols)
    else:
        return pd.DataFrame(columns=cols)


def save_history(df_history: pd.DataFrame):
    df_history.to_csv(HISTORY_FILE, index=False)


# =========================
# UI STREAMLIT
# =========================

st.set_page_config(page_title="Helios – Reporting CA", layout="wide")
st.title("Helios CrossFit – Reporting CA à partir des rapports TVA (PDF)")

st.markdown(
    """
**Process d'import :**  
1. Exporter le **rapport TVA PDF** du mois depuis ton logiciel.  
2. Choisir le **mois concerné** ci-dessous (une date à l'intérieur du mois).  
3. Uploader le PDF.  
4. L’app ajoute les lignes à l’historique (sans écraser l’existant).
"""
)

# Sélecteur de mois (on choisit une date dans le mois)
# Sélecteur simple : Année + Mois
annee_courante = datetime.today().year
annees = list(range(2022, annee_courante + 1))

mois_fr = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril",
    5: "Mai", 6: "Juin", 7: "Juillet", 8: "Août",
    9: "Septembre", 10: "Octobre", 11: "Novembre", 12: "Décembre",
}

col1, col2 = st.columns(2)

with col1:
    annee_select = st.selectbox("Année du rapport :", options=annees, index=len(annees)-1)

with col2:
    mois_select_num = st.selectbox("Mois du rapport :", options=list(mois_fr.keys()),
                                   format_func=lambda x: mois_fr[x])

selected_month = f"{annee_select}-{mois_select_num:02d}"


uploaded_pdf = st.file_uploader("Uploader un rapport TVA (PDF)", type=["pdf"])

if uploaded_pdf is not None:
    with st.spinner("Extraction des données du PDF..."):
        df_new = extract_sales_tables_from_pdf(
            BytesIO(uploaded_pdf.read()),
            forced_month=selected_month,
        )

    if df_new.empty:
        st.error("Impossible d'extraire des lignes de ventes depuis ce PDF. Vérifie le format.")
    else:
        mois = df_new["mois"].iloc[0]
        ca_new = df_new["total_ttc"].sum()
        nb_lignes = len(df_new)

        st.success(f"{nb_lignes} lignes importées pour le mois **{mois}** (CA : {ca_new:.2f} €)")
        st.dataframe(df_new.head(20))

        # Fusion avec l'historique
        df_hist = load_history()
        ids_existants = set(df_hist["id_ligne"].astype(str)) if not df_hist.empty else set()
        df_to_add = df_new[~df_new["id_ligne"].astype(str).isin(ids_existants)]

        df_hist = pd.concat([df_hist, df_to_add], ignore_index=True)
        save_history(df_hist)

        st.info(f"Historique mis à jour. {len(df_to_add)} nouvelles lignes ajoutées (sans doublons).")

st.markdown("---")

# ===== DASHBOARD GLOBAL =====

df_hist = load_history()
if df_hist.empty:
    st.warning("Aucune donnée historique pour l’instant. Uploade au moins un PDF.")
    st.stop()

# Types
df_hist["mois"] = df_hist["mois"].astype(str)
df_hist["total_ttc"] = df_hist["total_ttc"].astype(float)
df_hist["quantite"] = df_hist["quantite"].astype(int)

col_filtres, col_export = st.columns([3, 1])

with col_filtres:
    mois_dispo = sorted(df_hist["mois"].unique())
    mois_select = st.multiselect(
        "Mois à analyser",
        options=mois_dispo,
        default=[mois_dispo[-1]]  # dernier mois par défaut
    )

    if not mois_select:
        st.warning("Sélectionne au moins un mois.")
        st.stop()

df_sel = df_hist[df_hist["mois"].isin(mois_select)]

with col_export:
    st.download_button(
        "📥 Export CSV données filtrées",
        data=df_sel.to_csv(index=False).encode("utf-8"),
        file_name="helios_reporting_filtre.csv",
        mime="text/csv",
    )

# ----- KPIs -----

ca_total = df_sel["total_ttc"].sum()
quant_total = df_sel["quantite"].sum()

delta_ca_global = None
delta_ca_cat = {}

if len(mois_select) == 1:
    mois_unique = mois_select[0]
    mois_sorted = sorted(mois_dispo)
    idx = mois_sorted.index(mois_unique)
    if idx > 0:
        mois_prev = mois_sorted[idx - 1]
        df_curr = df_hist[df_hist["mois"] == mois_unique]
        df_prev = df_hist[df_hist["mois"] == mois_prev]

        ca_curr = df_curr["total_ttc"].sum()
        ca_prev = df_prev["total_ttc"].sum()
        if ca_prev > 0:
            delta_ca_global = (ca_curr - ca_prev) / ca_prev * 100

        for cat in [
            "Abonnements / cartes",
            "Boissons & compléments alimentaires",
            "Vestimentaire & accessoires sport",
        ]:
            ca_curr_cat = df_curr[df_curr["categorie"] == cat]["total_ttc"].sum()
            ca_prev_cat = df_prev[df_prev["categorie"] == cat]["total_ttc"].sum()
            if ca_prev_cat > 0:
                delta_ca_cat[cat] = (ca_curr_cat - ca_prev_cat) / ca_prev_cat * 100
            elif ca_curr_cat > 0:
                delta_ca_cat[cat] = None

st.subheader("Indicateurs clés")

c1, c2, c3, c4 = st.columns(4)
c1.metric("CA total", f"{ca_total:,.2f} €".replace(",", " "))
c2.metric("Quantités vendues", f"{quant_total}")
if len(mois_select) == 1 and delta_ca_global is not None:
    c3.metric("CA vs mois précédent", f"{delta_ca_global:+.1f} %")
else:
    c3.metric("CA vs mois précédent", "N/A")
c4.metric("Nb lignes (ventes)", f"{len(df_sel)}")

st.markdown("---")

# ----- RÉPARTITION PAR CATÉGORIE -----

st.subheader("Répartition du CA par catégorie")

ca_par_cat = (
    df_sel.groupby("categorie", as_index=False)["total_ttc"]
    .sum()
    .rename(columns={"total_ttc": "CA"})
)

col_pie, col_tab = st.columns([1, 1])

with col_pie:
    fig, ax = plt.subplots()
    ax.pie(ca_par_cat["CA"], labels=ca_par_cat["categorie"], autopct="%1.1f%%")
    ax.set_title("CA par catégorie")
    st.pyplot(fig)

with col_tab:
    st.dataframe(ca_par_cat.sort_values("CA", ascending=False))

st.markdown("---")

# ----- TENDANCE CA PAR MOIS -----

st.subheader("Évolution du CA par mois")

ca_mensuel = (
    df_hist.groupby("mois", as_index=False)["total_ttc"]
    .sum()
    .rename(columns={"total_ttc": "CA_total"})
    .sort_values("mois")
)
st.line_chart(ca_mensuel.set_index("mois"))

st.markdown("### Évolution par catégorie (CA mensuel)")

ca_cat_mensuel = (
    df_hist.groupby(["mois", "categorie"], as_index=False)["total_ttc"]
    .sum()
    .rename(columns={"total_ttc": "CA"})
)

pivot_cat = ca_cat_mensuel.pivot(index="mois", columns="categorie", values="CA").fillna(0)
st.area_chart(pivot_cat)

st.markdown("---")

# ----- DÉTAIL PAR CAT / PRODUIT -----

st.subheader("Détail par catégorie et produits")

col_cat, col_mois_detail = st.columns(2)

with col_cat:
    cat_dispo = sorted(df_hist["categorie"].unique())
    cat_select = st.selectbox("Catégorie", options=cat_dispo)

with col_mois_detail:
    mois_detail = st.selectbox(
        "Mois (pour le détail produits)",
        options=mois_dispo,
        index=len(mois_dispo) - 1
    )

df_detail = df_hist[
    (df_hist["categorie"] == cat_select) &
    (df_hist["mois"] == mois_detail)
]

if df_detail.empty:
    st.info("Aucune donnée pour cette combinaison mois / catégorie.")
else:
    top_produits = (
        df_detail.groupby("designation", as_index=False)
        .agg(
            CA=("total_ttc", "sum"),
            Quantites=("quantite", "sum")
        )
        .sort_values("CA", ascending=False)
    )

    st.markdown(
        f"**Top produits – {cat_select} – {mois_detail}** "
        "(triés par CA décroissant)"
    )
    st.dataframe(top_produits)

    top10 = top_produits.head(10).set_index("designation")
    st.bar_chart(top10["CA"])

# ----- PRODUITS EN AUTRE -----

st.markdown("---")
st.subheader("Produits classés en catégorie AUTRE (à optimiser)")

df_autre = (
    df_hist[df_hist["categorie"] == "AUTRE"]
    .groupby("designation", as_index=False)
    .agg(
        CA=("total_ttc", "sum"),
        Quantites=("quantite", "sum")
    )
    .sort_values("CA", ascending=False)
)

if df_autre.empty:
    st.info("Aucun produit en catégorie AUTRE actuellement.")
else:
    st.dataframe(df_autre)
