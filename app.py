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
HISTORY_FILE = os.path.join(DATA_DIR, "history.csv")
os.makedirs(DATA_DIR, exist_ok=True)

CATEGORIES = [
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
# UTILS
# =========================

def to_float(x):
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


# =========================
# CATEGORISATION
# =========================

def categorize_product(name: str):
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
    Lit un PDF Helios CrossFit - Rapport TVA et renvoie un DataFrame
    avec toutes les lignes de ventes (OFFRES + PRODUITS).
    """
    rows = []
    periode_debut = None
    periode_fin = None

    with pdfplumber.open(file_obj) as pdf:
        first_text = pdf.pages[0].extract_text() or ""
        mois_detecte, periode_debut, periode_fin = extract_period_from_text(first_text)

        # On utilise toujours le mois choisi dans l'UI
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
        cmain, csub = categorize_product(name)
        cat_main.append(cmain)
        cat_sub.append(csub)
    full_df["categorie"] = cat_main
    full_df["sous_categorie"] = cat_sub

    return full_df


# =========================
# HISTORIQUE
# =========================

def load_history() -> pd.DataFrame:
    cols = [
        "mois", "periode_debut", "periode_fin",
        "designation", "quantite",
        "total_ttc", "total_tva", "total_ht", "tva_pct",
        "categorie", "sous_categorie",
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
# AGGREGATIONS
# =========================

def build_month_summary(df_hist: pd.DataFrame) -> pd.DataFrame:
    df = df_hist.copy()
    res = df.groupby("mois").agg(
        CA_total=("total_ttc", "sum"),
        Qt_total=("quantite", "sum"),
    ).reset_index()

    for cat in CATEGORIES:
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


def add_deltas(df: pd.DataFrame, col_base: str) -> pd.DataFrame:
    df = df.copy()
    df[f"{col_base}_prec"] = df[col_base].shift(1)
    df[f"Delta_{col_base}"] = df[col_base] - df[f"{col_base}_prec"]
    df[f"Delta_%_{col_base}"] = (df[f"Delta_{col_base}"] / df[f"{col_base}_prec"] * 100)
    df[f"Delta_%_{col_base}"] = df[f"Delta_%_{col_base}"].replace([pd.NA, float("inf"), -float("inf")], pd.NA)
    return df


def style_delta(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: green; font-weight: bold;"
    if val < 0:
        return "color: red; font-weight: bold;"
    return ""


# =========================
# UI
# =========================

st.set_page_config(page_title="Helios – Reporting CA", layout="wide")
st.title("Helios CrossFit – Outil de reporting CA")

st.markdown(
    """
### Import de données
1. Exporter le **rapport TVA PDF** du mois depuis ton logiciel.  
2. Choisir le **mois concerné** (année + mois).  
3. Uploader le PDF.  
4. Cliquer sur **Importer / remplacer ce mois**.  

👉 L’app **supprime toutes les anciennes lignes** de ce mois et les remplace par les nouvelles,  
sans toucher aux autres mois.
"""
)

# --- Sélecteur de mois pour l'import ---
annee_courante = datetime.today().year
annees = list(range(2022, annee_courante + 1))

col_a, col_m = st.columns(2)
with col_a:
    annee_select = st.selectbox(
        "Année du rapport à importer",
        options=annees,
        index=len(annees) - 1,
        key="import_year",
    )
with col_m:
    mois_num = st.selectbox(
        "Mois du rapport à importer",
        options=list(MOIS_FR.keys()),
        format_func=lambda x: MOIS_FR[x],
        key="import_month",
    )

mois_import = f"{annee_select}-{mois_num:02d}"

uploaded_pdf = st.file_uploader("Uploader le rapport TVA (PDF)", type=["pdf"], key="pdf_uploader")
import_clicked = st.button("Importer / remplacer ce mois", key="import_button")

if import_clicked:
    if uploaded_pdf is None:
        st.error("Merci de choisir d'abord un fichier PDF.")
    else:
        with st.spinner("Extraction des données du PDF..."):
            df_new = extract_sales_tables_from_pdf(BytesIO(uploaded_pdf.read()), forced_month=mois_import)

        if df_new.empty:
            st.error("Impossible d'extraire des ventes depuis ce PDF. Vérifie le format.")
        else:
            df_hist = load_history()

            # On supprime l'ancien mois
            nb_ancien = len(df_hist[df_hist["mois"] == mois_import])
            df_autres = df_hist[df_hist["mois"] != mois_import]

            # On ajoute les nouvelles lignes
            df_hist_new = pd.concat([df_autres, df_new], ignore_index=True)
            save_history(df_hist_new)

            ca_new = df_new["total_ttc"].sum()

            st.success(
                f"{len(df_new)} lignes importées pour {format_mois_label(mois_import)} "
                f"(remplace {nb_ancien} lignes précédentes). CA : {ca_new:.2f} €."
            )
            st.dataframe(df_new, use_container_width=True)

st.markdown("---")

# ====== DATA DISPONIBLE ? ======
df_hist = load_history()
if df_hist.empty:
    st.warning("Aucune donnée historique pour l’instant.")
    st.stop()

df_hist["mois"] = df_hist["mois"].astype(str)
df_hist["total_ttc"] = df_hist["total_ttc"].astype(float)
df_hist["quantite"] = df_hist["quantite"].astype(int)

mois_dispo = sort_months(df_hist["mois"].unique())
month_summary = build_month_summary(df_hist)

# =========================
# TABS : 1) Vue mensuelle 2) Comparaison 3) Détail
# =========================

tab_mensuel, tab_comp, tab_detail = st.tabs(["📅 Vue mensuelle", "📈 Comparaison mensuelle", "🔍 Détail produits"])

# -------------------------------------------------------------------
# TAB 1 : VUE MENSUELLE
# -------------------------------------------------------------------
with tab_mensuel:
    st.subheader("Analyse d’un mois")

    mois_focus = st.selectbox(
        "Mois à analyser",
        options=mois_dispo,
        index=len(mois_dispo) - 1,
        format_func=format_mois_label,
        key="view_month_focus",
    )

    df_mois = df_hist[df_hist["mois"] == mois_focus]

    # Mois précédent
    mois_sorted = mois_dispo
    idx = mois_sorted.index(mois_focus)
    df_prev = None
    if idx > 0:
        mois_prev = mois_sorted[idx - 1]
        df_prev = df_hist[df_hist["mois"] == mois_prev]

    ca_mois = df_mois["total_ttc"].sum()
    qte_mois = df_mois["quantite"].sum()

    ca_prev = df_prev["total_ttc"].sum() if df_prev is not None else None
    delta_ca_abs = None
    delta_ca_pct = None
    if ca_prev and ca_prev != 0:
        delta_ca_abs = ca_mois - ca_prev
        delta_ca_pct = (delta_ca_abs / ca_prev) * 100

    ca_cat_mois = (
        df_mois.groupby("categorie", as_index=False)
        .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
        .sort_values("CA", ascending=False)
    )

    st.markdown(f"### Synthèse – {format_mois_label(mois_focus)}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CA total", f"{ca_mois:,.2f} €".replace(",", " "))
    c2.metric("Quantités vendues", int(qte_mois))
    if delta_ca_abs is not None:
        c3.metric("Δ CA vs mois précédent", f"{delta_ca_abs:+.0f} €", f"{delta_ca_pct:+.1f} %")
    else:
        c3.metric("Δ CA vs mois précédent", "N/A", "N/A")
    c4.metric("Nb lignes (ventes)", len(df_mois))

    st.markdown("---")

    # Répartition catégorie (camembert)
    st.markdown("#### Répartition du CA par catégorie")

    col_pie, col_tab = st.columns([1, 1])
    with col_pie:
        fig, ax = plt.subplots()
        if not ca_cat_mois.empty:
            ax.pie(ca_cat_mois["CA"], labels=ca_cat_mois["categorie"], autopct="%1.1f%%")
        ax.set_title("CA par catégorie")
        st.pyplot(fig)

    with col_tab:
        if ca_mois > 0:
            ca_cat_mois["% CA"] = (ca_cat_mois["CA"] / ca_mois * 100).round(1)
        else:
            ca_cat_mois["% CA"] = 0.0
        st.dataframe(ca_cat_mois, use_container_width=True)

    st.markdown("---")

    # Top produits par catégorie
    st.markdown("#### Top produits par catégorie (mois sélectionné)")

    cat_focus = st.selectbox(
        "Catégorie (top produits)",
        options=CATEGORIES,
        key="month_cat_focus",
    )
    df_cat_focus = df_mois[df_mois["categorie"] == cat_focus]

    if df_cat_focus.empty:
        st.info("Aucun produit pour cette catégorie ce mois-ci.")
    else:
        top_prod = (
            df_cat_focus.groupby("designation", as_index=False)
            .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
            .sort_values("CA", ascending=False)
        )

        st.markdown(f"Top produits – **{cat_focus}** – {format_mois_label(mois_focus)}")
        st.dataframe(top_prod, use_container_width=True)

        top10 = top_prod.head(10).set_index("designation")
        st.bar_chart(top10["CA"])


# -------------------------------------------------------------------
# TAB 2 : COMPARAISON
# -------------------------------------------------------------------
with tab_comp:
    st.subheader("Comparaison mois par mois")

    col_deb, col_fin = st.columns(2)
    with col_deb:
        mois_deb = st.selectbox(
            "Mois de début",
            options=mois_dispo,
            index=0,
            format_func=format_mois_label,
            key="comp_month_start",
        )
    with col_fin:
        mois_fin = st.selectbox(
            "Mois de fin",
            options=mois_dispo,
            index=len(mois_dispo) - 1,
            format_func=format_mois_label,
            key="comp_month_end",
        )

    idx_deb = mois_dispo.index(mois_deb)
    idx_fin = mois_dispo.index(mois_fin)
    if idx_deb > idx_fin:
        st.error("Le mois de début doit être antérieur ou égal au mois de fin.")
        st.stop()

    mois_range = mois_dispo[idx_deb:idx_fin + 1]
    df_range = df_hist[df_hist["mois"].isin(mois_range)]
    summary_range = build_month_summary(df_range)
    summary_range = add_deltas(summary_range, "CA_total")

    # Vue globale CA total
    st.markdown("### Vue globale – CA total par mois")

    col_chart, col_table = st.columns([1, 1])
    with col_chart:
        tmp = summary_range.set_index("mois")["CA_total"]
        tmp.index = [format_mois_label(m) for m in tmp.index]
        st.bar_chart(tmp)

    with col_table:
        df_display = summary_range[["mois", "CA_total", "Delta_CA_total", "Delta_%_CA_total"]].copy()
        df_display["mois"] = df_display["mois"].apply(format_mois_label)

        styled = df_display.style.format({
            "CA_total": "{:,.2f} €".format,
            "Delta_CA_total": lambda x: "—" if pd.isna(x) else f"{x:+.0f} €",
            "Delta_%_CA_total": lambda x: "—" if pd.isna(x) else f"{x:+.1f} %",
        }).applymap(style_delta, subset=["Delta_CA_total", "Delta_%_CA_total"])

        st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # Comparaison par catégorie
    st.markdown("### Comparaison par catégorie – CA mensuel")

    cat_choice = st.selectbox(
        "Catégorie à comparer",
        options=CATEGORIES,
        key="comp_cat_choice",
    )

    col_name = f"CA_{cat_choice}"
    df_cat = summary_range[["mois", col_name]].copy().rename(columns={col_name: "CA_cat"})
    df_cat = add_deltas(df_cat, "CA_cat")

    col_chart_cat, col_table_cat = st.columns([1, 1])

    with col_chart_cat:
        tmp = df_cat.set_index("mois")["CA_cat"]
        tmp.index = [format_mois_label(m) for m in tmp.index]
        st.bar_chart(tmp)

    with col_table_cat:
        df_cat_display = df_cat[["mois", "CA_cat", "Delta_CA_cat", "Delta_%_CA_cat"]].copy()
        df_cat_display["mois"] = df_cat_display["mois"].apply(format_mois_label)

        styled_cat = df_cat_display.style.format({
            "CA_cat": "{:,.2f} €".format,
            "Delta_CA_cat": lambda x: "—" if pd.isna(x) else f"{x:+.0f} €",
            "Delta_%_CA_cat": lambda x: "—" if pd.isna(x) else f"{x:+.1f} %",
        }).applymap(style_delta, subset=["Delta_CA_cat", "Delta_%_CA_cat"])

        st.dataframe(styled_cat, use_container_width=True)

    st.markdown("---")

    st.markdown("### Répartition par catégorie – empilée (stacked)")

    stacked = summary_range[["mois"] + [f"CA_{c}" for c in CATEGORIES]].copy()
    stacked = stacked.set_index("mois")
    stacked.index = [format_mois_label(m) for m in stacked.index]
    st.area_chart(stacked)


# -------------------------------------------------------------------
# TAB 3 : DÉTAIL PRODUITS
# -------------------------------------------------------------------
with tab_detail:
    st.subheader("Détail produits par mois et catégorie")

    col_cat_d, col_mois_d = st.columns(2)
    with col_cat_d:
        cat_det = st.selectbox(
            "Catégorie (détail produits)",
            options=CATEGORIES,
            key="detail_cat",
        )
    with col_mois_d:
        mois_det = st.selectbox(
            "Mois (détail produits)",
            options=mois_dispo,
            index=len(mois_dispo) - 1,
            format_func=format_mois_label,
            key="detail_month",
        )

    df_det = df_hist[(df_hist["categorie"] == cat_det) & (df_hist["mois"] == mois_det)]

    if df_det.empty:
        st.info("Aucune donnée pour cette combinaison.")
    else:
        top_prod = (
            df_det.groupby("designation", as_index=False)
            .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
            .sort_values("CA", ascending=False)
        )

        st.markdown(f"Top produits – **{cat_det}** – {format_mois_label(mois_det)}")
        st.dataframe(top_prod, use_container_width=True)

        top10 = top_prod.head(10).set_index("designation")
        st.bar_chart(top10["CA"])

    st.markdown("---")

    st.subheader("Produits classés en AUTRE (à recatégoriser)")
    df_autre = (
        df_hist[df_hist["categorie"] == "AUTRE"]
        .groupby("designation", as_index=False)
        .agg(CA=("total_ttc", "sum"), Quantites=("quantite", "sum"))
        .sort_values("CA", ascending=False)
    )

    if df_autre.empty:
        st.info("Aucun produit en catégorie AUTRE.")
    else:
        st.dataframe(df_autre, use_container_width=True)
