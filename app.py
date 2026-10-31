"""NOVA DISTRIBUTION - tableau de bord (Streamlit).

Lancement :  streamlit run app.py
Le fichier transactions_nova.csv doit être dans le même dossier
(ou être chargé depuis la barre latérale).
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nova_core as nc

st.set_page_config(page_title="NOVA Distribution - Tableau de bord", layout="wide")

CSV_PAR_DEFAUT = Path(__file__).parent / "transactions_nova.csv"
MOIS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


def eur(x: float) -> str:
    return f"{x:,.0f} €".replace(",", " ")


def pct(x: float, signe: bool = False) -> str:
    return f"{x:+.1%}" if signe else f"{x:.1%}"


@st.cache_data(show_spinner=False)
def lire(contenu: bytes | None, chemin: str) -> pd.DataFrame:
    return nc.charger(io.BytesIO(contenu) if contenu else chemin)


@st.cache_data(show_spinner=False)
def traiter(brut: pd.DataFrame, usd: str, paires: str, imputer: bool):
    d = nc.preparer(brut, usd=usd, paires=paires, imputer=imputer)
    return (d, nc.indicateurs(d), nc.cascade(d), nc.controles(brut, d, usd, paires, imputer),
            nc.effet_prix_volume(d))


@st.cache_data(show_spinner=False)
def scenarios(brut: pd.DataFrame) -> pd.DataFrame:
    return nc.sensibilite(brut)


# ---------------------------------------------------------------- barre latérale
st.sidebar.header("Données")
fichier = st.sidebar.file_uploader("Charger un autre CSV (optionnel)", type="csv")
if fichier is None and not CSV_PAR_DEFAUT.exists():
    st.error("transactions_nova.csv est introuvable : placez-le à côté de app.py "
             "ou chargez-le depuis la barre latérale.")
    st.stop()
brut = lire(fichier.getvalue() if fichier else None, str(CSV_PAR_DEFAUT))

st.sidebar.header("Hypothèses (conventions)")
usd = st.sidebar.radio(
    "Lignes hors EUR (USD)", ["isoler", "traiter_eur"],
    format_func={"isoler": "Isoler (pas de conversion)",
                 "traiter_eur": "Traiter comme EUR (hypothèse)"}.get,
    help="Le fichier ne contient aucun taux de change.")
paires = st.sidebar.radio(
    "FACTURE et AVOIR de même identifiant", ["garder", "doublon"],
    format_func={"garder": "Garder les deux lignes",
                 "doublon": "Avoir = doublon de la facture"}.get)
imputer = st.sidebar.checkbox("Reconstituer les coûts manquants (grille produit/année)", value=True)

d, k, casc, ctrl, pv = traiter(brut, usd, paires, imputer)
annuel = k["annuel"]
ref = (usd, paires, imputer) == (nc.PARAMS_DEFAUT["usd"], nc.PARAMS_DEFAUT["paires"], nc.PARAMS_DEFAUT["imputer"])
if not ref:
    st.sidebar.warning("Hypothèses modifiées : les chiffres diffèrent de la référence.")

# ---------------------------------------------------------------- en-tête
st.title("NOVA DISTRIBUTION - Tableau de bord 2024-2025")
st.caption("Chiffres calculés après audit : doublons, annulations, avoirs et devises traités. "
           "Aucune ligne n'est supprimée du fichier ; chaque exclusion est motivée (onglet Audit).")

c1, c2, c3, c4 = st.columns(4)
c1.metric("CA 2024", eur(annuel.loc[2024, "ca"]))
c2.metric("CA 2025", eur(annuel.loc[2025, "ca"]), pct(k["croissance"], True) + " vs 2024")
c3.metric("Marge 2025", eur(annuel.loc[2025, "marge"]),
          eur(annuel.loc[2025, "marge"] - annuel.loc[2024, "marge"]) + " vs 2024")
c4.metric("Taux de marge 2025", pct(annuel.loc[2025, "taux_marge"]),
          f"{(annuel.loc[2025, 'taux_marge'] - annuel.loc[2024, 'taux_marge']) * 100:+.2f} pt vs 2024")

onglets = st.tabs(["Synthèse", "Évolution mensuelle", "Clients et produits", "Audit de fiabilité", "Données"])

# ---------------------------------------------------------------- Synthèse
with onglets[0]:
    g1, g2 = st.columns(2)
    long = (annuel[["ca", "marge"]].rename(columns={"ca": "Chiffre d'affaires", "marge": "Marge commerciale"})
            .reset_index().melt("annee", var_name="indicateur", value_name="montant"))
    long["annee"] = long["annee"].astype(str)
    fig = px.bar(long, x="annee", y="montant", color="indicateur", barmode="group",
                 labels={"annee": "Exercice", "montant": "Montant (€)", "indicateur": ""},
                 title="CA et marge par exercice")
    g1.plotly_chart(fig, width="stretch")

    tab = annuel[["ca", "marge", "taux_marge", "unites"]].copy()
    tab.index = tab.index.astype(str)
    tab = tab.rename(columns={"ca": "CA (€)", "marge": "Marge (€)", "taux_marge": "Taux de marge",
                              "unites": "Unités nettes"})
    g2.subheader("Indicateurs annuels")
    g2.dataframe(tab.style.format({"CA (€)": "{:,.0f}", "Marge (€)": "{:,.0f}",
                                   "Taux de marge": "{:.2%}", "Unités nettes": "{:,.0f}"}),
                 width="stretch")
    g2.markdown(f"**Croissance du CA 2024-2025 : {pct(k['croissance'], True)}**")
    if pv:
        g2.markdown(
            f"Lecture prix / volume : à prix 2024, le CA 2025 serait de {eur(pv['ca25_aux_prix_2024'])} "
            f"({pct(pv['croissance_volume'], True)} de volume), et l'effet prix de la grille 2025 "
            f"est de {pct(pv['effet_prix'], True)}.")

# ---------------------------------------------------------------- Évolution mensuelle
with onglets[1]:
    m = k["mensuel"].copy()
    m["annee"] = m["annee"].astype(str)
    fig = px.line(m, x="mois", y="ca", color="annee", markers=True,
                  labels={"mois": "Mois", "ca": "CA (€)", "annee": "Exercice"},
                  title="Chiffre d'affaires mensuel")
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)), ticktext=MOIS)
    st.plotly_chart(fig, width="stretch")
    piv = m.pivot(index="mois", columns="annee", values="ca").reindex(range(1, 13))
    piv.index = MOIS
    piv["Évolution"] = piv["2025"] / piv["2024"] - 1
    st.dataframe(piv.style.format({"2024": "{:,.0f}", "2025": "{:,.0f}", "Évolution": "{:+.1%}"}),
                 width="stretch")

# ---------------------------------------------------------------- Clients et produits
with onglets[2]:
    cl, pr = st.columns(2)
    tc = k["top_clients"]
    fig = px.bar(tc, x="ca", y="client_id", orientation="h", text_auto=",.0f",
                 labels={"ca": "CA 2024-2025 (€)", "client_id": "Client"}, title="Top 5 clients (par CA)")
    fig.update_yaxes(categoryorder="total ascending")
    cl.plotly_chart(fig, width="stretch")
    tp = k["top_produits"]
    fig = px.bar(tp, x="produit", y="marge", text_auto=",.0f",
                 labels={"marge": "Marge 2024-2025 (€)", "produit": "Produit"},
                 title="Top 5 produits les plus rentables (par marge en €)")
    pr.plotly_chart(fig, width="stretch")
    cl.dataframe(tc.rename(columns={"client_id": "Client", "ca": "CA (€)"})
                 .style.format({"CA (€)": "{:,.0f}"}), width="stretch", hide_index=True)
    pr.dataframe(tp[["produit", "ca", "marge", "taux_marge"]]
                 .rename(columns={"produit": "Produit", "ca": "CA (€)", "marge": "Marge (€)",
                                  "taux_marge": "Taux de marge"})
                 .style.format({"CA (€)": "{:,.0f}", "Marge (€)": "{:,.0f}", "Taux de marge": "{:.1%}"}),
                 width="stretch", hide_index=True)
    st.caption("Classements sur les deux exercices cumulés. Le produit le plus rentable en euros "
               "n'est pas celui qui a le meilleur taux de marge.")

# ---------------------------------------------------------------- Audit
with onglets[3]:
    st.subheader("Du CA brut au CA retenu")
    an = st.radio("Exercice", list(nc.ANNEES), index=1, horizontal=True)
    etapes = casc["etape"].tolist()
    valeurs = [float(casc.loc[0, f"CA_{an}"])] + [float(casc.loc[i, f"delta_CA_{an}"]) for i in range(1, len(casc))]
    fig = go.Figure(go.Waterfall(
        x=etapes + ["CA retenu"], y=valeurs + [0],
        measure=["absolute"] + ["relative"] * (len(casc) - 1) + ["total"],
        text=[f"{v:,.0f}".replace(",", " ") for v in valeurs] + [eur(float(casc.iloc[-1][f"CA_{an}"]))],
        textposition="outside"))
    fig.update_layout(title=f"Cascade du CA {an} (€)", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Tableau des contrôles")
    st.dataframe(ctrl.rename(columns={
        "n": "N°", "controle": "Contrôle", "risque": "Risque", "methode": "Méthode",
        "resultat": "Résultat", "traitement": "Traitement retenu", "statut": "Statut",
        "effet_CA_2024": "Effet CA 2024 (€)", "effet_CA_2025": "Effet CA 2025 (€)",
        "effet_marge_total": "Effet marge (€)"}), width="stretch", hide_index=True)

    st.subheader("Sensibilité aux hypothèses")
    sc = scenarios(brut).rename(columns={
        "scenario": "Scénario", "CA_2024": "CA 2024", "CA_2025": "CA 2025", "croissance": "Croissance",
        "taux_marge_2024": "Taux marge 2024", "taux_marge_2025": "Taux marge 2025"})
    st.dataframe(sc.style.format({"CA 2024": "{:,.0f}", "CA 2025": "{:,.0f}", "Croissance": "{:+.1%}",
                                  "Taux marge 2024": "{:.2%}", "Taux marge 2025": "{:.2%}"}, na_rep="-"),
                 width="stretch", hide_index=True)

    st.subheader("Hypothèses et cas non résolus")
    st.markdown(
        "- **Devises** : aucun taux de change dans le fichier et prix USD identiques à la grille EUR. "
        "Il n'est pas établi que ces lignes soient réellement en dollars ; elles sont isolées, sans conversion.\n"
        "- **FACTURE et AVOIR de même identifiant** : doublon mal typé ou vraie annulation ? Non tranché ; "
        "effet maximal limité (voir contrôle 8).\n"
        "- **Avoirs** : aucune référence à la facture d'origine dans le fichier ; le rapprochement est indicatif.\n"
        "- **Prix 2025** : toute la grille est 9 % sous celle de 2024 ; la croissance du CA mélange effet prix et effet volume.\n"
        "- **Coûts manquants** : reconstitués avec la grille (coût unique par produit et par année) ; "
        "l'effet porte sur la marge, pas sur le CA.")

# ---------------------------------------------------------------- Données
with onglets[4]:
    motifs = ["Toutes", "Retenues"] + sorted(d.loc[~d["retenu"], "motif_exclusion"].unique().tolist())
    choix = st.selectbox("Afficher", motifs)
    vue = d if choix == "Toutes" else d[d["retenu"]] if choix == "Retenues" else d[d["motif_exclusion"] == choix]
    vue = vue[nc.COLONNES_EXPORT]
    st.caption(f"{len(vue):,} lignes affichées sur {len(d):,}.".replace(",", " "))
    st.dataframe(vue, width="stretch", hide_index=True)
    b1, b2 = st.columns(2)
    b1.download_button("Télécharger le fichier nettoyé (CSV)", nc.exporter_csv(d[nc.COLONNES_EXPORT]),
                       "nova_nettoye.csv", "text/csv")
    b2.download_button("Télécharger le tableau des contrôles (CSV)", nc.exporter_csv(ctrl),
                       "tableau_controles.csv", "text/csv")
