"""NOVA DISTRIBUTION - noyau de calcul (nettoyage, audit, indicateurs).

Ce module ne contient aucune interface : il est utilisé par app.py (Streamlit)
et peut aussi être lancé seul :

    python nova_core.py transactions_nova.csv

Principe : AUCUNE ligne n'est supprimée. Chaque ligne reçoit un statut
(`retenu` vrai/faux) et un `motif_exclusion` ; les indicateurs sont calculés
uniquement sur les lignes retenues.

Conventions (voir README.md) :
- doublon : une seule ligne par transaction_id (et par type d'opération) ;
- annulation : exclue du CA et de la marge ;
- avoir : compté en négatif (CA, coût, marge) ;
- devise : périmètre EUR ; lignes hors EUR isolées, sans conversion ;
- remise : prix net = prix x (1 - remise) ; le coût n'est pas remisé ;
- coût manquant : reconstitué via la grille produit/année, signalé.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ANNEES = (2024, 2025)
DEVISE_REF = "EUR"
COLONNES_ATTENDUES = [
    "transaction_id", "date_transaction", "client_id", "produit", "quantite",
    "prix_unitaire", "cout_unitaire", "remise_pct", "devise",
    "type_operation", "statut",
]

# Paramètres de scénario par défaut (conventions retenues à l'étape 2)
PARAMS_DEFAUT = {"usd": "isoler", "paires": "garder", "imputer": True}


# --------------------------------------------------------------------------
# 1. Chargement
# --------------------------------------------------------------------------
def charger(source) -> pd.DataFrame:
    """Lit le CSV brut et ajoute les colonnes de base (dates, prix net, CA brut)."""
    df = pd.read_csv(source)
    manquantes = set(COLONNES_ATTENDUES) - set(df.columns)
    if manquantes:
        raise ValueError(f"Colonnes manquantes dans le fichier : {sorted(manquantes)}")
    df = df.copy()
    df["ligne_source"] = np.arange(len(df)) + 2  # n° de ligne dans le CSV (en-tête = 1)
    df["date"] = pd.to_datetime(df["date_transaction"], format="%Y-%m-%d", errors="coerce")
    df["annee"] = df["date"].dt.year.astype("Int64")
    df["mois"] = df["date"].dt.month.astype("Int64")
    df["prix_net_unitaire"] = df["prix_unitaire"] * (1 - df["remise_pct"] / 100)
    df["ca_brut_ligne"] = df["quantite"] * df["prix_net_unitaire"]
    return df


# --------------------------------------------------------------------------
# 2. Nettoyage non destructif
# --------------------------------------------------------------------------
def preparer(df: pd.DataFrame, usd: str = "isoler", paires: str = "garder",
             imputer: bool = True) -> pd.DataFrame:
    """Applique les conventions. Retourne les MÊMES lignes + colonnes de traitement.

    usd     : "isoler" (hors périmètre, pas de conversion) ou "traiter_eur"
              (hypothèse : devise mal étiquetée, prix identiques à la grille EUR).
    paires  : "garder" (FACTURE et AVOIR de même id = 2 opérations) ou
              "doublon" (l'avoir est traité comme un doublon de la facture).
    imputer : reconstituer les coûts manquants avec la grille produit/année.
    """
    if usd not in ("isoler", "traiter_eur"):
        raise ValueError("usd doit valoir 'isoler' ou 'traiter_eur'")
    if paires not in ("garder", "doublon"):
        raise ValueError("paires doit valoir 'garder' ou 'doublon'")

    d = df.copy()
    d["_cout_manquant"] = d["cout_unitaire"].isna()

    # -- Doublons : on garde en priorité la ligne avec coût, puis en EUR, puis FACTURE,
    #    puis la plus ancienne dans le fichier.
    d["_rang"] = (d["_cout_manquant"].astype(int) * 100
                  + (d["devise"] != DEVISE_REF).astype(int) * 10
                  + (d["type_operation"] != "FACTURE").astype(int))
    d = d.sort_values(["_rang", "ligne_source"])
    cle = ["transaction_id"] if paires == "doublon" else ["transaction_id", "type_operation"]
    g = d.groupby(cle, sort=False)
    est_doublon = g.cumcount() > 0
    k_type = g["type_operation"].transform("first")
    k_devise = g["devise"].transform("first")
    k_cout_na = g["_cout_manquant"].transform("first")
    motif = np.select(
        [est_doublon & (d["type_operation"] != k_type),
         est_doublon & (d["devise"] != k_devise),
         est_doublon & (d["_cout_manquant"] != k_cout_na)],
        ["DOUBLON_FACTURE_AVOIR", "DOUBLON_DEVISE", "DOUBLON_COUT_MANQUANT"],
        default="")
    motif = np.where(est_doublon & (motif == ""), "DOUBLON_IDENTIQUE", motif)
    d["motif_exclusion"] = motif

    # -- Annulations puis devises (sur les lignes non déjà exclues)
    libre = d["motif_exclusion"] == ""
    d.loc[libre & (d["statut"] == "ANNULEE"), "motif_exclusion"] = "ANNULEE"
    libre = d["motif_exclusion"] == ""
    if usd == "isoler":
        d.loc[libre & (d["devise"] != DEVISE_REF), "motif_exclusion"] = "DEVISE_HORS_EUR"
    d["retenu"] = d["motif_exclusion"] == ""

    # -- Coûts : reconstitution via la grille produit/année
    grille = (d.dropna(subset=["cout_unitaire"])
                .groupby(["produit", "annee"])["cout_unitaire"].first().to_dict())
    cout_grille = pd.Series([grille.get((p, a)) for p, a in zip(d["produit"], d["annee"])],
                            index=d.index, dtype="float64")
    d["cout_impute"] = d["_cout_manquant"] & imputer & cout_grille.notna()
    d["cout_retenu"] = np.where(d["cout_impute"], cout_grille, d["cout_unitaire"])

    # -- Alertes (information, n'excluent rien)
    types_par_id = d.groupby("transaction_id")["type_operation"].transform("nunique")
    alertes = pd.DataFrame({
        "PAIRE_FACTURE_AVOIR": (types_par_id > 1) & d["retenu"] & (paires == "garder"),
        "DEVISE_ASSIMILEE_EUR": (d["devise"] != DEVISE_REF) & d["retenu"],
        "COUT_IMPUTE": d["cout_impute"] & d["retenu"],
        "COUT_ABSENT": d["cout_retenu"].isna() & d["retenu"],
    })
    d["alerte"] = [";".join(c for c in alertes.columns if row[c])
                   for row in alertes.to_dict("records")]

    # -- Montants retenus (signe : avoir = -1)
    d["signe"] = np.where(d["type_operation"] == "AVOIR", -1, 1)
    d["marge_brute_ligne"] = d["quantite"] * (d["prix_net_unitaire"] - d["cout_retenu"])
    k = d["retenu"].astype(int) * d["signe"]
    d["ca_retenu"] = k * d["ca_brut_ligne"]
    d["marge_retenue"] = k * d["marge_brute_ligne"].fillna(0.0)
    d["ca_pour_marge"] = np.where(d["cout_retenu"].notna(), d["ca_retenu"], 0.0)
    d["qte_retenue"] = k * d["quantite"]

    d = d.sort_values("ligne_source")
    return d.drop(columns=[c for c in d.columns if c.startswith("_")])


# --------------------------------------------------------------------------
# 3. Cascade brut -> retenu (sert au tableau des contrôles et au graphique)
# --------------------------------------------------------------------------
def cascade(d: pd.DataFrame) -> pd.DataFrame:
    """CA et marge à chaque étape de nettoyage, par année (effets chiffrés)."""
    m = d["motif_exclusion"]
    dup = m.str.startswith("DOUBLON")
    ann = m == "ANNULEE"
    dev = m == "DEVISE_HORS_EUR"
    tout = pd.Series(True, index=d.index)
    etapes = [
        ("Brut (toutes lignes, avoirs en positif)", tout, False),
        ("Après retrait des doublons", ~dup, False),
        ("Après exclusion des annulations", ~dup & ~ann, False),
        ("Après avoirs en négatif", ~dup & ~ann, True),
        ("Après isolement hors EUR (= retenu)", ~dup & ~ann & ~dev, True),
    ]
    marge_brute = d["marge_brute_ligne"].fillna(0.0)
    lignes = []
    for nom, masque, avec_signe in etapes:
        s = d["signe"] if avec_signe else 1
        r = {"etape": nom}
        for an in ANNEES:
            sel = masque & (d["annee"] == an)
            r[f"CA_{an}"] = float((d["ca_brut_ligne"] * s)[sel].sum())
            r[f"Marge_{an}"] = float((marge_brute * s)[sel].sum())
        lignes.append(r)
    out = pd.DataFrame(lignes)
    for col in [c for c in out.columns if c != "etape"]:
        out[f"delta_{col}"] = out[col].diff()
    return out


# --------------------------------------------------------------------------
# 4. Indicateurs
# --------------------------------------------------------------------------
def indicateurs(d: pd.DataFrame) -> dict:
    r = d[d["retenu"]]
    annuel = r.groupby("annee").agg(
        ca=("ca_retenu", "sum"), marge=("marge_retenue", "sum"),
        ca_pour_marge=("ca_pour_marge", "sum"), unites=("qte_retenue", "sum"))
    annuel = annuel.reindex(list(ANNEES)).fillna(0.0)
    annuel["taux_marge"] = annuel["marge"] / annuel["ca_pour_marge"].replace(0, np.nan)

    mensuel = (r.groupby(["annee", "mois"])
                 .agg(ca=("ca_retenu", "sum"), marge=("marge_retenue", "sum"))
                 .reset_index())
    mensuel["annee"] = mensuel["annee"].astype(int)
    mensuel["mois"] = mensuel["mois"].astype(int)

    ca24, ca25 = annuel.loc[2024, "ca"], annuel.loc[2025, "ca"]
    croissance = ca25 / ca24 - 1 if ca24 else np.nan

    top_clients = (r.groupby("client_id")["ca_retenu"].sum()
                     .nlargest(5).rename("ca").reset_index())
    prod = r.groupby("produit").agg(ca=("ca_retenu", "sum"), marge=("marge_retenue", "sum"),
                                    ca_pour_marge=("ca_pour_marge", "sum"))
    prod["taux_marge"] = prod["marge"] / prod["ca_pour_marge"].replace(0, np.nan)
    top_produits = prod.nlargest(5, "marge").reset_index()

    return {"annuel": annuel, "mensuel": mensuel, "croissance": croissance,
            "top_clients": top_clients, "top_produits": top_produits}


def effet_prix_volume(d: pd.DataFrame) -> dict:
    """Sépare la croissance du CA en effet volume et effet prix.

    CA 2025 revalorisé aux prix catalogue 2024 (par produit) : l'écart avec le
    CA 2024 mesure le volume ; le reste est l'effet prix (baisse de la grille).
    """
    r = d[d["retenu"]]
    p = r.groupby(["produit", "annee"])["prix_unitaire"].first().unstack()
    if not set(ANNEES) <= set(p.columns):
        return {}
    r25 = r[r["annee"] == 2025]
    ca25_prix24 = float((r25["signe"] * r25["quantite"] * r25["produit"].map(p[2024])
                         * (1 - r25["remise_pct"] / 100)).sum())
    ca24 = float(r.loc[r["annee"] == 2024, "ca_retenu"].sum())
    ca25 = float(r25["ca_retenu"].sum())
    vol = ca25_prix24 / ca24 - 1
    tot = ca25 / ca24 - 1
    return {"croissance_totale": tot, "croissance_volume": vol,
            "effet_prix": (1 + tot) / (1 + vol) - 1, "ca25_aux_prix_2024": ca25_prix24}


# --------------------------------------------------------------------------
# 5. Contrôles de fiabilité
# --------------------------------------------------------------------------
def ca_independant(df: pd.DataFrame, usd: str, paires: str) -> dict:
    """Recalcul du CA par une méthode volontairement différente (boucle Python pure,
    dictionnaires) pour recouper le calcul pandas de `preparer`."""
    groupes: dict = {}
    for row in df.itertuples(index=False):
        cle = (row.transaction_id,) if paires == "doublon" else (row.transaction_id, row.type_operation)
        groupes.setdefault(cle, []).append(row)
    total: dict = {}
    for lignes in groupes.values():
        lignes.sort(key=lambda r: (r.cout_unitaire != r.cout_unitaire,
                                   r.devise != DEVISE_REF, r.type_operation != "FACTURE",
                                   r.ligne_source))
        r = lignes[0]
        if r.statut == "ANNULEE":
            continue
        if r.devise != DEVISE_REF and usd == "isoler":
            continue
        signe = -1 if r.type_operation == "AVOIR" else 1
        an = int(str(r.date_transaction)[:4])
        total[an] = total.get(an, 0.0) + signe * r.quantite * r.prix_unitaire * (1 - r.remise_pct / 100)
    return total


def controles(brut: pd.DataFrame, d: pd.DataFrame, usd: str = "isoler",
              paires: str = "garder", imputer: bool = True) -> pd.DataFrame:
    """Tableau des contrôles : risque, méthode, résultat, traitement, effet chiffré."""
    c = cascade(d)
    lignes = []

    def ajouter(num, nom, risque, methode, resultat, traitement, statut,
                e_ca24=0.0, e_ca25=0.0, e_marge=0.0):
        lignes.append({"n": num, "controle": nom, "risque": risque, "methode": methode,
                       "resultat": resultat, "traitement": traitement, "statut": statut,
                       "effet_CA_2024": round(float(e_ca24), 2),
                       "effet_CA_2025": round(float(e_ca25), 2),
                       "effet_marge_total": round(float(e_marge), 2)})

    def eff(i, col):
        return float(c.loc[i, col]) if pd.notna(c.loc[i, col]) else 0.0

    # 1. Doublons ----------------------------------------------------------
    m = d["motif_exclusion"]
    dup = m.str.startswith("DOUBLON")
    detail = m[dup].value_counts().to_dict()
    ajouter(1, "Doublons de transaction_id",
            "Le même enregistrement compté plusieurs fois gonfle CA et marge.",
            "Regroupement par transaction_id (et type d'opération) ; comparaison des lignes d'un même groupe champ par champ.",
            f"{int(dup.sum())} lignes écartées ({d.loc[dup, 'transaction_id'].nunique()} id concernés) ; "
            f"années concernées : {sorted(d.loc[dup, 'annee'].dropna().astype(int).unique().tolist())} ; détail : {detail}.",
            "Une seule ligne conservée par id (priorité : coût renseigné, EUR, FACTURE, plus ancienne). Lignes écartées conservées et signalées.",
            "Anomalie traitée" if dup.any() else "Aucune anomalie",
            eff(1, "delta_CA_2024"), eff(1, "delta_CA_2025"),
            eff(1, "delta_Marge_2024") + eff(1, "delta_Marge_2025"))

    # 2. Annulations ---------------------------------------------------------
    ann = m == "ANNULEE"
    ajouter(2, "Opérations annulées",
            "Compter une vente annulée surestime le CA.",
            "Filtre sur statut = ANNULEE parmi les lignes non doublons.",
            f"{int(ann.sum())} lignes annulées (types : {d.loc[ann, 'type_operation'].value_counts().to_dict()}).",
            "Exclues du CA et de la marge, conservées et signalées.",
            "Anomalie traitée" if ann.any() else "Aucune anomalie",
            eff(2, "delta_CA_2024"), eff(2, "delta_CA_2025"),
            eff(2, "delta_Marge_2024") + eff(2, "delta_Marge_2025"))

    # 3. Avoirs ---------------------------------------------------------------
    av = d[d["retenu"] & (d["type_operation"] == "AVOIR")]
    fa = d[d["retenu"] & (d["type_operation"] == "FACTURE")]
    rapp = av.merge(fa, on=["client_id", "produit", "quantite"], suffixes=("_av", "_fa"))
    n_rapp = rapp.loc[rapp["date_fa"] <= rapp["date_av"], "transaction_id_av"].nunique()
    ajouter(3, "Avoirs (signe)",
            "Quantités et prix des avoirs sont positifs dans le fichier : additionnés tels quels, ils augmentent le CA.",
            "Recensement des AVOIR ; test de signe ; rapprochement avec une facture antérieure de même client, produit et quantité.",
            f"{len(av)} avoirs, tous quantités/prix positifs ; {n_rapp} rapprochables d'une facture antérieure (le fichier n'a pas de référence à la facture d'origine : lien non prouvé).",
            "Avoirs comptés en négatif (CA, coût, marge), à la date de l'avoir.",
            "Anomalie traitée" if len(av) else "Aucune anomalie",
            eff(3, "delta_CA_2024"), eff(3, "delta_CA_2025"),
            eff(3, "delta_Marge_2024") + eff(3, "delta_Marge_2025"))

    # 4. Devises ----------------------------------------------------------------
    hors = d[d["devise"] != DEVISE_REF]
    hors_ret = hors[hors["retenu"] | (hors["motif_exclusion"] == "DEVISE_HORS_EUR")]
    prix_id = bool(((brut.loc[brut["devise"] != DEVISE_REF, "prix_unitaire"].isin(
        brut.loc[brut["devise"] == DEVISE_REF, "prix_unitaire"])).all())) if len(hors) else True
    res4 = (f"{len(hors)} lignes hors EUR ({hors['devise'].value_counts().to_dict()}), dont {len(hors_ret)} après retrait des doublons ; "
            f"aucun taux de change dans le fichier ; prix identiques à la grille EUR : {'oui' if prix_id else 'non'}.")
    trait4 = ("Lignes hors EUR isolées (aucune conversion sans taux justifié) ; montant présenté à part."
              if usd == "isoler" else
              "HYPOTHÈSE : devise mal étiquetée, lignes traitées comme EUR (non prouvé).")
    ajouter(4, "Devises mélangées",
            "Additionner des USD et des EUR sans conversion fausse les totaux.",
            "Comptage par devise ; comparaison des prix unitaires USD avec la grille EUR ; recherche d'un taux dans les données.",
            res4, trait4,
            "Cas non résolu" if len(hors) else "Aucune anomalie",
            eff(4, "delta_CA_2024"), eff(4, "delta_CA_2025"),
            eff(4, "delta_Marge_2024") + eff(4, "delta_Marge_2025"))

    # 5. Coûts manquants --------------------------------------------------------
    na_brut = int(brut["cout_unitaire"].isna().sum())
    na_ret = d[d["retenu"] & d["cout_unitaire"].isna()]
    marge_imp = float(na_ret["marge_retenue"].sum()) if imputer else 0.0
    unique = (brut.dropna(subset=["cout_unitaire"]).groupby(["produit", "annee"])["cout_unitaire"].nunique().max())
    ajouter(5, "Coûts unitaires manquants",
            "Un coût vide traité comme 0 gonfle la marge ; l'exclure sans le dire la fausse aussi.",
            "Comptage des cout_unitaire vides ; test d'unicité du coût par produit et par année (condition pour reconstituer).",
            f"{na_brut} coûts vides dans le fichier, {len(na_ret)} sur lignes retenues ; coût unique par produit/année : {'oui' if unique == 1 else 'non'}.",
            ("Coût reconstitué via la grille produit/année et signalé (COUT_IMPUTE)." if imputer
             else "Pas d'imputation : ces lignes restent dans le CA mais sont exclues du calcul de la marge."),
            "Anomalie traitée" if na_brut else "Aucune anomalie",
            0.0, 0.0, marge_imp)

    # 6. Cohérence prix / coût / quantité / remise ------------------------------
    anom = {"quantité ≤ 0": int((brut["quantite"] <= 0).sum()),
            "prix ≤ 0": int((brut["prix_unitaire"] <= 0).sum()),
            "coût ≤ 0": int((brut["cout_unitaire"] <= 0).sum()),
            "remise hors 0-100 %": int(((brut["remise_pct"] < 0) | (brut["remise_pct"] > 100)).sum()),
            "prix net < coût": int((brut["prix_net_unitaire"] < brut["cout_unitaire"]).sum())}
    gr = brut.groupby(["produit", "annee"]).agg(p=("prix_unitaire", "nunique"), c=("cout_unitaire", "nunique"))
    multi = int(((gr["p"] > 1) | (gr["c"] > 1)).sum())
    prix = brut.groupby(["produit", "annee"])["prix_unitaire"].first().unstack()
    obs = ""
    if set(ANNEES) <= set(prix.columns):
        ratio = prix[2025] / prix[2024]
        obs = f" Observation : prix 2025 = {ratio.min():.2f} à {ratio.max():.2f} x prix 2024 selon les produits (effet prix à isoler de l'effet volume)."
    total_anom = sum(anom.values()) + multi
    ajouter(6, "Cohérence prix, coût, quantité, remise",
            "Valeurs aberrantes ou grille tarifaire incohérente faussent CA et marge.",
            "Tests de bornes ; unicité du prix et du coût par produit et par année ; comparaison des grilles 2024/2025.",
            f"Anomalies : {anom} ; combinaisons produit/année avec plusieurs prix ou coûts : {multi}.{obs}",
            "Aucune correction nécessaire." if total_anom == 0 else "À examiner ligne par ligne.",
            "Aucune anomalie" if total_anom == 0 else "À examiner")

    # 7. Dates et identifiants ----------------------------------------------------
    annee_id = pd.to_numeric(brut["transaction_id"].str[1:5], errors="coerce")
    n_date = int(brut["date"].isna().sum())
    n_hors = int((~brut["annee"].isin(ANNEES)).sum())
    n_incoh = int((annee_id != brut["annee"].astype("float")).sum())
    n_fmt = int((~brut["transaction_id"].str.match(r"^T\d{4}-\d{5}$")).sum())
    tot7 = n_date + n_hors + n_incoh + n_fmt
    ajouter(7, "Dates, périmètre et identifiants",
            "Dates invalides ou hors période = lignes mal affectées à un exercice.",
            "Lecture stricte au format AAAA-MM-JJ ; test d'appartenance à 2024-2025 ; cohérence année de l'id / année de la date ; format de l'id.",
            f"Dates illisibles : {n_date} ; hors 2024-2025 : {n_hors} ; année id ≠ année date : {n_incoh} ; format d'id invalide : {n_fmt}.",
            "Aucune correction nécessaire." if tot7 == 0 else "À examiner ligne par ligne.",
            "Aucune anomalie" if tot7 == 0 else "À examiner")

    # 8. Paires FACTURE / AVOIR de même id ---------------------------------------
    ids_mixtes = brut.groupby("transaction_id")["type_operation"].nunique()
    ids_mixtes = ids_mixtes[ids_mixtes > 1].index
    if paires == "garder":
        p = d[d["retenu"] & d["transaction_id"].isin(ids_mixtes)]
        net = {an: float(p.loc[p["annee"] == an, "ca_retenu"].sum()) for an in ANNEES}
        ca_fact = float(p.loc[p["type_operation"] == "FACTURE", "ca_retenu"].sum())
        ajouter(8, "Paires FACTURE/AVOIR de même identifiant",
                "Un avoir qui reprend l'id de sa facture est soit une vraie annulation, soit un doublon mal typé : deux lectures opposées.",
                "Recherche des id portés à la fois par une FACTURE et un AVOIR (mêmes client, produit, quantité).",
                f"{len(ids_mixtes)} id concernés ; effet net des deux lignes conservées : {net[2024] + net[2025]:.0f} € ; "
                f"si l'avoir était un doublon, le CA augmenterait de {ca_fact:.0f} €.",
                "Les deux lignes sont conservées (facture + avoir = effet net nul). CAS NON RÉSOLU : à confirmer auprès de la comptabilité.",
                "Cas non résolu" if len(ids_mixtes) else "Aucune anomalie",
                net[2024], net[2025], 0.0)
    else:
        sel = d["motif_exclusion"] == "DOUBLON_FACTURE_AVOIR"
        ajouter(8, "Paires FACTURE/AVOIR de même identifiant",
                "Un avoir qui reprend l'id de sa facture est soit une vraie annulation, soit un doublon mal typé.",
                "Recherche des id portés à la fois par une FACTURE et un AVOIR.",
                f"{len(ids_mixtes)} id concernés ; {int(sel.sum())} avoirs traités comme doublons (hypothèse).",
                "HYPOTHÈSE : l'avoir est un doublon de la facture, donc écarté. CAS NON RÉSOLU.",
                "Cas non résolu" if len(ids_mixtes) else "Aucune anomalie")

    # 9. Réconciliation par recalcul indépendant ----------------------------------
    indep = ca_independant(brut, usd, paires)
    kpi = indicateurs(d)["annuel"]["ca"]
    ecart = max(abs(indep.get(an, 0.0) - float(kpi[an])) for an in ANNEES)
    ecart_cascade = max(abs(float(c.iloc[-1][f"CA_{an}"]) - float(kpi[an])) for an in ANNEES)
    ok = ecart < 0.01 and ecart_cascade < 0.01
    ajouter(9, "Réconciliation par recalcul indépendant",
            "Un programme sans erreur peut produire un résultat faux : il faut recouper le calcul par une autre voie.",
            "Recalcul du CA par une boucle Python pure (dictionnaires) à partir du CSV brut ; comparaison avec le CA du programme et avec la cascade brut→retenu.",
            f"Écart max CA : {ecart:.4f} € (recalcul indépendant) et {ecart_cascade:.4f} € (cascade).",
            "Aucun traitement : contrôle de cohérence interne.",
            "Contrôle OK" if ok else "ÉCART À ANALYSER")
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# 6. Sensibilité aux hypothèses
# --------------------------------------------------------------------------
def sensibilite(brut: pd.DataFrame) -> pd.DataFrame:
    """CA, croissance et taux de marge selon les hypothèses retenues."""
    scenarios = [
        ("Référence (conventions retenues)", dict(PARAMS_DEFAUT)),
        ("Lignes USD traitées comme EUR", {**PARAMS_DEFAUT, "usd": "traiter_eur"}),
        ("Paires FACTURE/AVOIR = doublons", {**PARAMS_DEFAUT, "paires": "doublon"}),
        ("Sans imputation des coûts manquants", {**PARAMS_DEFAUT, "imputer": False}),
    ]
    lignes = []
    for nom, p in scenarios:
        d = preparer(brut, **p)
        k = indicateurs(d)
        a = k["annuel"]
        lignes.append({"scenario": nom, "CA_2024": a.loc[2024, "ca"], "CA_2025": a.loc[2025, "ca"],
                       "croissance": k["croissance"], "taux_marge_2024": a.loc[2024, "taux_marge"],
                       "taux_marge_2025": a.loc[2025, "taux_marge"]})
    c = cascade(preparer(brut, **PARAMS_DEFAUT)).iloc[0]
    lignes.append({"scenario": "Brut, sans aucun traitement", "CA_2024": c["CA_2024"], "CA_2025": c["CA_2025"],
                   "croissance": c["CA_2025"] / c["CA_2024"] - 1,
                   "taux_marge_2024": np.nan, "taux_marge_2025": np.nan})
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------
# 7. Exports et lancement en ligne de commande
# --------------------------------------------------------------------------
COLONNES_EXPORT = ["ligne_source", "transaction_id", "date_transaction", "client_id", "produit",
                   "quantite", "prix_unitaire", "cout_unitaire", "remise_pct", "devise",
                   "type_operation", "statut", "retenu", "motif_exclusion", "alerte", "signe",
                   "cout_retenu", "cout_impute", "ca_retenu", "marge_retenue"]


def exporter_csv(df: pd.DataFrame) -> bytes:
    """CSV lisible par Excel français (séparateur ; et virgule décimale)."""
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")


def main(chemin: str = "transactions_nova.csv", sortie: str = "sorties") -> None:
    brut = charger(chemin)
    d = preparer(brut, **PARAMS_DEFAUT)
    k = indicateurs(d)
    ctrl = controles(brut, d, **PARAMS_DEFAUT)
    out = Path(sortie)
    out.mkdir(exist_ok=True)
    (out / "nova_nettoye.csv").write_bytes(exporter_csv(d[COLONNES_EXPORT]))
    (out / "tableau_controles.csv").write_bytes(exporter_csv(ctrl))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(k["annuel"].round(2).to_string())
    print(f"\nCroissance 2024-2025 : {k['croissance']:.2%}")
    print("\nTop 5 clients :\n", k["top_clients"].round(0).to_string(index=False))
    print("\nTop 5 produits (marge) :\n", k["top_produits"].round(2).to_string(index=False))
    print("\nContrôles :\n", ctrl[["n", "controle", "statut", "effet_CA_2024", "effet_CA_2025",
                                    "effet_marge_total"]].to_string(index=False))
    print(f"\nFichiers écrits dans {out.resolve()}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
