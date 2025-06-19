import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from one import (
    predict_risk_matrix_outputs,
    CausesActions_df as df_CausesActions,
    HSE_df as df_HSE,
    nettoyer_texte , generer_registre_des_risques_depuis_excel,afficher_risques_hse
)

# === Chargement des modèles ===
colonnes_features = joblib.load("colonnes_features.pkl")
colonnes_risques = joblib.load("colonnes_risques.pkl")

model_risques = joblib.load("model_risques.pkl")
model_depassement = joblib.load("model_depassement.pkl")
model_ecart_duree = joblib.load("model_ecart_duree.pkl")
model_cout_reel = joblib.load("model_cout_reel.pkl")
model_duree_reelle = joblib.load("model_duree_reelle.pkl")
model_pg = joblib.load("model_Probabilité_Gravité.pkl")
model_hse = joblib.load("model_hse.pkl")

# === Prédiction des risques binaires ===
def predict_risques(data_dict: dict):
    df_input = pd.DataFrame([data_dict])[colonnes_features]
    y_pred = model_risques.predict(df_input)
    return pd.DataFrame(y_pred, columns=colonnes_risques), df_input

# === Prédiction des impacts ===
def predict_impacts(X_input: pd.DataFrame, y_pred_risques_binaires: np.ndarray) -> dict:
    df_risques = pd.DataFrame(y_pred_risques_binaires, columns=colonnes_risques)
    X_full = pd.concat([X_input.reset_index(drop=True), df_risques], axis=1)

    return {
        "Dépassement Coût": model_depassement.predict(X_full)[0],
        "Écart Durée": model_ecart_duree.predict(X_full)[0],
        "Coût Réel": model_cout_reel.predict(X_full)[0],
        "Durée Réelle": model_duree_reelle.predict(X_full)[0]
    }

# === Simulation Monte Carlo ===
def simulation_monte_carlo(data_reelle, titre="Dépassement de Coût", unite="$", couleur="blue"):
    data = np.array(data_reelle.dropna())
    if len(data) < 10:
        moyenne = data[0] if len(data) > 0 else 1.0
        ecart_type = abs(moyenne) * 0.15 if moyenne != 0 else 1.0
        data = np.random.normal(loc=moyenne, scale=ecart_type, size=1000)
    else:
        data = np.random.choice(data, size=1000, replace=True)
    return np.sort(data)

# === Barres comparatives ===
def afficher_barres_comparatives(st, prevu, valeurs_simulees, titre, unite, couleur_prevu, couleur_simule):
    labels = [f"{titre} Prévu", "Scénario P20", "Scénario P50", "Scénario P80"]
    valeurs = [prevu] + valeurs_simulees
    couleurs = [couleur_prevu] + [couleur_simule]*3

    valeurs_affichables = [v / 1_000_000 if unite.lower() in ["$", "usd"] else v for v in valeurs]
    ylabel = "Montant (en millions $)" if unite.lower() in ["$", "usd"] else f"Durée ({unite})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, valeurs_affichables, color=couleurs)
    ax.set_title(f"📊 {titre} Prévu vs Simulation Monte Carlo")
    ax.set_ylabel(ylabel)
    ax.grid(axis='y')
    st.pyplot(fig)

# === Affichage matrice des risques ===
from one import plot_risk_matrix, predict_risk_matrix_outputs as model_predict_pg

def predict_risk_matrix_outputs(index_projet: int, X_input: pd.DataFrame, y_pred_binaires: np.ndarray):
    # Créer un dataframe des risques actifs
    df_risques = pd.DataFrame(y_pred_binaires, columns=colonnes_risques)
    risques_actifs = df_risques.columns[(df_risques.loc[index_projet] == 1)].tolist()

    lignes = []
    for risque in risques_actifs:
        ligne = X_input.iloc[index_projet].copy()
        for r in colonnes_risques:
            ligne[r] = 1 if r == risque else 0
        ligne["Risque"] = risque
        lignes.append(ligne)

    df_eval = pd.DataFrame(lignes)
    pred_df = model_predict_pg(df_eval, model_pg)

    # Afficher la matrice
    plot_risk_matrix(index_projet + 1, pred_df)


# === Générer registre des risques ===
def generer_registre_des_risques_depuis_excel() -> pd.DataFrame:
    import unicodedata

    def nettoyer_texte(texte):
        if pd.isna(texte):
            return ''
        texte = str(texte).strip().lower()
        texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8')
        return texte.replace("’", "'").replace("–", "-").replace("œ", "oe").strip()

    # Chargement des feuilles
    df_risques = pd.read_excel("project risk management.xlsx", sheet_name="Risques")
    df_causes_actions = pd.read_excel("project risk management.xlsx", sheet_name="Causes et Actions preventive")

    # Nettoyage
    df_risques['Risque_clean'] = df_risques['Risques'].apply(nettoyer_texte)
    df_causes_actions['Risque_clean'] = df_causes_actions['Risque'].apply(nettoyer_texte)

    # Jointure
    df_registre = pd.merge(
        df_risques,
        df_causes_actions[['Risque_clean', 'Cause', 'Action Préventive']],
        on='Risque_clean',
        how='left'
    )

    # Calcul de l'impact
    df_registre['Impact'] = df_registre['Probabilité'] * df_registre['Gravité']

    # Marquage des risques critiques
    def marquer_critique(row):
        if row['Impact'] == 9:
            return f"{row['Risques']} (critique)"
        elif row['Impact'] == 6 and row['Gravité'] == 3:
            return f"{row['Risques']} (critique)"
        else:
            return row['Risques']

    df_registre['Risque'] = df_registre.apply(marquer_critique, axis=1)

    # Colonnes finales
    df_registre_final = df_registre[[
        'Risque', 'Probabilité', 'Gravité', 'Impact', 'Cause', 'Action Préventive'
    ]]

    df_registre_final.fillna("❓ Non trouvée", inplace=True)

    return df_registre_final


# === Risques HSE par type de projet ===
def afficher_risques_hse(st, type_projet_choisi: str):
    resultats = df_HSE[df_HSE['Type de Projet'] == type_projet_choisi][
        ['les Risques ', 'La criticite ', 'Causes', 'Actions correctives']
    ]
    if resultats.empty:
        st.warning(f"Aucun risque HSE trouvé pour : {type_projet_choisi}")
        return

    criticite_counts = resultats['La criticite '].value_counts().sort_index()
    total = criticite_counts.sum()
    labels = [f'Criticité {int(c)} ({round(100 * n / total)}%)' for c, n in criticite_counts.items()]
    couleurs = {1: 'green', 2: 'orange', 3: 'red'}

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(criticite_counts, labels=labels,
           colors=[couleurs[int(c)] for c in criticite_counts.index],
           autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title(f"Répartition des risques HSE - {type_projet_choisi}")
    ax.axis('equal')
    st.pyplot(fig)

    st.dataframe(resultats.rename(columns={
        'les Risques ': 'Risque',
        'La criticite ': 'Criticité',
        'Causes': 'Causes',
        'Actions correctives': 'Actions'
    }).reset_index(drop=True))
def generer_registre_des_risques_from_prediction(X_input: pd.DataFrame, y_pred_binaires: np.ndarray) -> pd.DataFrame:
    registre = []
    df_risques = pd.DataFrame(y_pred_binaires, columns=colonnes_risques)

    for i, ligne_risque in df_risques.iterrows():
        risques_actifs = ligne_risque[ligne_risque == 1].index.tolist()

        for risque in risques_actifs:
            ligne_eval = X_input.iloc[i].copy()
            for r in colonnes_risques:
                ligne_eval[r] = 1 if r == risque else 0

            ligne_eval['Risque'] = risque
            X_eval_pg = pd.DataFrame([ligne_eval])[colonnes_features + colonnes_risques]
            preds_pg = model_pg.predict(X_eval_pg)[0]
            proba = int(np.clip(round(preds_pg[0]), 1, 3))
            grav = int(np.clip(round(preds_pg[1]), 1, 3))
            impact = proba * grav

            criticite = "Critique" if (impact == 9 or (impact == 6 and grav == 3 and proba == 2)) else "Non critique"

            Risque_clean = nettoyer_texte(risque)
            df_CausesActions['Risque_clean'] = df_CausesActions['Risque'].apply(nettoyer_texte)

            cause = df_CausesActions['Cause'].values[0] if not df_CausesActions.empty else "❓ Non trouvée"
            action = df_CausesActions['Action Préventive'].values[0] if not df_CausesActions.empty else "❓ Non trouvée"

            registre.append({
                'Risque': f"{risque} (Critique)" if criticite == "Critique" else risque,
                'Probabilité': proba,
                'Gravité': grav,
                'Impact': impact,
                'Cause': cause,
                'Action Préventive': action
            })

    return pd.DataFrame(registre).sort_values(by='Impact', ascending=False)
