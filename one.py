# Système et fichiers
import os
import joblib
import unicodedata

# Traitement de données
import pandas as pd
import numpy as np

# Visualisation
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import plotly
import textwrap

# Interfaces
import streamlit as st

# Statistiques et traitement scientifique
import scipy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    mean_squared_error,
    r2_score,
)

# Modèles de machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

# Simulation
import simpy
import textwrap
# ***READING DATA***"""

file_path = 'project risk management.xlsx'
sheets = ['Projets', 'Risques', 'Causes et Actions preventive','les Risques HSE ']
data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheets}
for sheet, df in data.items():
    print(f"Premières lignes de la feuille {sheet}:")
    print(df.head(), "\n")

df_projets = data['Projets']
null_counts = df_projets.isnull().sum()
print(null_counts)
df_Risques = data['Risques']
null_counts = df_Risques.isnull().sum()
print(null_counts)
df_CausesActions = data['Causes et Actions preventive']
null_counts = df_CausesActions.isnull().sum()
print(null_counts)
df_HSE = data['les Risques HSE ']
null_counts = df_HSE.isnull().sum()
print(null_counts)

projets_df = data['Projets']
risques_df = data['Risques']
CausesActions_df = data['Causes et Actions preventive']
HSE_df=data['les Risques HSE ']
print("Informations sur 'Projets' :")
print(projets_df.info())
print("\nInformations sur 'Risques' :")
print(risques_df.info())
print("\nInformations sur 'les Risques HSE' :")
print(HSE_df.info())
print("\nInformations sur 'Causes et Actions preventive' :")
print(CausesActions_df.info())

for sheet, df in data.items():
    print(f"\n=== Sheet: {sheet} ===")
    print(df.head())

for sheet_name, df in data.items():
    print(f"Columns in sheet '{sheet_name}':")
    print(df.columns.tolist())
    print("\n" + "="*50)

projets_df = data['Projets']
risques_df = data['Risques']
CausesActions_df = data['Causes et Actions preventive']
HSE_df=data['les Risques HSE ']
print("types de 'Projets' :")
print(projets_df.dtypes)
print("\ntypes sur 'Risques' :")
print(risques_df.dtypes)
print("\ntypes sur 'Causes et Actions preventive' :")
print(CausesActions_df.dtypes)
print("\ntypes sur 'les Risques HSE' :")
print(HSE_df.dtypes)

print (projets_df.columns)
print (risques_df.columns)
print (CausesActions_df.columns)
print (HSE_df.columns)

# Example: Detect outliers in 'impact_Risque' in sheet 2
Q1 = risques_df['impact_Risque'].quantile(0.25)
Q3 = risques_df['impact_Risque'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outliers = risques_df[
    (risques_df['impact_Risque'] < Q1 - 1.5 * IQR) |
    (risques_df['impact_Risque'] > Q3 + 1.5 * IQR)
]

print("Outliers in impact_Risque:")
print(outliers[['Risques', 'impact_Risque']])

print(risques_df['impact_Risque'].dtype)
print(risques_df['impact_Risque'].isna().sum())

Q1 = risques_df['Probabilité'].quantile(0.25)
Q3 = risques_df['Probabilité'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outliers = risques_df[
    (risques_df['Probabilité'] < Q1 - 1.5 * IQR) |
    (risques_df['Probabilité'] > Q3 + 1.5 * IQR)
]

print("Outliers in Probabilité:")
print(outliers[['Risques', 'Probabilité']])

print(risques_df['Probabilité'].dtype)
print(risques_df['Probabilité'].isna().sum())

Q1 = risques_df['Gravité'].quantile(0.25)
Q3 = risques_df['Gravité'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outliers = risques_df[
    (risques_df['Gravité'] < Q1 - 1.5 * IQR) |
    (risques_df['Gravité'] > Q3 + 1.5 * IQR)
]

print("Outliers in Gravité:")
print(outliers[['Risques', 'Gravité']])

print(risques_df['Gravité'].dtype)
print(risques_df['Gravité'].isna().sum())
print(HSE_df['La criticite '].dtype)
print(HSE_df['La criticite '].isna().sum())

Q1 = HSE_df['La criticite '].quantile(0.25)
Q3 = HSE_df['La criticite '].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outliers = HSE_df[
    (HSE_df['La criticite '] < Q1 - 1.5 * IQR) |
    (HSE_df['La criticite '] > Q3 + 1.5 * IQR)
]

print("Outliers in La criticite:")
print(outliers[['les Risques ', 'La criticite ']])

"""# ***Cleaning Data***"""

xls = pd.ExcelFile('project risk management.xlsx')

# Feuille 1 : Projets
projets_df = pd.read_excel(xls, sheet_name='Projets')

# Encoder 'Risque Niveau' (ordre : faible < moyen < élevé)
niveau_map = {'Faible': 1, 'Moyen': 2, 'Élevé': 3}
projets_df['Risque Niveau'] = projets_df['Risque Niveau'].map(niveau_map)

# One-hot encoding pour 'Type de Projet' et 'Localisation'
projets_df = pd.get_dummies(projets_df, columns=['Type de Projet', 'Localisation'], drop_first=False,dtype=int)

# Feuille 2 : Risques
risques_df = pd.read_excel(xls, sheet_name='Risques')

# Label encoding pour la colonne 'Risques'
le_risques = LabelEncoder()
# Keep full DataFrame, just add the new column
risques_df['Risques'] = le_risques.fit_transform(risques_df['Risques'])
# Affichage pour vérification
print("=== Colonnes de la feuille Projets  ===")
print(projets_df.columns)
print("\n=== Risques  dans la feuille Risques ===")
print(risques_df.head())
print("\n=== Feuille Causes et Actions preventive  ===")
print(CausesActions_df.head())
print("\n=== Feuille les Risques HSE   ===")
print(HSE_df.head())

projets_df['Date Début'] = pd.to_datetime(projets_df['Date Début'])
projets_df['Date Fin'] = pd.to_datetime(projets_df['Date Fin'])
print(projets_df[['Date Début', 'Date Fin']].head())

projets_df['Date Début'] = pd.to_datetime(projets_df['Date Début'])

projets_df['Date Début'] = pd.to_datetime(projets_df['Date Début'])

projets_df["Date Fin"] = pd.to_datetime(projets_df["Date Fin"])

# Exemple : extraire l'année, le mois ou le nombre de jours
projets_df["Debut_annee"] = projets_df['Date Début'].dt.year
projets_df["Debut_mois"] = projets_df['Date Début'].dt.month
projets_df["Debut_jours"] = projets_df['Date Début'].dt.day
projets_df["Fin_annee"] = projets_df["Date Fin"].dt.year
projets_df["Fin_mois"] = projets_df["Date Fin"].dt.month
projets_df["Fin_jours"] = projets_df["Date Fin"].dt.day

projets_df = projets_df.drop(columns=['Date Début'])
projets_df = projets_df.drop(columns=["Date Fin"])

print (projets_df.columns)



# Crée un fichier Excel avec plusieurs feuilles
with pd.ExcelWriter("projet risk management final.xlsx") as writer:
    projets_df.to_excel(writer, sheet_name="Projets", index=False)
    risques_df.to_excel(writer, sheet_name="Risques", index=False)
    CausesActions_df.to_excel(writer, sheet_name="Causes et Actions preventive", index=False)
    HSE_df.to_excel(writer, sheet_name="les Risques HSE ", index=False)

print("Fichier 'projet risk management final.xlsx' créé avec succès.")





# === 1. Charger les données ===
df = pd.read_excel("projet risk management final.xlsx")

# === 2. Définir les features (X) ===
colonnes_features = [
      'Coût Prévu', 'Durée Prévue',
      ' vous avez constaté des problèmes de performance ou de fiabilité des équipements utilisés ?',
       'Le contrôle qualité n’est-il pas systématiquement effectué lors de la réception des matériaux ?',
       'Le chantier ne dispose-t-il pas d’un plan qualité formel et appliqué ?',
       "Estimez-vous que la main-d'œuvre actuellement disponible est insuffisamment qualifiée pour ce projet ?",
       'Un plan de maintenance préventive  n’est-il pas mis en place pour les équipements ?',
       'Pensez-vous que le chantier pourrait connaître des arrêts de plus de 5 jours au cours d’un mois ?',
       'N’existe-t-il pas un plan logistique structuré   pour organiser l’approvisionnement, le stockage et les flux sur le chantier ? ',
       'Type de Projet_Barrage',
       'Type de Projet_Bâtiment', 'Type de Projet_Pont',
       'Type de Projet_Route', 'Type de Projet_Tunnel', 'Localisation_Abidjan',
       'Localisation_Accra', 'Localisation_Adrar', 'Localisation_Alger',
       'Localisation_Annaba', 'Localisation_Blida', 'Localisation_Béjaïa',
       'Localisation_Cairo', 'Localisation_Cape Town', 'Localisation_Chicago',
       'Localisation_Constantine', 'Localisation_Dakar',
       'Localisation_Houston', 'Localisation_Lagos',
       'Localisation_Los Angeles', 'Localisation_Nairobi',
       'Localisation_New York', 'Localisation_Oran', 'Localisation_Seattle',
       'Localisation_Tamanrasset', 'Localisation_Tizi Ouzou',
       'Localisation_Tlemcen', 'Localisation_Tunis', 'Debut_annee',
       'Debut_mois', 'Debut_jours', 'Fin_annee', 'Fin_mois', 'Fin_jours'
       ]

X = df[colonnes_features]

# === 3. Définir les cibles ===
all_targets = [
    'Mauvaise qualité des matériaux ou non-conformité',
       'Délais administratifs (permis, autorisations…)\n',
       'Défaillance d’équipement ou de machines',
       'Conditions imprévues sur le site', 'Calculs erronés des quantités',
       'Retards dans la livraison des matériaux et des équipements',
       'Retards dans l’obtention des dessins ou rapports de travail',
       "Non-disponibilité des matériaux, équipements ou main-d'œuvre (Problèmes d’approvisionnement)",
       'Mauvaise coordination du site',
       'Échec de la construction (mauvaise exécution)',
       'Conditions imprévues du sol',
       'Retard dans le transport du béton prêt à l’emploi ',
       'Répétitions ou reprises de travaux (rework)\n',
       'Conception inadéquate et erreurs de conception',
       'Modifications imprévues multiples du périmètre du projet',
       'fluctuation des prix ',
       'Difficultés financières/défaillance du sous-traitant',
]

risk_columns_classif = all_targets[:21]


# === 4. Split train/test ===
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X, df[all_targets], test_size=0.2, random_state=42)

# === 5. Modèles ===
clf = MultiOutputClassifier(RandomForestClassifier(random_state=42, class_weight='balanced'))
clf.fit(X_train_risk, y_train_risk[risk_columns_classif])

# === 6. Prédiction ===
y_pred_clf = clf.predict(X_test_risk)

# === 7. Évaluation - Classification ===
print("📌 Rapport de classification par risque binaire :")
for i, col in enumerate(risk_columns_classif):
    print(f"\n🔹 {col}")
    print(classification_report(y_test_risk[col], y_pred_clf[:, i]))

# === 8. Calcul de l'accuracy totale ===
accuracy_total = 0
for i, col in enumerate(risk_columns_classif):
    accuracy_total += accuracy_score(y_test_risk[col], y_pred_clf[:, i])

accuracy_total /= len(risk_columns_classif)

print("\n📊 Risk prediction Accuracy totale :", accuracy_total)
joblib.dump(clf, 'model_risques.pkl')        # Sauvegarde
clf = joblib.load('model_risques.pkl')       # Chargement

# Après avoir défini colonnes_features et risk_columns_classif
joblib.dump(colonnes_features, 'colonnes_features.pkl')
joblib.dump(risk_columns_classif, 'colonnes_risques.pkl')


# Sauvegarder les modèles
joblib.load("colonnes_features.pkl")
joblib.load("colonnes_risques.pkl")
"""# ***LES DEPASSEMENTS***"""




# === 1. Fusion des données projet + risques prédits ===
df = pd.read_excel("projet risk management final.xlsx")

risks_pred_df = pd.DataFrame(y_pred_clf, columns=risk_columns_classif)

X_risk_impact = pd.concat([
    X_test_risk.reset_index(drop=True),  # Infos projet
    risks_pred_df.reset_index(drop=True)  # Risques prédits
], axis=1)
# === 1. Créer les variables pour les prédictions ===
y_pred_depassement = []
y_pred_ecart_duree = []
y_pred_duree_reelle = []
y_pred_cout_reel = []

# === 2. Variables cibles à prédire ===
targets_impact = [
    'Dépassement Coût',
    'Écart Durée',
    'Durée Réelle',
    'Coût Réel'
]

# === 3. Boucle d'entraînement, sauvegarde et chargement ===
for target in targets_impact:
    print(f"\n🎯 Modèle de prédiction pour : {target}")
    y_target = df.loc[X_test_risk.index, target]  # Valeurs réelles

    # Entraînement
    model = RandomForestRegressor(random_state=42)
    model.fit(X_risk_impact, y_target)

    # Prédiction
    y_pred = model.predict(X_risk_impact)

    # Évaluation
    print(f"R2 Score: {r2_score(y_target, y_pred):.3f}")
    rmse = np.sqrt(mean_squared_error(y_target, y_pred))
    print(f"RMSE: {rmse:.2f}")

    # Sauvegarde et chargement du modèle
    if target == 'Dépassement Coût':
        joblib.dump(model, 'model_depassement.pkl')
        model_depassement = joblib.load('model_depassement.pkl')
        y_pred_depassement = y_pred
    elif target == 'Écart Durée':
        joblib.dump(model, 'model_ecart_duree.pkl')
        model_ecart_duree = joblib.load('model_ecart_duree.pkl')
        y_pred_ecart_duree = y_pred
    elif target == 'Durée Réelle':
        joblib.dump(model, 'model_duree_reelle.pkl')
        model_duree_reelle = joblib.load('model_duree_reelle.pkl')
        y_pred_duree_reelle = y_pred
    elif target == 'Coût Réel':
        joblib.dump(model, 'model_cout_reel.pkl')
        model_cout_reel = joblib.load('model_cout_reel.pkl')
        y_pred_cout_reel = y_pred



# === Fonction Simulation Monte Carlo ===
def simulation_monte_carlo(data_reelle, titre="Dépassement de Coût", unite="$", couleur="blue"):
    data = np.array(data_reelle.dropna())

    # ✅ Si données insuffisantes, générer des données artificielles avec bruit gaussien
    if len(data) < 10:
        print(f"⚠️ Données insuffisantes ({len(data)} valeurs). Simulation avec bruit artificiel.")
        moyenne = data[0] if len(data) > 0 else 1.0  # valeur par défaut
        ecart_type = abs(moyenne) * 0.15 if moyenne != 0 else 1.0
        data = np.random.normal(loc=moyenne, scale=ecart_type, size=1000)
    else:
        data = np.random.choice(data, size=1000, replace=True)

    simulations_sorted = np.sort(data)
    probabilites = np.linspace(0, 1, len(simulations_sorted))

    if unite.lower() in ["$", "$"]:
        x_data = simulations_sorted / 1_000_000
        x_label = f"{titre} (en millions $)"
        format_value = lambda v: f"{v/1_000_000:.2f} M $"
    elif unite.lower() in ["jours", "jour", "days"]:
        x_data = simulations_sorted
        x_label = f"{titre} (jours)"
        format_value = lambda v: f"{int(v)} jours"
    else:
        x_data = simulations_sorted
        x_label = f"{titre} ({unite})"
        format_value = lambda v: f"{v:.2f} {unite}"

    plt.figure(figsize=(8, 5))
    plt.plot(x_data, probabilites, color=couleur, lw=2)

    seuils = [0.2, 0.5, 0.8]
    couleurs = ['orange', 'green', 'red']
    resultats = []

    for s, c in zip(seuils, couleurs):
        value = simulations_sorted[int(s * len(simulations_sorted))]
        x_val = value / 1_000_000 if unite.lower() in ["$", "$"] else value
        resultats.append(value)
        plt.axhline(y=s, color=c, linestyle='--')
        plt.text(x_val, s + 0.02, f'{int(s*100)}% ≤ {format_value(value)}', color=c, fontweight='bold')

    plt.title(f"📈 Simulation Monte Carlo - {titre}")
    plt.xlabel(x_label)
    plt.ylabel("Probabilité cumulée")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return resultats


# === Fonction d’affichage barres comparatives ===
def afficher_barres_comparatives(prevu, valeurs_simulees, titre, unite, couleur_prevu, couleur_simule):
    labels = [f"{titre} Prévu", "Scénario 20%", "Scénario 50%", "Scénario 80%"]
    valeurs = [prevu] + valeurs_simulees
    couleurs = [couleur_prevu] + [couleur_simule]*3

    if unite.lower() in ["$", "$"]:
        valeurs_affichables = [v / 1_000_000 for v in valeurs]
        ylabel = "Montant (en millions $)"
    else:
        valeurs_affichables = valeurs
        ylabel = f"Durée ({unite})"

    plt.figure(figsize=(8, 5))
    plt.bar(labels, valeurs_affichables, color=couleurs)
    plt.title(f"📊 {titre} Prévu vs Réels (Simulation Monte Carlo)")
    plt.ylabel(ylabel)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# === Application projet par projet ===
for index, row in df.iterrows():
    print(f"\n📌 Projet {index + 1}")

    # --- Monte Carlo sur le coût ---
    valeurs_cout = simulation_monte_carlo(df['Dépassement Coût'], "Dépassement de Coût", "$", "green")
    couts_totaux = [row['Coût Prévu'] + v for v in valeurs_cout]
    afficher_barres_comparatives(row['Coût Prévu'], couts_totaux, "Coût", "$", "lightgreen", "green")

    # --- Monte Carlo sur la durée ---
    valeurs_duree = simulation_monte_carlo(df['Écart Durée'], "Écart de Durée", "jours", "blue")
    durees_totales = [row['Durée Prévue'] + v for v in valeurs_duree]
    afficher_barres_comparatives(row['Durée Prévue'], durees_totales, "Durée", "jours", "lightblue", "blue")

    # Limiter à 1 ou 2 projets si test
    if index == 2:
        break

 
# pour une meilleure visualisation

# Moyennes des prédictions
mean = [np.mean(y_pred_depassement), np.mean(y_pred_ecart_duree)]

# Matrice de covariance calculée à partir des prédictions
cov_matrix = np.cov(y_pred_depassement, y_pred_ecart_duree)

# Simulation multivariée (1000 simulations)
simulations = np.random.multivariate_normal(mean, cov_matrix, size=1000)

cout_simules = simulations[:, 0]
duree_simulee = simulations[:, 1]

# Visualisation

plt.figure(figsize=(10, 6))
sns.kdeplot(x=cout_simules, y=duree_simulee, cmap="viridis", fill=True, thresh=0.05, levels=100)
plt.scatter(cout_simules, duree_simulee, s=10, alpha=0.3, color='blue', label='Simulations Monte Carlo')

plt.title("Simulation Monte Carlo multivariée : Dépassement Coût vs Écart Durée")
plt.xlabel("Dépassement Coût")
plt.ylabel("Écart Durée (jours)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







# --- Chargement des données ---
df_HSE = pd.read_excel("project risk management.xlsx", sheet_name='les Risques HSE ')

# --- Fonction pour afficher et exporter les risques HSE d'un type de projet ---
def afficher_risques_hse(type_projet_choisi):
    resultats = df_HSE[df_HSE['Type de Projet'] == type_projet_choisi][['les Risques ', 'La criticite ']]

    if resultats.empty:
        print(f"Aucun risque trouvé pour le type de projet : {type_projet_choisi}")
        return

    print(f"\n📋 Risques HSE pour le type de projet : {type_projet_choisi}")

    # Compter les criticités
    criticite_counts = resultats['La criticite '].value_counts().sort_index()
    total = criticite_counts.sum()

    # Préparer les données pour le camembert
    labels = [f'Criticité {int(c)} ({round(100 * n / total)}%)' for c, n in criticite_counts.items()]
    criticite_map = {1: 'green', 2: 'orange', 3: 'red'}

    # Lister les risques par criticité
    risques_par_criticite = {
        1: resultats[resultats['La criticite '] == 1]['les Risques '].tolist(),
        2: resultats[resultats['La criticite '] == 2]['les Risques '].tolist(),
        3: resultats[resultats['La criticite '] == 3]['les Risques '].tolist(),
    }

    # Camembert
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.pie(criticite_counts, labels=labels, colors=[criticite_map[int(i)] for i in criticite_counts.index],
           autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
    ax.set_title(f"Répartition des risques HSE par criticité - {type_projet_choisi}")
    ax.axis('equal')

    # Légende
    legend_elements = []
    for criticite in [3, 2, 1]:
        couleur = criticite_map[criticite]
        risques = risques_par_criticite.get(criticite, [])
        if risques:
            legend_elements.append(mpatches.Patch(color=couleur, label=f"Criticité {criticite} :"))
            for risque in risques:
                legend_elements.append(mpatches.Patch(color=couleur, label=f"  • {risque}"))

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    plt.tight_layout()
    plt.show()

    # --- Export tableau détaillé ---
    colonnes_utiles = ['les Risques ', 'La criticite ', 'Causes', 'Actions correctives']
    if all(col in df_HSE.columns for col in colonnes_utiles):
        tableau_details = df_HSE[df_HSE['Type de Projet'] == type_projet_choisi][colonnes_utiles]
        tableau_details.columns = ['Risque', 'Criticité', 'Causes', 'Actions']

        # Nettoyage du nom de fichier
        nom_fichier = f"risques_HSE_{type_projet_choisi.strip().replace(' ', '_')}.xlsx"
        tableau_details.to_excel(nom_fichier, index=False)
        print(f"✅ Fichier exporté : {nom_fichier}")
    else:
        print("⚠️ Une ou plusieurs colonnes (Causes, Actions correctives...) sont manquantes.")

# --- Modèle prédictif ---
X = df_HSE[['Type de Projet']]
y = df_HSE['La criticite ']
X_encoded = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"🎯 Précision du modèle : {accuracy * 100:.2f}%")

joblib.dump(model, 'model_hse.pkl')
df_HSE['Criticite_predite'] = model.predict(X_encoded)

print("\n--- Exemple de prédictions ajoutées ---")
print(df_HSE[['Type de Projet', 'les Risques ', 'Criticite_predite']].head())

# --- Lancer l'affichage/export pour chaque type de projet ---
types_projets = df_HSE['Type de Projet'].dropna().unique()
for type_projet in types_projets:
    afficher_risques_hse(type_projet)






# Fonction pour générer un nom court (acronyme) à partir d'un nom complet
def generer_nom_court(risque):
    return ''.join([mot[0].upper() for mot in risque.split() if mot.strip()])
base_model = RandomForestRegressor(random_state=42)
model_pg = MultiOutputRegressor(base_model)
# Prédiction de Probabilité, Gravité, Impact avec noms courts
def predict_risk_matrix_outputs(X_eval_risques, model_pg):
    """
    Prédit la probabilité, la gravité et l'impact de chaque risque pour chaque projet
    et génère un nom court pour chaque risque.
    """
    X_model_input = X_eval_risques.drop(columns=['Risque'], errors='ignore')

    preds = model_pg.predict(X_model_input)
    preds_df = pd.DataFrame(preds, columns=['Probabilité', 'Gravité'])

    preds_df['Probabilité'] = preds_df['Probabilité'].round().astype(int).clip(1, 3)
    preds_df['Gravité'] = preds_df['Gravité'].round().astype(int).clip(1, 3)
    preds_df['Impact'] = preds_df['Probabilité'] * preds_df['Gravité']

    noms_complets = X_eval_risques['Risque'].values
    noms_courts = [generer_nom_court(nom) for nom in noms_complets]

    preds_df['Risque_Court'] = noms_courts
    preds_df['Risque_Complet'] = noms_complets

    return preds_df

# 1. Construction des données
X_train_pg = pd.concat([
    X_test_risk.reset_index(drop=True),
    risks_pred_df.reset_index(drop=True)
], axis=1)

y_train_pg = risques_df.loc[X_test_risk.index, ['Probabilité', 'Gravité']]

# 2. Entraînement du modèle
model_pg = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model_pg.fit(X_train_pg, y_train_pg)

# 3. Sauvegarde
joblib.dump(model_pg, 'model_Probabilité_Gravité.pkl')



# Appliquer style global
matplotlib.rcParams['font.family'] = 'Calibri'
matplotlib.rcParams['font.size'] = 14

def plot_risk_matrix(projet_id, risques_df):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    import textwrap
    import streamlit as st

    # === Fonction pour forcer les retours à la ligne ===
    def wrap_label(label, width=30):
        return '\n'.join(textwrap.wrap(label, width=width))

    color_map = {
        (1, 1): "green", (1, 2): "green", (1, 3): "yellow",
        (2, 1): "green", (2, 2): "yellow", (2, 3): "orange",
        (3, 1): "yellow", (3, 2): "orange", (3, 3): "red"
    }

    colors = np.empty((3, 3), dtype=object)
    labels = np.empty((3, 3), dtype=object)

    for p in range(1, 4):
        for g in range(1, 4):
            i, j = p - 1, g - 1
            impact = p * g
            colors[i, j] = color_map[(p, g)]
            labels[i, j] = f"{impact}\n"

    for _, row in risques_df.iterrows():
        p, g = int(row['Probabilité']), int(row['Gravité'])
        name = row.get('Risque_Complet', row.get('Risque_Court', 'Inconnu'))
        i, j = p - 1, g - 1
        wrapped = wrap_label(f"• {name}", width=40)
        labels[i, j] += wrapped + "\n"

    unique_colors = ['green', 'yellow', 'orange', 'red']
    cmap = ListedColormap(unique_colors)
    color_to_index = {color: i for i, color in enumerate(unique_colors)}
    color_indices = np.vectorize(color_to_index.get)(colors)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(color_indices, cmap=cmap)

    # Texte dans chaque cellule
    for i in range(3):
        for j in range(3):
            text_color = 'white' if colors[i, j] in ['red', 'orange'] else 'black'
            cell_text = labels[i, j].strip()
            ax.text(j, i, cell_text, ha='center', va='center', color=text_color,
                    fontsize=6, fontweight='bold', linespacing=1.3)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(['Mineure (1)', 'Modérée (2)', 'Critique (3)'], fontsize=12)
    ax.set_yticklabels(['Rare (1)', 'Possible (2)', 'Probable (3)'], fontsize=12)
    ax.set_xlabel('Gravité →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probabilité ↓', fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_title(f"Matrice d'Évaluation des Risques – Projet {projet_id}", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    st.pyplot(fig)

model_pg = joblib.load("model_Probabilité_Gravité.pkl")

    # ✅ Affichage dans Streamlit



# Fonction robuste de nettoyage
def nettoyer_texte(texte):
    if pd.isna(texte):
        return ''
    texte = str(texte).strip().lower()
    texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8')
    texte = texte.replace("’", "'").replace("–", "-").replace("œ", "oe").replace("'", "'").strip()
    return texte

# Charger les données depuis Excel
df_CausesActions = pd.read_excel("project risk management.xlsx", sheet_name="Causes et Actions preventive")

# Nettoyer les risques de la base
df_CausesActions['Risque_clean'] = df_CausesActions['Risque'].apply(nettoyer_texte)

# Exemple : liste de risques prédits
risques_predits = [
    ['Mauvaise qualité des matériaux ou non-conformité'],
    ['Délais administratifs (permis, autorisations…)'],
    ['Défaillance d’équipement ou de machines'],
    ['Conditions imprévues sur le site'],
    ['Calculs erronés des quantités'],
    ['Retards dans la livraison des matériaux et des équipements'],
    ['Retards dans l’obtention des dessins ou rapports de travail'],
    ["Non-disponibilité des matériaux, équipements ou main-d'œuvre (Problèmes d’approvisionnement)"],
    ['Mauvaise coordination du site'],
    ['Échec de la construction (mauvaise exécution)'],
    ['Conditions imprévues du sol'],
    ['Retard dans le transport du béton prêt à l’emploi'],
    ['Répétitions ou reprises de travaux (rework)'],
    ['Conception inadéquate et erreurs de conception'],
    ['Modifications imprévues multiples du périmètre du projet'],
    ['fluctuation des prix'],
    ['Difficultés financières/défaillance du sous-traitant'],
]

# === Affichage formaté par projet ===
total_risques = 0
trouves = 0

for idx, liste_risques in enumerate(risques_predits):
    print(f"\n🧾 Résultats pour le projet test n°{idx + 1} :\n")
    tableau = []

    for risque in liste_risques:
        total_risques += 1
        risque_clean = nettoyer_texte(risque)
        ligne = df_CausesActions[df_CausesActions['Risque_clean'] == risque_clean]

        if not ligne.empty:
            trouves += 1
            cause = ligne['Cause'].values[0]
            action = ligne['Action Préventive'].values[0]

            causes_formatees = '\n'.join([f"• {c.strip().strip(',')}" for c in cause.split('•') if c.strip()])
            actions_formatees = '\n'.join([f"• {a.strip().strip(',')}" for a in action.split('•') if a.strip()])
        else:
            causes_formatees = "❓ Non trouvée"
            actions_formatees = "❓ Non trouvée"

        tableau.append([risque, causes_formatees, actions_formatees])

    df_result = pd.DataFrame(tableau, columns=["Risque", "Cause", "Action Préventive"])
    print(df_result.to_string(index=False))

# === Accuracy finale ===
accuracy_mapping = trouves / total_risques if total_risques > 0 else 0
print(f"\n🎯 Accuracy du mapping Risque → (Cause, Action) : {accuracy_mapping:.2%}")



def generer_registre_des_risques_depuis_excel() -> pd.DataFrame:

# === Fonction de nettoyage ===
 def nettoyer_texte(texte):
      if pd.isna(texte):
        return ''
      texte = str(texte).strip().lower()
      texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8')
      return texte.replace("’", "'").replace("–", "-").replace("œ", "oe").strip()

# === Chargement des données ===
df_causes_actions = pd.read_excel("project risk management.xlsx", sheet_name="Causes et Actions preventive")
df_risques = pd.read_excel("project risk management.xlsx", sheet_name="Risques")

# === Nettoyage des noms de risque ===
df_causes_actions['Risque_clean'] = df_causes_actions['Risque'].apply(nettoyer_texte)
df_risques['Risque_clean'] = df_risques['Risques'].apply(nettoyer_texte)

# === Jointure pour récupérer causes et actions ===
df_registre = pd.merge(
    df_risques,
    df_causes_actions[['Risque_clean', 'Cause', 'Action Préventive']],
    on='Risque_clean',
    how='left'
)

# === Calcul de l'impact ===
df_registre['Impact'] = df_registre['Probabilité'] * df_registre['Gravité']

# === Détection des risques critiques ===
def marquer_critique(row):
    if row['Impact'] == 9:
        return f"{row['Risques']} (critique)"
    elif row['Impact'] == 6 and row['Gravité'] == 3:
        return f"{row['Risques']} (critique)"
    else:
        return row['Risques']

df_registre['Risque'] = df_registre.apply(marquer_critique, axis=1)

# === Réorganisation des colonnes ===
df_registre_final = df_registre[[
    'Risque', 'Probabilité', 'Gravité', 'Impact', 'Cause', 'Action Préventive'
]]

# === Remplissage des manquants ===
df_registre_final.fillna("❓ Non trouvée", inplace=True)

# === Export Excel ===
df_registre_final.to_excel("Registre_des_Risques.xlsx", index=False)

print("✅ Registre des risques généré avec succès.")


