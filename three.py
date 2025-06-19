import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
from two import generer_registre_des_risques_depuis_excel

from two import (
    colonnes_features,
    predict_risques,
    predict_impacts,
    simulation_monte_carlo,
    afficher_barres_comparatives,
    generer_registre_des_risques_depuis_excel, 
    predict_risk_matrix_outputs,
    afficher_risques_hse,
    generer_registre_des_risques_from_prediction
)

def plot_cdf(sim_data, titre, unite, color, p20, p50, p80):
    sorted_data = np.sort(sim_data)
    probs = np.linspace(0, 1, len(sorted_data))

    fig, ax = plt.subplots()
    ax.plot(sorted_data, probs, color=color, linewidth=2)
    ax.axhline(0.2, linestyle='--', color='orange')
    ax.axhline(0.5, linestyle='--', color='green')
    ax.axhline(0.8, linestyle='--', color='blue')
    ax.axvline(p20, linestyle='--', color='orange')
    ax.axvline(p50, linestyle='--', color='green')
    ax.axvline(p80, linestyle='--', color='blue')

    ax.text(p20, 0.05, f"P20 = {p20:.2f} {unite}", color='orange')
    ax.text(p50, 0.55, f"P50 = {p50:.2f} {unite}", color='green')
    ax.text(p80, 0.85, f"P80 = {p80:.2f} {unite}", color='blue')

    ax.set_title(titre)
    ax.set_xlabel(f"{titre} ({unite})")
    ax.set_ylabel("Probabilité cumulée")
    ax.grid(True)
    st.pyplot(fig)

def plot_barres_comparatives(prevu, p20, p50, p80, titre, unite, couleur):
    val_p20 = prevu + p20
    val_p50 = prevu + p50
    val_p80 = prevu + p80

    categories = [f"{titre} Prévu", "Scénario 20%", "Scénario 50%", "Scénario 80%"]
    valeurs = [prevu, val_p20, val_p50, val_p80]
    couleurs = ["lightgray", couleur, couleur, couleur]

    fig, ax = plt.subplots()
    bars = ax.bar(categories, valeurs, color=couleurs)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom')

    ax.set_ylabel(unite)
    ax.set_title(f"{titre} : Comparaison Prévu vs Simulation Monte Carlo")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

# --- Configuration de la page ---
st.set_page_config(
    page_title="Risk NO Risk",
    page_icon="logo.png",
    layout="wide"
)

# --- Barre latérale avec le menu personnalisé ---
st.sidebar.image("logo.png", width=150)
st.sidebar.title("Risk NO Risk")

menu = st.sidebar.radio("Navigation", [
    "Page principale",
    "Modèle de prédiction",
    "Résultats",
    "À propos"
])

# --- Contenu des différentes pages ---
if menu == "Page principale":
    st.header("⚙️ Page principale")
    st.markdown("### 🗂 Paramètres du Projet")

    nom = st.text_input("Nom du projet")
    localisation = st.selectbox("Localisation", [
        "Alger", "Abidjan", "Accra", "Adrar", "Annaba", "Blida", "Béjaïa", "Cairo",
        "Cape Town", "Chicago", "Constantine", "Dakar", "Houston", "Lagos",
        "Los Angeles", "Nairobi", "New York", "Oran", "Seattle", "Tamanrasset",
        "Tizi Ouzou", "Tlemcen", "Tunis"
    ])
    date_debut = st.date_input("Date de début")
    date_fin = st.date_input("Date de fin")
    type_projet = st.selectbox("Type de projet", ["Bâtiment", "Route", "Pont", "Tunnel", "Barrage"])
    cout_prevu = st.number_input("💰 Coût prévu du projet (en $)", min_value=1000, step=1000, value=1000000)
    duree_prevue = st.number_input("🕒 Durée prévue (en jours)", min_value=1, step=1, value=180)
    st.markdown("### 🧩 Questionnaire de Risques Complémentaires")
    q1 = st.radio("Vous avez constaté des problèmes de performance ou de fiabilité des équipements utilisés ?", ["1 (Oui)", "0 (Non)"])
    q2 = st.radio("Le contrôle qualité n’est-il pas systématiquement effectué lors de la réception des matériaux ?", ["1 (Oui)", "0 (Non)"])
    q3 = st.radio("Le chantier ne dispose-t-il pas d’un plan qualité formel et appliqué ?", ["1 (Oui)", "0 (Non)"])
    q4 = st.radio("Estimez-vous que la main-d'œuvre actuellement disponible soit insuffisamment qualifiée pour ce projet ?", ["1 (Oui)", "0 (Non)"])
    q5 = st.radio("Un plan de maintenance préventive n’est-il pas mis en place pour les équipements ?", ["1 (Oui)", "0 (Non)"])
    q6 = st.radio("Pensez-vous que le chantier pourrait connaître des arrêts de plus de 5 jours au cours d’un mois ?", ["1 (Oui)", "0 (Non)"])
    q7 = st.radio("N’existe-t-il pas un plan logistique structuré pour organiser l’approvisionnement, le stockage et les flux sur le chantier ?", ["1 (Oui)", "0 (Non)"])

    if st.button("🚀 Lancer la prédiction"):
        with st.spinner("Calcul en cours..."):
            raw_data = {
                 "Coût Prévu": float(cout_prevu),
                 "Durée Prévue": float(duree_prevue),
                 " vous avez constaté des problèmes de performance ou de fiabilité des équipements utilisés ?": 1 if q1.startswith("1") else 0,
                 "Le contrôle qualité n’est-il pas systématiquement effectué lors de la réception des matériaux ?": 1 if q2.startswith("1") else 0,
                 "Le chantier ne dispose-t-il pas d’un plan qualité formel et appliqué ?": 1 if q3.startswith("1") else 0,
                 "Estimez-vous que la main-d'œuvre actuellement disponible est insuffisamment qualifiée pour ce projet ?": 1 if q4.startswith("1") else 0,
                 "Un plan de maintenance préventive  n’est-il pas mis en place pour les équipements ?": 1 if q5.startswith("1") else 0,
                 "Pensez-vous que le chantier pourrait connaître des arrêts de plus de 5 jours au cours d’un mois ?": 1 if q6.startswith("1") else 0,
                 "N’existe-t-il pas un plan logistique structuré   pour organiser l’approvisionnement, le stockage et les flux sur le chantier ? ": 1 if q7.startswith("1") else 0,
                 f"Type de Projet_{type_projet}": 1,
                 f"Localisation_{localisation}": 1,
                 "Debut_annee": date_debut.year,
                 "Debut_mois": date_debut.month,
                 "Debut_jours": date_debut.day,
                 "Fin_annee": date_fin.year,
                 "Fin_mois": date_fin.month,
                 "Fin_jours": date_fin.day,
}


            for col in colonnes_features:
                if col not in raw_data:
                    raw_data[col] = 0

            y_pred, X_input = predict_risques(raw_data)
            impacts = predict_impacts(X_input, y_pred)

            sim_cout = simulation_monte_carlo(
                pd.Series([impacts["Dépassement Coût"]]), "Dépassement de Coût", "$", "green")
            p20_cout, p50_cout, p80_cout = np.percentile(sim_cout, [20, 50, 80])
            cout_total_p80 = raw_data["Coût Prévu"] + p80_cout

            sim_duree = simulation_monte_carlo(
                pd.Series([impacts["Écart Durée"]]), "Écart de Durée", "jours", "blue")
            p20_duree, p50_duree, p80_duree = np.percentile(sim_duree, [20, 50, 80])
            duree_total_p80 = raw_data["Durée Prévue"] + p80_duree
           

# Exemple dans la section "Résultats"


            st.subheader("📈 Résultats simulés - Valeurs P80")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💰 Dépassement Coût (P80)", f"{p80_cout:,.2f} $")
                st.metric("💰 Coût Réel estimé (P80)", f"{cout_total_p80:,.2f} $")
            with col2:
                st.metric("🕒 Écart Durée (P80)", f"{p80_duree:.2f} jours")
                st.metric("🕒 Durée Réelle estimée (P80)", f"{duree_total_p80:.2f} jours")

            st.subheader("🎲 Courbes de probabilité cumulative")
            plot_cdf(sim_cout, "Dépassement de Coût", "$", "green", p20_cout, p50_cout, p80_cout)
            plot_cdf(sim_duree, "Écart de Durée", "jours", "blue", p20_duree, p50_duree, p80_duree)

            st.subheader("📊 Comparaison des scénarios simulés")
            plot_barres_comparatives(raw_data["Coût Prévu"], p20_cout, p50_cout, p80_cout, "Coût", "$", "darkgreen")
            plot_barres_comparatives(raw_data["Durée Prévue"], p20_duree, p50_duree, p80_duree, "Durée", "jours", "blue")

            # Matrice (ex : afficher le premier projet utilisateur)
            st.subheader("🧱 Matrice des risques")
            predict_risk_matrix_outputs(0, X_input, y_pred)

            st.subheader("🧯 Risques HSE associés")
            afficher_risques_hse(st, type_projet)

           
            st.subheader("📋 Registre des risques prédits")
            from two import generer_registre_des_risques_from_prediction
            registre = generer_registre_des_risques_from_prediction(X_input, y_pred)
            st.dataframe(registre)

            if st.button("📥 Exporter le registre en Excel"):
              
              output = BytesIO()

              with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                
                registre.to_excel(writer, index=False, sheet_name='Registre')
              output.seek(0)

              st.download_button(
                label="📦 Télécharger le registre",
                data=output,
                file_name="Registre_des_Risques.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


elif menu == "Modèle de prédiction":
    st.header("📊 Modèle de prédiction")
    st.image("processus_prediction.png", caption="Figure : Processus global de prédiction des risques", use_container_width=True)
    st.markdown("""
   Le cœur de l’application Risk NO R repose sur un système intelligent de prédiction des risques dans les projets de construction. Ce système a été conçu pour anticiper les problèmes les plus fréquents rencontrés sur les chantiers, tels que les retards, les dépassements de coûts, les accidents HSE, ou encore les problèmes de qualité.

Il fonctionne à partir d’un travail de fond basé sur l’analyse de plus de 2000 projets de contruction, dans lesquels ont été recensés les types de risques, leurs causes, leurs impacts et les actions mises en place. À partir de cette base, un modèle d’intelligence artificielle a été entraîné pour reconnaître les schémas de risque et générer des prédictions fiables pour de nouveaux projets.

🔄 Étapes de fonctionnement du modèle
1. Collecte et structuration des données
Le développement du système commence par la collecte de données issues de projets passés. Chaque projet est décrit par :

ses caractéristiques générales (type, durée, budget, localisation…),

les risques rencontrés (qualité, sécurité, délais, etc.),

les dépassements enregistrés,

les causes identifiées,

et les actions de prévention appliquées.

Ces données ont été organisées dans un fichier Excel structuré en plusieurs feuilles : projets, risques, causes, actions, risques HSE. Chaque élément est codé de manière binaire ou chiffrée pour être exploitable par des algorithmes.

2. Traitement et préparation
Les données brutes sont ensuite nettoyées, standardisées et codées. On y calcule pour chaque risque :

une probabilité (fréquence d’apparition),

une gravité (niveau d’impact),

et un score d’impact (probabilité × gravité).

Ces scores permettent de hiérarchiser les risques dans une matrice de criticité.

3. Entraînement des modèles de machine learning
Les modèles sont entraînés à partir des données traitées :

Régression Logistique : pour des prédictions simples (présence ou non d’un risque).

Random Forest Classifier : puissant pour traiter des ensembles de données complexes.

MultiOutputClassifier : capable de prédire plusieurs risques simultanément (par exemple un risque de retard + un risque HSE + un surcoût).

Chaque modèle apprend à détecter des configurations de projets à risque à partir des expériences passées.

4. Prédiction des risques
Quand l’utilisateur saisit un nouveau projet dans l’application :

Le système analyse les paramètres saisis (type, durée, coût, localisation, réponses aux questions…).

Il compare ces caractéristiques à celles de projets déjà analysés.

Il génère une liste des risques probables, en indiquant pour chacun :

sa probabilité,

sa gravité,

son score d’impact.

En complément, les causes les plus probables de chaque risque sont extraites, ainsi que les actions de prévention recommandées.

5. Prédiction des dépassements (coût & durée)
Le système va plus loin en évaluant le risque de dépassement de budget ou de retard de livraison. Il combine deux approches :

Un modèle supervisé de machine learning,

Une simulation Monte Carlo qui génère 1000 scénarios futurs par projet.

Pour chaque projet, trois estimations sont données :

Optimiste (20 %),

Moyenne (50 %),

Pessimiste (80 %).

Ces prédictions permettent de mieux anticiper les scénarios les plus défavorables.

6. Évaluation des risques HSE
Les risques liés à la sécurité et à l’environnement (HSE) sont évalués spécifiquement selon le type de projet (bâtiment, route, tunnel, etc.). Chaque type de projet est associé à une liste de risques HSE potentiels, avec un niveau de criticité adapté.

🧰 Technologies utilisées
Excel : pour la préparation et l’organisation des données.

Python : pour tout le traitement, la modélisation, et la prédiction.

Pandas, Scikit-learn, Seaborn, Matplotlib : pour manipuler, visualiser et modéliser les données.

Streamlit : pour créer une interface simple et interactive.

Google Colab : pour entraîner les modèles dans un environnement cloud.

Visual Studio Code : pour le développement final de l’application web.

📋 Résultats générés par le système
Lorsque l’utilisateur lance la prédiction, plusieurs résultats sont affichés automatiquement :

Un registre des risques contenant pour chaque risque :

description,

probabilité,

gravité,

score d’impact,

causes,

actions préventives.

Une matrice des risques qui permet de visualiser les risques critiques (impact élevé).

Un graphe des risques HSE (en camembert).

Des diagrammes de comparaison entre le coût prévu/réel et la durée prévue/réelle.

Des simulations Monte Carlo pour visualiser les incertitudes.

🎯 Objectif final
L’objectif du modèle est de transformer des données passées en informations utiles pour l’avenir. Il ne se contente pas de décrire le passé, il prédit, recommande, alerte. Il aide les chefs de projet à prendre des décisions préventives, avant même que les problèmes n’apparaissent.

C’est un véritable outil d’aide à la décision pour mieux planifier, mieux anticiper, et réduire les impacts négatifs des risques dans les projets de construction.


    """)



# ---Page Résultats ---
elif menu == "Résultats":
    st.header("📈 Résultats de la prédiction")

    st.markdown("""
Lorsque vous lancez la prédiction, Risk NO Risk génère automatiquement plusieurs visualisations pour vous aider à interpréter les résultats.
""")

    # 1. Registre des risques
    st.subheader("📋 1. Registre des risques")
    st.image("registre_risques.png", caption="Figure 1 : Exemple de registre des risques généré", use_container_width=True)

    # 2. Matrice des risques
    st.subheader("🟨 2. Matrice des risques")
    st.image("matrice.png", caption="Figure 2 : Matrice de criticité des risques", use_container_width=True)

    # 3. Graphe HSE
    st.subheader("🧯 3. Graphe des risques HSE")
    st.image("les risques hse.png", caption="Figure 3 : Répartition des risques HSE", use_container_width=True)

    # 4. Comparaison coûts
    st.subheader("📊 4. Comparaison des coûts prévus et réels")
    st.image("Comparaison des coûts prévu et réel.png", caption="Figure 4 : Coût prévu vs Coût réel estimé", use_container_width=True)

    # 4. Comparaison durées
    st.subheader("📊 4. Comparaison des durées prévues et réelles")
    st.image("Comparaison des durée prévue et réelle.png", caption="Figure 5 : Durée prévue vs Durée réelle estimée", use_container_width=True)

    # 5. Simulation Monte Carlo (Coût)
    st.subheader("🎲 5. Simulation Monte Carlo - Dépassement de coût")
    st.image("depassement de cout simulés.png", caption="Figure 6 : Simulation Monte Carlo - Coûts", use_container_width=True)

    # 5. Simulation Monte Carlo (Durée)
    st.subheader("🎲 6. Simulation Monte Carlo - Écart de durée")
    st.image("Trois ecarts duree potentielles.png", caption="Figure 7 : Simulation Monte Carlo - Durées", use_container_width=True)

    # Conclusion
    st.markdown("""
---
🎯 **En résumé**, la page « Résultats » de Risk NO Risk vous fournit une vue complète et graphique des risques, permettant une prise de décision proactive, visuelle et hiérarchisée.
""")



elif menu == "À propos":
    st.header("ℹ️ À propos")
    st.markdown("""
   Ce projet de fin d’études a été réalisé par HALHALI MERIEM, étudiante en cinquième année de Génie Industriel, spécialité Management Industriel et Logistique. Il vise à développer un modèle prédictif pour anticiper les risques dans les projets de construction, en s’appuyant sur une base de données de plus de 2000 projets historiques.

Ce travail combine des techniques statistiques, des algorithmes d’apprentissage automatique et des simulations avancées afin de fournir des outils permettant d’optimiser la gestion des risques et la prise de décision dans le secteur de la construction.

Ce mémoire s’inscrit dans le cadre du management industriel et logistique, avec un focus sur l’intégration des technologies numériques pour améliorer la maîtrise des risques dans les projets.
    """)

# --- Footer toujours visible en bas ---
st.markdown("---")
st.markdown("© 2025 Risk NO Risk | Développé par [HALHALI meriem]")