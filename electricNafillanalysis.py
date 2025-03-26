import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import KNNImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Fonction pour générer le dataset
def generate_data(n_users=50):
    """Génère un dataset de consommation d'électricité."""
    date_range = pd.date_range(start='2023-01-09', end='2023-01-30', freq='30T')

    data = {
        'user_id': np.repeat(np.arange(n_users), len(date_range)),
        'timestamp': np.tile(date_range, n_users),
        'temperature': np.random.normal(loc=20, scale=5, size=n_users * len(date_range)),
        'day_type': np.tile(np.where(np.array(date_range.weekday) < 5, 'Weekday', 'Weekend'), n_users),
        'time_of_day': np.tile(np.where(date_range.hour < 12, 'Morning',
                                         np.where(date_range.hour < 18, 'Afternoon', 'Evening')), n_users),
        'normal_consumption': np.random.poisson(lam=10, size=n_users * len(date_range))
    }
    df = pd.DataFrame(data)
    df['consumption'] = (
        df['normal_consumption'] + (0.5 * df['temperature']) +
        np.where(df['day_type'] == 'Weekend', 10, 0) +
        np.where(df['time_of_day'] == 'Evening', 5, 0) +
        np.random.normal(scale=2, size=df.shape[0])  # Bruit aléatoire
    )
    return df

# Méthodes de remplissage
def fill_missing_values_mean(df):
    """Remplie les valeurs manquantes par la moyenne."""
    return df.fillna(df['consumption'].mean())

def fill_missing_values_knn(df):
    """Remplie les valeurs manquantes en utilisant KNN."""
    imputer = KNNImputer(n_neighbors=5)
    df[['consumption', 'temperature']] = imputer.fit_transform(df[['consumption', 'temperature']])
    return df

def fill_missing_values_regression(df):
    """Remplie les valeurs manquantes en utilisant une régression."""
    df_copy = df.copy()
    
    # Séparer les lignes avec et sans valeurs manquantes
    df_missing = df_copy[df_copy['consumption'].isnull()]
    df_not_missing = df_copy.dropna(subset=['consumption'])

    # Vérifier si df_not_missing ne contient pas d'entrées
    if df_not_missing.empty:
        return df_copy  # Pas de fill possible, retourner le dataframe original

    # Préparer les données pour la régression
    X = df_not_missing[['temperature']]
    y = df_not_missing['consumption']

    # Vérification de la présence de NaN dans y
    if y.isnull().any():
        return df_copy  # Pas de fill possible, retourner le dataframe original

    model = LinearRegression()
    model.fit(X, y)

    # Prédire les valeurs manquantes
    missing_temperatures = df_missing[['temperature']]
    predicted_values = model.predict(missing_temperatures)

    df_copy.loc[df_copy['consumption'].isnull(), 'consumption'] = predicted_values

    return df_copy

def make_predictions(df, user_id):
    """Fait des prédictions de consommation sur la base des caractéristiques disponibles pour un utilisateur spécifique."""
    user_data = df[df['user_id'] == user_id]

    if user_data.empty:
        return None, None  # Pas de données pour cet utilisateur

    # On garde uniquement les lignes où la consommation n'est pas NaN
    user_data = user_data.dropna(subset=['consumption'])

    if user_data.empty:
        return None, None  # Pas de données valides pour faire des prédictions

    features = pd.get_dummies(user_data[['temperature', 'day_type', 'time_of_day']], drop_first=True)
    target = user_data['consumption']

    # Séparation de l'ensemble de données
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # On prédit en utilisant X_test
    predictions = model.predict(X_test)

    # On récupère les timestamps pour les prédictions
    predicted_data = user_data.loc[X_test.index].copy()
    predicted_data['predictions'] = predictions

    mse = mean_squared_error(y_test, predictions)

    return predicted_data, mse

# Lancer l'application Streamlit
st.title("Analyse de la Consommation Électrique")

# Génération du dataset
if 'df_complete' not in st.session_state:
    df_complete = generate_data()
    n_missing = int(df_complete.shape[0] * 0.1)  # 10 % du dataset
    missing_indices = np.random.choice(df_complete.index, n_missing, replace=False)
    df_complete.loc[missing_indices, 'consumption'] = np.nan
    st.session_state.df_complete = df_complete

# Utilisation de l'état de session pour stocker les données remplies
if 'df_filled' not in st.session_state:
    st.session_state.df_filled = st.session_state.df_complete

# Options pour l'utilisateur
st.subheader("Sélectionnez une méthode pour remplir les valeurs manquantes :")
method = st.selectbox("Méthode", ["Aucune", "Moyenne", "KNN", "Régression"])

# Appliquer la méthode choisie
if st.button("Appliquer la méthode"):
    with st.spinner("Please wait..."):
        if method == "Moyenne":
            st.session_state.df_filled = fill_missing_values_mean(st.session_state.df_complete)
        elif method == "KNN":
            st.session_state.df_filled = fill_missing_values_knn(st.session_state.df_complete)
        elif method == "Régression":
            st.session_state.df_filled = fill_missing_values_regression(st.session_state.df_complete)

# Option pour afficher le DataFrame avant et après le remplissage
if st.checkbox("Voir le DataFrame original (avant le remplissage)"):
    st.write(st.session_state.df_complete)

if st.checkbox("Voir le DataFrame rempli (après le remplissage)"):
    st.write(st.session_state.df_filled)

# Filtrage des données pour l'utilisateur sélectionné
user_id = st.selectbox("Sélectionnez un utilisateur pour visualiser la consommation :", st.session_state.df_filled['user_id'].unique())

# Filtrer les données pour l'utilisateur sélectionné
user_data = st.session_state.df_filled[st.session_state.df_filled['user_id'] == user_id]

# Visualisation des résultats pour un utilisateur sélectionné
st.subheader(f"Visualisation de la consommation pour l'utilisateur {user_id} :")
plt.figure(figsize=(10, 5))
sns.lineplot(data=user_data, x='timestamp', y='consumption', color='blue', label='Consommation Réelle')
plt.title(f"Consommation d'électricité pour l'utilisateur {user_id}")
plt.xlabel("Date/Heure")
plt.ylabel("Consommation")
plt.xticks(rotation=45)

# Faire des prédictions pour l'utilisateur sélectionné
predicted_data, mse = make_predictions(st.session_state.df_filled, user_id)

# Affichage des prédictions
if predicted_data is not None:
    # Ajouter les prédictions au graphique
    sns.lineplot(data=predicted_data, x='timestamp', y='predictions', color='orange', label='Prédictions')
    st.line_chart(predicted_data[['timestamp', 'predictions']].set_index('timestamp'), use_container_width=True)

    plt.legend()
    st.pyplot(plt)
    st.write(f"Erreur quadratique moyenne des prédictions : {mse:.2f}")
else:
    st.write("Aucune donnée disponible pour cet utilisateur pour faire des prédictions.")
