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
    
    df_missing = df_copy[df_copy['consumption'].isnull()]
    df_not_missing = df_copy.dropna(subset=['consumption'])

    if df_not_missing.empty:
        return df_copy

    X = df_not_missing[['temperature']]
    y = df_not_missing['consumption']

    model = LinearRegression()
    model.fit(X, y)

    missing_temperatures = df_missing[['temperature']]
    predicted_values = model.predict(missing_temperatures)
    df_copy.loc[df_copy['consumption'].isnull(), 'consumption'] = predicted_values

    return df_copy

# Fonction pour faire des prédictions
def make_predictions(df):
    """Fait des prédictions de consommation sur la base des caractéristiques disponibles."""
    features = pd.get_dummies(df[['temperature', 'day_type', 'time_of_day']], drop_first=True)
    target = df['consumption']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mse = mean_squared_error(y_test, predictions)
    
    return predictions, mse

# Lancer l'application Streamlit
st.title("Analyse de la Consommation Électrique")

# Génération du dataset
df_complete = generate_data()

# Appliquer des valeurs manquantes aléatoires
n_missing = int(df_complete.shape[0] * 0.1)  # 10 % du dataset
missing_indices = np.random.choice(df_complete.index, n_missing, replace=False)
df_complete.loc[missing_indices, 'consumption'] = np.nan

# Options pour l'utilisateur
st.subheader("Sélectionnez une méthode pour remplir les valeurs manquantes :")
method = st.selectbox("Méthode", ["Aucune", "Moyenne", "KNN", "Régression"])

# Appliquer la méthode choisie et afficher le message "Please wait"
if st.button("Appliquer la méthode"):
    with st.spinner("Please wait..."):
        if method == "Moyenne":
            df_filled = fill_missing_values_mean(df_complete)
        elif method == "KNN":
            df_filled = fill_missing_values_knn(df_complete)
        elif method == "Régression":
            df_filled = fill_missing_values_regression(df_complete)
        else:
            df_filled = df_complete.copy()

    # Faire des prédictions
    predictions, mse = make_predictions(df_filled)

    # Afficher les résultats
    st.subheader("Données originales (avec valeurs manquantes) :")
    st.write(df_complete)

    st.subheader("Données après remplissage :")
    st.write(df_filled)

    # Visualisation des résultats
    st.subheader("Visualisation de la consommation après remplissage :")
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_filled, x='timestamp', y='consumption', hue='user_id', palette='tab10', legend=None)
    plt.title("Consommation après remplissage des valeurs manquantes")
    plt.xlabel("Date/Heure")
    plt.ylabel("Consommation")
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Affichage des prédictions
    st.subheader("Prédictions de consommation après remplissage :")
    st.line_chart(pd.Series(predictions, name='Predictions'), use_container_width=True)
    st.write(f"Erreur quadratique moyenne des prédictions : {mse:.2f}")
