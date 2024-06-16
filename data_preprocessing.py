import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(data_path):
    data = pd.read_csv(data_path)
    print(data.head())
    return data


def preprocess_data(data):
    numerical_columns = data.select_dtypes(include=['number']).columns
    X = data[numerical_columns]

    num_rows_with_nan = X.isnull().sum()
    print(f"nan pre zamene: {num_rows_with_nan}")

    X = X.fillna(X.mean())
    X = X.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark'])
    X = X.drop(columns=['id', 'secretID', 'authorMeta.id',])
    X = X.drop(columns=['authorMeta.fans', 'videoMeta.width', 'videoMeta.height'])

    num_rows_with_nan_after = X.isnull().sum()
    print(f"nan posle zamene: {num_rows_with_nan_after}")

    if X.isnull().values.any():
        raise ValueError("Nakon zamene, još uvek postoje NaN vrednosti u datasetu.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, X

def vizualize_correlation(X):
    correlation_matrix = X.iloc[:, :40].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrica korelacije za prvih 40 kolona')
    plt.show()


def apply_kmeans(X_scaled, data):
    kmeans = KMeans(n_clusters=12, random_state=42)  # Pretpostavimo 12 klastera, možete promeniti broj klastera
    data['Cluster'] = kmeans.fit_predict(X_scaled)


def apply_pca(X_scaled, X):
    pca = PCA(n_components=4)
    pca_components = pca.fit_transform(X_scaled)

    pca_loadings = pca.components_

    feature_importance = np.abs(pca_loadings)

    total_importance = np.sum(feature_importance, axis=0)

    sorted_indices = np.argsort(total_importance)[::-1]

    print("Bitne kolone prema PCA doprinosu:")
    for i, idx in enumerate(sorted_indices):
        print(f"{i+1}. {X.columns[idx]}: {total_importance[idx]}")

    plt.figure(figsize=(12, 6))
    plt.bar(range(X.shape[1]), total_importance[sorted_indices], align='center')
    plt.xticks(range(X.shape[1]), X.columns[sorted_indices], rotation=90)
    plt.xlabel('Kolone')
    plt.ylabel('PCA Doprinos')
    plt.title('Bitne kolone prema PCA doprinosu')
    plt.tight_layout()
    plt.show()
