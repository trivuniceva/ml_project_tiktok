import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv('/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv')
print(data.head())

numerical_columns = data.select_dtypes(include=['number']).columns
X = data[numerical_columns]

num_rows_with_nan = X.isnull().sum()
print(f"nan pre zamene: {num_rows_with_nan}")

X = X.fillna(X.mean())
X = X.drop(columns=['videoUrlNoWaterMark', 'videoApiUrlNoWaterMark'])

num_rows_with_nan_after = X.isnull().sum()
print(f"nan posle zamene: {num_rows_with_nan_after}")

if X.isnull().values.any():
    raise ValueError("Nakon zamene, još uvek postoje NaN vrednosti u datasetu.")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)  # Pretpostavimo 3 klastera, možete promeniti broj klastera
data['Cluster'] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)

data['PCA1'] = pca_components[:, 0]
data['PCA2'] = pca_components[:, 1]

plt.figure(figsize=(10, 7))
scatter = plt.scatter(data['PCA1'], data['PCA2'], c=data['Cluster'], cmap='viridis', s=50, alpha=0.7)
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA plot podataka')
plt.colorbar(scatter, label='Klaster')
plt.show()