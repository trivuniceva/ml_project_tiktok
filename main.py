from data_preprocessing import *

data_path = '/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv'

data = load_data(data_path)

X_scaled, X = preprocess_data(data)

vizualize_correlation(X)

apply_kmeans(X_scaled, data)
apply_pca(X_scaled, X)
