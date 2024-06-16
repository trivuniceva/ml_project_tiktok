import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_path = '/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv'
df = pd.read_csv(data_path)

columns_to_drop = ['secretID', 'covers.default', 'covers.origin', 'covers.dynamic', 
                   'webVideoUrl', 'videoUrl', 'videoUrlNoWaterMark', 'videoApiUrlNoWaterMark']
df_cleaned = df.drop(columns=columns_to_drop)
df_cleaned['createTime'] = pd.to_datetime(df_cleaned['createTime'], unit='s')

df_cleaned['days_since_creation'] = (pd.Timestamp.now() - df_cleaned['createTime']).dt.days
df_cleaned['engagement_rate'] = df_cleaned['authorMeta.heart'] / df_cleaned['authorMeta.video']

scaler = StandardScaler()
columns_to_scale = ['days_since_creation', 'engagement_rate']
df_cleaned[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df_cleaned['text'].values.astype('U'))
df_text_features = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())
df_cleaned_with_text_features = pd.concat([df_cleaned, df_text_features], axis=1)

df_cleaned_numeric = df_cleaned_with_text_features.select_dtypes(include=[float, int])

correlation_matrix = df_cleaned_numeric.corr()
print(correlation_matrix)

print(df_cleaned[['authorMeta.fans', 'authorMeta.heart', 'authorMeta.video']].describe())

target_variable = 'authorMeta.heart'

print("Treniranje modela...")

X = df_cleaned_numeric.drop(columns=[target_variable])
y = df_cleaned_numeric[target_variable]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

model_rf.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)

print(f'R^2: {r2_rf:.2f}')
print(f'MAE: {mae_rf:.2f}')
print(f'MSE: {mse_rf:.2f}')
