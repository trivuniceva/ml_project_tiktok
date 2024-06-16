import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

data_path = '/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv'
# data_path = 'small_data.csv'

df = pd.read_csv(data_path)

columns_to_drop = ['secretID', 'covers.default', 'covers.origin', 'covers.dynamic', 
                   'webVideoUrl', 'videoUrl', 'videoUrlNoWaterMark', 'videoApiUrlNoWaterMark']
df = df.drop(columns=columns_to_drop)

df['createTime'] = pd.to_datetime(df['createTime'], unit='s')

df['days_since_creation'] = (pd.Timestamp.now() - df['createTime']).dt.days
df['engagement_rate'] = df['authorMeta.heart'] / df['authorMeta.video']

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(df['text'].fillna('').values.astype('U'))
df_text_features = pd.DataFrame(text_features.toarray(), columns=vectorizer.get_feature_names_out())

df_cleaned = pd.concat([df.drop(columns=['text']), df_text_features], axis=1)

scaler = MinMaxScaler()
columns_to_scale = ['days_since_creation', 'engagement_rate']
df_cleaned[columns_to_scale] = scaler.fit_transform(df_cleaned[columns_to_scale])

numeric_columns = df_cleaned.select_dtypes(include=['float64', 'int64']).columns
X = df_cleaned[numeric_columns].drop(columns=['authorMeta.heart'])  # X su sve karakteristike osim ciljne promenljive
y = df_cleaned['authorMeta.heart']  # Y je ciljna promenljiva

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
