import pandas as pd 

data_path = '/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv'
df = pd.read_csv(data_path)

for column_name in df.columns: 
    print(column_name)
