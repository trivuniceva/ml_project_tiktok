import pandas as pd

# Učitavanje CSV fajla
file_path = '/Users/nikolina/Desktop/Projekti/ml_tiktok_data/metadata.csv'
df = pd.read_csv(file_path)

# Ako je broj redova u fajlu manji ili jednak 100, nema potrebe za smanjivanjem
if len(df) <= 100:
    print(f"Broj redova u fajlu {file_path} je već <= 100. Nema potrebe za smanjivanjem.")
else:
    # Uzimanje prvih 100 redova
    df_reduced = df.head(100)

    # Čuvanje smanjenog data frame-a nazad u CSV fajl
    reduced_file_path = 'small_data.csv'
    df_reduced.to_csv(reduced_file_path, index=False)

    print(f"Broj redova u fajlu {file_path} je smanjen na 100 i sačuvan je u {reduced_file_path}.")
