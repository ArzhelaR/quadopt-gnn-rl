import pandas as pd

df = pd.read_csv("full_dataset_obs36_perso-v0.csv")

if 'Value' in df.columns:
    alpha =0.1
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("full_dataset_obs36_perso-v0.csv",index=False)