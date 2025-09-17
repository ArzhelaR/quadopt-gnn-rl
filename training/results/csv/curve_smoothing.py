import pandas as pd

df = pd.read_csv("validity_obs108.csv")

if 'Value' in df.columns:
    alpha =0.05
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("validity_obs108.csv",index=False)