import pandas as pd

df = pd.read_csv("normalized_return_perso_2.csv")

if 'Value' in df.columns:
    alpha =0.01
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("normalized_return_perso_2.csv",index=False)
