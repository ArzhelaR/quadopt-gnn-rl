import pandas as pd

df = pd.read_csv("full-dataset-obs36-v0_1.csv")

if 'Value' in df.columns:
    alpha =0.1
    df["Smoothed"]=df["Value"].ewm(alpha=alpha, adjust=False).mean()

    df.to_csv("full-dataset-obs36-v0_1.csv",index=False)