import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv("validity_perso.csv")
df2 = pd.read_csv("validity_obs36.csv")

# Fenêtre de lissage
W = 5

plt.figure(figsize=(10,6))

# --- PPO TMesh ---
# min et max sur fenêtre
lower1 = df1["Value"].rolling(W, min_periods=1).min()
upper1 = df1["Value"].rolling(W, min_periods=1).max()

# puis on lisse encore ces bornes
lower1_smooth = lower1.rolling(W, min_periods=1).mean()
upper1_smooth = upper1.rolling(W, min_periods=1).mean()

plt.fill_between(df1["Step"], lower1_smooth, upper1_smooth,
                 color="blue", alpha=0.2, label="Raw PPO TMesh")
plt.plot(df1["Step"], df1["Smoothed"], color="blue", linewidth=2, label="PPO TMesh")

# --- PPO SB3 ---
lower2 = df2["Value"].rolling(W, min_periods=1).min()
upper2 = df2["Value"].rolling(W, min_periods=1).max()

lower2_smooth = lower2.rolling(W, min_periods=1).mean()
upper2_smooth = upper2.rolling(W, min_periods=1).mean()

plt.fill_between(df2["Step"], lower2_smooth, upper2_smooth,
                 color="orange", alpha=0.2, label="Raw PPO SB3")
plt.plot(df2["Step"], df2["Smoothed"], color="orange", linewidth=2, label="PPO SB3")

plt.xlabel("Episodes")
plt.ylabel("Action validity (%)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

