import pandas as pd
from functools import reduce

obs4 = pd.read_csv("results_obs4-5500000.csv")
obs12 = pd.read_csv("results_obs12-5500000.csv")
obs36 = pd.read_csv("results_obs36-5500000.csv")
obs108 = pd.read_csv("results_obs108-5500000.csv")

obs4 = obs4.transpose()
obs12 = obs12.transpose()
obs36 = obs36.transpose()
obs108 = obs108.transpose()

obs4.columns = obs4.iloc[0]
obs12.columns = obs12.iloc[0]
obs36.columns = obs36.iloc[0]
obs108.columns = obs108.iloc[0]

obs4 = obs4[1:]
obs12 = obs12[1:]
obs36 = obs36[1:]
obs108 = obs108[1:]


obs4 = obs4.rename(columns={
    "avg_normalized_return": "avg_normalized_return_obs4",
    "std_normalized_return": "std_normalized_return_obs4",
    "avg_length": "avg_length_obs4",
    "nb_wins": "wins_obs4"
})

obs12 = obs12.rename(columns={
    "avg_normalized_return": "avg_normalized_return_obs12",
    "std_normalized_return": "std_normalized_return_obs12",
    "avg_length": "avg_length_obs12",
    "nb_wins": "wins_obs12"
})

obs36 = obs36.rename(columns={
    "avg_normalized_return": "avg_normalized_return_obs36",
    "std_normalized_return": "std_normalized_return_obs36",
    "avg_length": "avg_length_obs36",
    "nb_wins": "wins_obs36"
})

obs108 = obs108.rename(columns={
    "avg_normalized_return": "avg_normalized_return_obs108",
    "std_normalized_return": "std_normalized_return_obs108",
    "avg_length": "avg_length_obs108",
    "nb_wins": "wins_obs108"
})

obs4 = obs4.drop(columns=["avg_mesh_rewards"])
obs12 = obs12.drop(columns=["avg_mesh_rewards"])
obs36 = obs36.drop(columns=["avg_mesh_rewards"])
obs108 = obs108.drop(columns=["avg_mesh_rewards"])

print(obs4.columns)
# Supposons que 'mesh_id' est la colonne clé
df_results = obs4.merge(obs12, on="mesh_id").merge(obs36, on="mesh_id").merge(obs108, on="mesh_id")

# Arrondir les valeurs à 2 décimales
df_results = df_results.round(2)

# On ajoute une étiquette pour la dernière ligne
means = df_results.drop(columns=["mesh_id"]).mean()

# Transformer en DataFrame avec une seule ligne
means = means.to_frame().T

# Ajouter la colonne 'mesh_id' avec le label "Moyenne"
means.insert(0, "mesh_id", "Mean")
print(means)

# Ajout au DataFrame
df_results = pd.concat([df_results, means], ignore_index=True)

new_order = ["mesh_id", "avg_normalized_return_obs4", "avg_normalized_return_obs12", "avg_normalized_return_obs36", "avg_normalized_return_obs108", "std_normalized_return_obs4", "std_normalized_return_obs12", "std_normalized_return_obs36", "std_normalized_return_obs108", "avg_length_obs4", "avg_length_obs12", "avg_length_obs36", "avg_length_obs108", "wins_obs4", "wins_obs12", "wins_obs36", "wins_obs108"]

df_results= df_results[new_order]
df_results.drop(df_results.columns[0], axis=1, inplace=True)

print(df_results)

latex_code = df_results.to_latex(index=True, header=True, float_format="%.2f")

print(latex_code)
