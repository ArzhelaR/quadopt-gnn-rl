import pandas as pd
from functools import reduce

TMesh = pd.read_csv("results_IMR/results_perso.csv")
sb3 = pd.read_csv("results_IMR/results_5darts.csv")

TMesh = TMesh.transpose()
sb3 = sb3.transpose()

TMesh.columns = TMesh.iloc[0]
sb3.columns = sb3.iloc[0]

TMesh = TMesh[1:]
sb3 = sb3[1:]


TMesh = TMesh.rename(columns={
    "avg_normalized_return": "avg_normalized_return_TMesh",
    "std_normalized_return": "std_normalized_return_TMesh",
    "avg_length": "avg_length_TMesh",
    "nb_wins": "wins_TMesh"
})

sb3 = sb3.rename(columns={
    "avg_normalized_return": "avg_normalized_return_sb3",
    "std_normalized_return": "std_normalized_return_sb3",
    "avg_length": "avg_length_sb3",
    "nb_wins": "wins_sb3"
})

TMesh = TMesh.drop(columns=["avg_mesh_rewards"])
sb3 = sb3.drop(columns=["avg_mesh_rewards"])

print(TMesh.columns)
# Supposons que 'mesh_id' est la colonne clé
df_results = TMesh.merge(sb3, on="mesh_id")

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

new_order = ["mesh_id", "avg_normalized_return_TMesh", "avg_normalized_return_sb3", "std_normalized_return_TMesh", "std_normalized_return_sb3", "avg_length_TMesh", "avg_length_sb3", "wins_TMesh", "wins_sb3"]

df_results= df_results[new_order]
df_results.drop(df_results.columns[0], axis=1, inplace=True)

print(df_results)

latex_code = df_results.to_latex(index=True, header=True, float_format="%.2f")

print(latex_code)
