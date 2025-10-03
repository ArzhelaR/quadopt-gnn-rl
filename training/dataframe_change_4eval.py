import pandas as pd
from functools import reduce

darts3 = pd.read_csv("results_IMR/results_3darts.csv")
darts5 = pd.read_csv("results_IMR/results_5darts.csv")
darts10 = pd.read_csv("results_IMR/results_10darts.csv")
darts15 = pd.read_csv("results_IMR/results_15darts.csv")

darts3 = darts3.transpose()
darts5 = darts5.transpose()
darts10 = darts10.transpose()
darts15 = darts15.transpose()

darts3.columns = darts3.iloc[0]
darts5.columns = darts5.iloc[0]
darts10.columns = darts10.iloc[0]
darts15.columns = darts15.iloc[0]

darts3 = darts3[1:]
darts5 = darts5[1:]
darts10 = darts10[1:]
darts15 = darts15[1:]


darts3 = darts3.rename(columns={
    "avg_normalized_return": "avg_normalized_return_darts3",
    "std_normalized_return": "std_normalized_return_darts3",
    "avg_length": "avg_length_darts3",
    "nb_wins": "wins_darts3"
})

darts5 = darts5.rename(columns={
    "avg_normalized_return": "avg_normalized_return_darts5",
    "std_normalized_return": "std_normalized_return_darts5",
    "avg_length": "avg_length_darts5",
    "nb_wins": "wins_darts5"
})

darts10 = darts10.rename(columns={
    "avg_normalized_return": "avg_normalized_return_darts10",
    "std_normalized_return": "std_normalized_return_darts10",
    "avg_length": "avg_length_darts10",
    "nb_wins": "wins_darts10"
})

darts15 = darts15.rename(columns={
    "avg_normalized_return": "avg_normalized_return_darts15",
    "std_normalized_return": "std_normalized_return_darts15",
    "avg_length": "avg_length_darts15",
    "nb_wins": "wins_darts15"
})

darts3 = darts3.drop(columns=["avg_mesh_rewards"])
darts5 = darts5.drop(columns=["avg_mesh_rewards"])
darts10 = darts10.drop(columns=["avg_mesh_rewards"])
darts15 = darts15.drop(columns=["avg_mesh_rewards"])

print(darts3.columns)
# Supposons que 'mesh_id' est la colonne clé
df_results = darts3.merge(darts5, on="mesh_id").merge(darts10, on="mesh_id").merge(darts15, on="mesh_id")

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

new_order = ["mesh_id", "avg_normalized_return_darts3", "avg_normalized_return_darts5", "avg_normalized_return_darts10", "avg_normalized_return_darts15", "std_normalized_return_darts3", "std_normalized_return_darts5", "std_normalized_return_darts10", "std_normalized_return_darts15", "avg_length_darts3", "avg_length_darts5", "avg_length_darts10", "avg_length_darts15", "wins_darts3", "wins_darts5", "wins_darts10", "wins_darts15"]

df_results= df_results[new_order]
df_results.drop(df_results.columns[0], axis=1, inplace=True)

print(df_results)

latex_code = df_results.to_latex(index=True, header=True, float_format="%.2f")

print(latex_code)
