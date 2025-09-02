import matplotlib.pyplot as plt
from numpy import ndarray
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def plot_test_results(rewards: ndarray, wins: ndarray, steps: ndarray, normalized_return: ndarray) -> None:
    """
    Plot the rewards obtained on the test data, the number of times the agent wins, and the length (number of time steps)
    of each episode.
    :param rewards: list of rewards obtained on the test data
    :param wins: list of wins obtained on the test data
    :param steps: list of steps of each episode obtained on the test data
    :param normalized_return: list of normalized return obtained on the test data
    """
    nb_episodes = len(rewards)
    cat = [i for i in range(nb_episodes)]
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)  # 4 lignes, 1 colonne, graphique 1
    plt.bar(cat, rewards, label='avg_rewards')
    plt.title('Average rewards on test data')
    plt.legend()

    # Ajouter le deuxième sous-graphe
    plt.subplot(4, 1, 2)  # 4 lignes, 1 colonne, graphique 2
    plt.bar(cat, wins, label='avg_wins', color='orange')
    plt.title('Average wins on test data')
    plt.legend()

    # Ajouter le troisième sous-graphe
    plt.subplot(4, 1, 3)  # 4 lignes, 1 colonne, graphique 3
    plt.bar(cat, steps, label='avg_steps', color='green')
    plt.title('Average length of episodes on test data')
    plt.legend()

    # Ajouter le quatrième sous-graphe
    plt.subplot(4, 1, 4)  # 4 lignes, 1 colonne, graphique 4
    plt.bar(cat, normalized_return, label='avg_normalized_return', color='green')
    plt.title('Average normalized return obtained on test data')
    plt.legend()
    # Ajuster l'espacement entre les sous-graphes
    plt.tight_layout()
    # Afficher les graphiques
    plt.show()


def plot_density(normalized_return: ndarray):

    # Figure
    plt.figure(figsize=(15, 4))

    # --- 1. Histogramme brut ---
    plt.subplot(1, 3, 1)
    plt.hist(normalized_return, bins=10, color="green", edgecolor="black")
    plt.title("Histogramme brut")
    plt.xlabel("Retour normalisé")
    plt.ylabel("Nombre de maillages")

    # --- 2. Histogramme + densité normalisée ---
    plt.subplot(1, 3, 2)
    sns.histplot(normalized_return, bins=10, color="green", edgecolor="black", stat="density", kde=False,
                 label="Histogramme (densité)")
    sns.kdeplot(normalized_return, color="red", linewidth=2, label="Densité KDE", clip=(0,1))
    plt.title("Histogramme + densité normalisée")
    plt.xlabel("Retour normalisé")
    plt.ylabel("Densité")
    plt.legend()

    # --- 3. Histogramme + densité mise à l’échelle ---
    plt.subplot(1, 3, 3)
    counts, bins, _ = plt.hist(normalized_return, bins=10, color="green", edgecolor="black", label="Histogramme")

    # KDE avec mise à l'échelle
    x = np.linspace(0, 1, 200)
    kde = gaussian_kde(normalized_return)
    kde_scaled = kde(x) * len(normalized_return) * (bins[1] - bins[0])
    plt.plot(x, kde_scaled, color="red", linewidth=2, label="Densité mise à l’échelle")

    plt.title("Histogramme + densité à l’échelle")
    plt.xlabel("Retour normalisé")
    plt.ylabel("Nombre de maillages")
    plt.xlim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_training_results(rewards: ndarray, wins: ndarray, steps: ndarray) -> None:
    """
    Plot the rewards obtained during training, the number of times the agent wins, and the length (number of time steps)
    of each episode.
    :param rewards: list of rewards obtained during training
    :param wins: list of wins obtained during training
    :param steps: list of steps obtained during training
    """
    nb_episodes = len(rewards)
    real_runs = 1

    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Learning Rewards, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()
    plt.figure()
    plt.plot(wins)
    plt.xlabel('Episodes')
    plt.ylabel('Wins')
    plt.title('Learning Wins, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()
    plt.figure()
    plt.plot(steps)
    plt.xlabel('Episodes')
    plt.ylabel('Number of steps per episode')
    plt.title('Learning steps, nb_episodes={}'.format(nb_episodes) + ', nb_runs={}'.format(real_runs))
    plt.legend(loc="best")
    plt.show()