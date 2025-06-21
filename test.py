import sys
import os
import pickle
import numpy as np
import gc
import neat
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slimevolley import SlimeVolleyEnv

# Configurable parameters
N_GAMES = 1000  # Number of games to evaluate
MAX_STEPS = 3000  # Max steps per game


def play_one_game(genome, config, max_steps=3000):
    env = SlimeVolleyEnv()
    obs = env.reset()
    done = False
    t = 0
    points_won = 0
    points_lost = 0

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done and t < max_steps:
        obs = np.array(obs)
        action_raw = net.activate(obs)
        action = 1 / (1 + np.exp(-np.array(action_raw)))
        action = (action > 0.5).astype(int)

        obs, reward, done, info = env.step(action)

        if reward > 0:
            points_won += 1
        elif reward < 0:
            points_lost += 1

        t += 1

    return points_won, points_lost


def parallel_eval_games(genome, config, n_games):
    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(n_games):
            futures.append(
                executor.submit(
                    play_one_game,
                    genome,
                    config,
                    MAX_STEPS,
                )
            )

        total_won = 0
        total_lost = 0
        wins = 0
        draws = 0
        losses = 0

        for future in futures:
            points_won, points_lost = future.result()
            total_won += points_won
            total_lost += points_lost
            if points_won > points_lost:
                wins += 1
            elif points_won == points_lost:
                draws += 1
            else:
                losses += 1

    return wins, draws, losses, total_won, total_lost


def main():
    # Load configuration
    config_path = "slimevolley.ini"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    # Load the saved genome
    genome_path = "models/slime_1038.pkl"

    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    print(f"Evaluating genome from {genome_path}")
    print(f"Playing {N_GAMES} games...")

    wins, draws, losses, total_won, total_lost = parallel_eval_games(
        genome, config, N_GAMES
    )

    print(f"\nResults after {N_GAMES} games:")
    print(f"Wins:   {wins}")
    print(f"Draws:  {draws}")
    print(f"Losses: {losses}")
    print(f"Wins ratio: {wins / N_GAMES:.2%}")
    print(f"\nTotal points:")
    print(f"Won:  {total_won}")
    print(f"Lost: {total_lost}")
    print(f"Points ratio: {total_won/(total_won + total_lost):.2%}")


if __name__ == "__main__":
    main()
