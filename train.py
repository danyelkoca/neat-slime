"""
Example NEAT implementation for SlimeVolley using neat-python.
"""

import neat
from neat.parallel import ParallelEvaluator
import os
import sys
import numpy as np
import gc
import pickle
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from slimevolley import SlimeVolleyEnv
import json


# Configurable parameters
MAX_GENERATIONS = 2000  # Number of generations to run NEAT
MAX_STEPS = 3000  # Max steps per episode
EPISODES = 5  # Number of episodes per evaluation
WIN_BONUS = 200
SCORE_SCALE = 100
SHAPED_DECAY = 500


def eval_genome_slimevolley(genome, config, max_steps, episodes, generation=0):
    total_fitness = 0.0
    total_game_reward = 0.0
    total_shaped_reward = 0.0
    shaping_weight = max(0.0, 1.0 - generation / SHAPED_DECAY)

    for _ in range(episodes):
        try:
            env = SlimeVolleyEnv(random_ball_start=generation >= 1000)
            obs = env.reset()
            net = neat.nn.FeedForwardNetwork.create(genome, config)

            episode_fitness = 0.0
            episode_score = 0.0
            shaped_reward_total = 0.0

            for _ in range(max_steps):
                action_raw = net.activate(obs)
                action = (1 / (1 + np.exp(-np.array(action_raw)))) > 0.5
                obs, reward, done, _ = env.step(action.astype(int))

                x, _, _, _, ball_x, _, *_ = obs
                proximity = (
                    np.exp(-abs(x - ball_x)) if generation < SHAPED_DECAY else 0.0
                )
                shaped_component = shaping_weight * proximity
                score_component = SCORE_SCALE * reward

                episode_fitness += shaped_component + score_component
                shaped_reward_total += shaped_component
                episode_score += reward

                if done:
                    break

            if episode_score > 0:
                episode_fitness += WIN_BONUS
            elif episode_score < 0:
                episode_fitness -= WIN_BONUS

            total_fitness += episode_fitness
            total_game_reward += episode_score
            total_shaped_reward += shaped_reward_total

        except Exception as e:
            print(f"Exception in fitness eval: {e}")

    avg_fitness = total_fitness / episodes
    avg_game_reward = total_game_reward / episodes
    avg_shaped_reward = total_shaped_reward / episodes
    return avg_fitness, avg_game_reward, avg_shaped_reward


def parallel_eval_genome(genome, config):
    # Use the global generation variable
    global_parallel_generation += 1
    avg_fitness, avg_game_reward, avg_shaped_reward = eval_genome_slimevolley(
        genome,
        config,
        max_steps=MAX_STEPS,
        episodes=EPISODES,
        generation=global_parallel_generation,  # Pass the global generation
    )
    genome.fitness = float(avg_fitness)
    genome.game_reward = float(avg_game_reward)
    genome.shaped_reward = float(avg_shaped_reward)
    return avg_fitness  # Only return the fitness for NEAT


def parallel_eval_genomes(genomes, config, generation):
    with ProcessPoolExecutor() as executor:
        futures = []
        for genome_id, genome in genomes:
            futures.append(
                executor.submit(
                    eval_genome_slimevolley,
                    genome,
                    config,
                    MAX_STEPS,
                    EPISODES,
                    generation,
                )
            )
        for (genome_id, genome), future in zip(genomes, futures):
            avg_fitness, avg_game_reward, avg_shaped_reward = future.result()
            genome.fitness = float(avg_fitness)
            genome.game_reward = float(avg_game_reward)
            genome.shaped_reward = float(avg_shaped_reward)


metrics = []
models = {}


def run_slimevolley():

    config_path = os.path.join(os.path.dirname(__file__), "./slimevolley.ini")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = ParallelEvaluator(8, parallel_eval_genome)

    def eval_genomes_and_log(genomes, config):
        global metrics
        global models
        generation = p.generation
        parallel_eval_genomes(genomes, config, generation)

        best = max((g for _, g in genomes), key=lambda g: g.fitness)
        avg_fitness = np.mean([g.fitness for _, g in genomes])
        best_fitness = best.fitness
        avg_game_reward = np.mean([getattr(g, "game_reward", 0) for _, g in genomes])
        avg_shaped_reward = np.mean(
            [getattr(g, "shaped_reward", 0) for _, g in genomes]
        )
        best_game_reward = getattr(best, "game_reward", 0)
        best_shaped_reward = getattr(best, "shaped_reward", 0)

        metric = {
            "generation": generation,
            "best_fitness": float(best_fitness),
            "best_game_reward": float(best_game_reward),
            "best_shaped_reward": float(best_shaped_reward),
            "avg_fitness": float(avg_fitness),
            "avg_game_reward": float(avg_game_reward),
            "avg_shaped_reward": float(avg_shaped_reward),
        }
        metrics.append(metric)
        models[generation] = best
        if generation % 10 == 0 or generation == MAX_GENERATIONS - 1:
            for key, model in models.items():
                model_path = f"models/slime_{key}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

            metrics_path = f"logs/slime_{generation}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(
                f"[INFO] Saved best genome and metrics to {model_path}, {metrics_path}"
            )
            metrics = []
            gc.collect()

    winner = p.run(eval_genomes_and_log, MAX_GENERATIONS)
    print("\nBest genome:\n", winner)


if __name__ == "__main__":
    run_slimevolley()
