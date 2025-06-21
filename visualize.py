"""
Visualize NEAT agent playing SlimeVolley
"""

import sys
import os
import pickle
import numpy as np
import imageio
import argparse
from evojax.task.slimevolley import SlimeVolley


def load_genome(path):
    """Load a NEAT genome from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def run_and_record_slimevolley(genome, gif_path, max_steps=3000):
    """Run the game and record it as a GIF."""
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    env = SlimeVolley(max_steps=max_steps, test=True)
    import jax

    key = jax.random.PRNGKey(0)
    state = env.reset(jax.random.split(key, 1))
    done = False
    frames = []
    t = 0

    # Set up NEAT network
    import neat

    CONFIG_PATH = "slimevolley.ini"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    while not done and t < max_steps:
        # Get observation and compute action
        obs = np.array(state.obs[0])
        action_raw = net.activate(obs)
        output = 1 / (1 + np.exp(-np.array(action_raw)))
        action = (output > 0.5).astype(int)

        # Step environment
        import jax.numpy as jnp

        next_state, reward, terminated = env.step(state, jnp.expand_dims(action, 0))

        # Convert JAX arrays to numpy for rendering
        def deep_to_numpy_state(state):
            if hasattr(state, "__dataclass_fields__"):
                return state.__class__(
                    **{
                        k: deep_to_numpy(getattr(state, k))
                        for k in state.__dataclass_fields__
                    }
                )
            return state

        def deep_to_numpy(obj):
            import jax
            import jax.numpy as jnp
            import numpy as np

            if isinstance(obj, (jax.Array, jnp.ndarray)):
                arr = np.asarray(obj)
                return arr.item() if arr.shape == () else arr
            elif isinstance(obj, dict):
                return {k: deep_to_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(deep_to_numpy(v) for v in obj)
            elif hasattr(obj, "__dataclass_fields__"):
                return obj.__class__(
                    **{
                        k: deep_to_numpy(getattr(obj, k))
                        for k in obj.__dataclass_fields__
                    }
                )
            return obj

        # Render and save frame
        render_state = deep_to_numpy_state(state)
        frame = np.asarray(SlimeVolley.render(render_state))
        frames.append(frame)

        done = bool(terminated[0])
        state = next_state
        t += 1

    # Save GIF
    imageio.mimsave(gif_path, frames, duration=0.005, loop=0)
    print(f"[INFO] Game recording saved to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize NEAT agent playing SlimeVolley"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/slime_1038.pkl",
        help="Path to the model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output GIF (default: results/model_name.gif)",
    )
    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        args.output = f"results/{model_name}_play.gif"

    genome = load_genome(args.model)
    run_and_record_slimevolley(genome, args.output)
