import imageio
import jax
import os
import timeit
import pyrallis
import xminigrid
import orbax.checkpoint
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from dataclasses import dataclass
from nn import ActorCriticRNNNull, ActorCriticRNN
from tqdm import trange
from xminigrid.wrappers import GymAutoResetWrapper


@dataclass
class Config:
    # Environment settings
    env_id: str = "XLand-MiniGrid-R4-9x9"
    benchmark_set: str = "high-1m"
    env_seed: int = 42

    # Model settings
    model_type: str = "NULL"
    checkpoint_path: str = "checkpoints/H0_42"

    # Rollout settings
    num_steps: int = 10000
    rng_seed: int = 42

    # Video settings
    video_path: str = "model_rollout.mp4"
    fps: int = 16


def make_env(
    env_id: str = "XLand-MiniGrid-R4-9x9",
    benchmark_set: str = "high-1m",
    seed: int = 42,
):
    env, env_params = xminigrid.make(env_id)
    # do not forget to use auto reset wrapper!
    env = GymAutoResetWrapper(env)

    rng = jax.random.key(seed)
    benchmark = xminigrid.load_benchmark(benchmark_set)
    rulesets = benchmark.sample_ruleset(rng)
    meta_env_params = env_params.replace(ruleset=rulesets)

    return env, meta_env_params


def build_model_rollout(env, env_params, network, network_params, num_steps):
    """
    Build a rollout function that uses the trained model to select actions
    """

    def rollout(rng):
        def _step_fn(carry, _):
            rng, timestep, hstate, prev_action = carry
            rng, policy_rng = jax.random.split(rng)

            direction = timestep.state.agent.direction
            direction_onehot = jax.nn.one_hot(direction, num_classes=4)

            # Prepare observation for the network
            obs = {
                "obs_img": timestep.observation[
                    None, None, ...
                ],  # Add batch and sequence dims
                "obs_dir": direction_onehot[
                    None, None, ...
                ],  # One-hot encoded agent direction
                "prev_action": jnp.array(prev_action)[None, None],
                "prev_reward": timestep.reward[None, None].astype(
                    jnp.float32
                ),  # Previous reward
            }

            # Get action distribution from the network
            action_dist, value, new_hstate = network.apply(network_params, obs, hstate)

            # Sample action from the distribution
            action = action_dist.sample(seed=policy_rng).squeeze()

            # Step the environment
            new_timestep = env.step(env_params, timestep, action)

            return (rng, new_timestep, new_hstate, action), new_timestep

        # Initialize
        rng, reset_rng = jax.random.split(rng)
        timestep = env.reset(env_params, reset_rng)

        # Initialize hidden state for the RNN
        init_hstate = network.initialize_carry(batch_size=1)

        # Run the rollout
        (rng, final_timestep, final_hstate, final_action), transitions = jax.lax.scan(
            _step_fn, (rng, timestep, init_hstate, 0), None, length=num_steps
        )

        return transitions

    return rollout


def load_model(model_type: str, checkpoint_path: str):
    network = None
    if model_type == "H0":
        network = ActorCriticRNNNull(
            num_actions=6,
            obs_emb_dim=32,
            action_emb_dim=32,
            rnn_hidden_dim=1024,
            rnn_num_layers=1,
            head_hidden_dim=256,
        )
    elif model_type == "META":
        network = ActorCriticRNN(
            num_actions=6,
            obs_emb_dim=32,
            action_emb_dim=32,
            rnn_hidden_dim=1024,
            rnn_num_layers=1,
            head_hidden_dim=256,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Convert relative path to absolute path
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)

    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    params = orbax_checkpointer.restore(checkpoint_path)

    return network, params


def render_model_rollout(
    env,
    meta_env_params,
    transitions,
    total_steps,
    video_path="model_rollout.mp4",
    fps=32,
):
    # Create visualization of the model's behavior
    print("Rendering model rollout frames...")
    model_images = []

    for i in trange(total_steps, desc="Rendering frames"):
        model_timestep = jtu.tree_map(lambda x: x[i], transitions)
        model_images.append(env.render(meta_env_params, model_timestep))

    # Save the model rollout as a video
    imageio.mimsave(video_path, model_images, fps=fps)
    print(f"Model rollout saved as {video_path}")

    # Display some key statistics
    total_reward = jnp.sum(transitions.reward)
    episode_lengths = []
    current_length = 0

    for i in range(total_steps):
        current_length += 1
        if transitions.step_type[i] == 2:  # LAST step (episode ended)
            episode_lengths.append(current_length)
            current_length = 0

    print("\nModel Performance Summary:")
    print(f"Total reward collected: {total_reward:.2f}")
    print(f"Number of episodes completed: {len(episode_lengths)}")
    if episode_lengths:
        print(f"Average episode length: {np.mean(episode_lengths):.2f}")
        print(f"Shortest episode: {min(episode_lengths)}")
        print(f"Longest episode: {max(episode_lengths)}")
    else:
        print("No episodes completed in the rollout")


@pyrallis.wrap()  # type: ignore
def run(config: Config):
    env, meta_env_params = make_env(
        env_id=config.env_id, benchmark_set=config.benchmark_set, seed=config.env_seed
    )
    network, params = load_model(config.model_type, config.checkpoint_path)

    # Build the model rollout function
    model_rollout_fn = jax.jit(
        build_model_rollout(
            env, meta_env_params, network, params, num_steps=config.num_steps
        )
    )

    # Initialize the random number generator
    rng = jax.random.key(config.rng_seed)

    # Measure the time taken for the rollout
    print("Running model rollout...")
    start_time = timeit.default_timer()
    transitions = model_rollout_fn(rng)
    end_time = timeit.default_timer()

    print(f"Model rollout completed in {end_time - start_time:.4f} seconds")

    render_model_rollout(
        env,
        meta_env_params,
        transitions,
        config.num_steps,
        video_path=config.video_path,
        fps=config.fps,
    )


if __name__ == "__main__":
    run()  # type: ignore
