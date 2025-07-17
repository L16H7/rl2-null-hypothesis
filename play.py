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
    fps: int = 4
    high_quality: bool = True  # Use high quality video encoding


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


def add_text_overlay(image, text, position="top_left", font_size=16, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add text overlay to an image with anti-aliasing."""
    from PIL import Image, ImageDraw, ImageFont
    
    # Convert numpy array to PIL Image and ensure it's in RGB mode
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image).convert('RGB')
    else:
        pil_image = image.convert('RGB')
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Try to use a better font, fallback to default if not available
    font = None
    font_paths = [
        "/System/Library/Fonts/Arial.ttf",  # macOS
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "Arial.ttf",  # Windows/general
        "arial.ttf",  # Linux
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    
    # Get text dimensions
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    else:
        # Rough estimation if font fails
        text_width = len(text) * 8
        text_height = 16
    
    # Calculate position
    img_width, img_height = pil_image.size
    
    if position == "top_left":
        x, y = 10, 10
    elif position == "top_right":
        x, y = img_width - text_width - 10, 10
    elif position == "bottom_left":
        x, y = 10, img_height - text_height - 10
    elif position == "bottom_right":
        x, y = img_width - text_width - 10, img_height - text_height - 10
    else:
        x, y = 10, 10
    
    # Draw semi-transparent background rectangle
    padding = 6
    overlay = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Create background with transparency
    bg_rgba = bg_color + (180,) if len(bg_color) == 3 else bg_color
    overlay_draw.rectangle([
        x - padding, y - padding,
        x + text_width + padding, y + text_height + padding
    ], fill=bg_rgba)
    
    # Composite the overlay
    pil_image = Image.alpha_composite(pil_image.convert('RGBA'), overlay).convert('RGB')
    draw = ImageDraw.Draw(pil_image)
    
    # Draw text with anti-aliasing
    if font:
        draw.text((x, y), text, fill=color, font=font)
    else:
        draw.text((x, y), text, fill=color)
    
    # Convert back to numpy array
    return np.array(pil_image)


def render_model_rollout(
    env,
    meta_env_params,
    transitions,
    total_steps,
    video_path="model_rollout.mp4",
    fps=4,
    high_quality=True,
):
    # Create visualization of the model's behavior
    print("Rendering model rollout frames...")
    model_images = []

    # Pre-compute episode information
    episode_info = []
    current_episode = 0
    episode_step = 0
    episode_reward = 0.0
    
    for i in range(total_steps):
        episode_step += 1
        episode_reward += float(transitions.reward[i])
        
        # Check if episode ended
        if transitions.step_type[i] == 2:  # LAST step (episode ended)
            # Determine episode outcome based on reward and length
            if episode_reward > 0:
                outcome = "WIN"
                outcome_color = (0, 255, 0)  # Green
            else:
                # Check if it's a timeout (long episode with no reward) vs failure
                if episode_step > 50:  # Arbitrary threshold for "long" episode
                    outcome = "TIMEOUT"
                    outcome_color = (255, 255, 0)  # Yellow
                else:
                    outcome = "FAIL"
                    outcome_color = (255, 0, 0)  # Red
            
            episode_info.append({
                'end_step': i,
                'length': episode_step,
                'reward': episode_reward,
                'outcome': outcome,
                'color': outcome_color
            })
            
            # Reset for next episode
            current_episode += 1
            episode_step = 0
            episode_reward = 0.0
        else:
            episode_info.append(None)

    # Render frames with overlays
    for i in trange(total_steps, desc="Rendering frames"):
        model_timestep = jtu.tree_map(lambda x: x[i], transitions)
        frame = env.render(meta_env_params, model_timestep)
        
        # Find current episode info by tracking through all steps
        current_ep_idx = 0
        current_ep_step = 0
        current_ep_reward = 0.0
        wins_count = 0
        
        for j in range(i + 1):
            current_ep_step += 1
            current_ep_reward += float(transitions.reward[j])
            if episode_info[j] is not None:  # Episode ended at step j
                # Count wins up to this point
                if episode_info[j]['outcome'] == 'WIN':
                    wins_count += 1
                    
                if j < i:  # This episode ended before current step i
                    current_ep_idx += 1
                    current_ep_step = 1  # Start counting from 1 for next episode
                    current_ep_reward = float(transitions.reward[j + 1]) if j + 1 <= i else 0.0
                    # Continue tracking from the next step after episode end
                    for k in range(j + 2, i + 1):
                        current_ep_step += 1
                        current_ep_reward += float(transitions.reward[k])
                        if episode_info[k] is not None and k < i:
                            if episode_info[k]['outcome'] == 'WIN':
                                wins_count += 1
                            current_ep_idx += 1
                            current_ep_step = 1
                            current_ep_reward = float(transitions.reward[k + 1]) if k + 1 <= i else 0.0
                    break
        
        # Calculate cumulative reward up to current step
        cumulative_reward = sum(float(transitions.reward[j]) for j in range(i + 1))
        
        # Add episode info overlay with cumulative reward and wins
        episode_text = f"Ep {current_ep_idx + 1} | Step {current_ep_step} | Reward: {cumulative_reward:.1f} | Wins: {wins_count}"
        frame = add_text_overlay(frame, episode_text, position="top_left", 
                                 font_size=12, color=(255, 255, 255), bg_color=(0, 0, 0))
        
        model_images.append(frame)

    # Save the model rollout as a video with quality settings
    if high_quality:
        # High quality H.264 encoding
        writer = imageio.get_writer(
            video_path, 
            fps=fps, 
            codec='libx264',
            quality=9,  # Very high quality (1-10 scale)
            pixelformat='yuv420p',  # Better compatibility
            macro_block_size=None,
            ffmpeg_params=[
                '-crf', '15',  # Constant Rate Factor: lower = higher quality
                '-preset', 'slow',  # Better compression at cost of speed
                '-profile:v', 'high',  # H.264 profile
                '-level', '4.0',
                '-x264-params', 'ref=6:bframes=8:b-adapt=2:direct=auto:me=umh:subme=10:merange=24:trellis=2'
            ]
        )
    else:
        # Standard quality encoding
        writer = imageio.get_writer(
            video_path, 
            fps=fps, 
            codec='libx264',
            quality=6,  # Standard quality
            pixelformat='yuv420p',
            ffmpeg_params=['-crf', '23', '-preset', 'medium']
        )
    
    print(f"Writing {len(model_images)} frames to {'high-quality' if high_quality else 'standard-quality'} video...")
    for frame in model_images:
        writer.append_data(frame)
    writer.close()
    
    print(f"Model rollout saved as {video_path}")

    # Display some key statistics
    total_reward = jnp.sum(transitions.reward)
    episode_lengths = []
    episode_outcomes = {"WIN": 0, "TIMEOUT": 0, "FAIL": 0}
    current_length = 0

    for i in range(total_steps):
        current_length += 1
        if transitions.step_type[i] == 2:  # LAST step (episode ended)
            episode_lengths.append(current_length)
            
            # Count outcomes
            if episode_info[i] is not None:
                episode_outcomes[episode_info[i]['outcome']] += 1
            
            current_length = 0

    print("\nModel Performance Summary:")
    print(f"Total reward collected: {total_reward:.2f}")
    print(f"Number of episodes completed: {len(episode_lengths)}")
    print(f"Episode outcomes: {episode_outcomes}")
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
        high_quality=config.high_quality,
    )


if __name__ == "__main__":
    run()  # type: ignore
