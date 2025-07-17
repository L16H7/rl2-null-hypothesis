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


def add_simple_text_overlay(frame, text):
    """Simple text overlay using basic numpy operations when OpenCV is not available."""
    # For simplicity, just return the frame without text when PIL/OpenCV is not available
    # You could implement a basic bitmap text renderer here if needed
    return frame


# Global font cache to avoid repeated font loading
_FONT_CACHE = {}

def get_cached_font(font_size=16):
    """Get a cached font object to avoid repeated loading."""
    from PIL import ImageFont
    
    if font_size in _FONT_CACHE:
        return _FONT_CACHE[font_size]
    
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
    
    _FONT_CACHE[font_size] = font
    return font


def add_text_overlay_fast(image, text, position="top_left", font_size=16, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add text overlay to an image with optimized performance."""
    import cv2
    
    # Use OpenCV for faster text rendering (much faster than PIL)
    if isinstance(image, np.ndarray):
        img = image.copy()
    else:
        img = np.array(image)
    
    # Ensure image is in the right format
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # OpenCV uses BGR
    
    # Calculate text size
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_size / 30.0  # Approximate scaling
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    
    # Calculate position
    img_height, img_width = img.shape[:2]
    
    if position == "top_left":
        x, y = 10, 30
    elif position == "top_right":
        x, y = img_width - text_width - 10, 30
    elif position == "bottom_left":
        x, y = 10, img_height - 10
    elif position == "bottom_right":
        x, y = img_width - text_width - 10, img_height - 10
    else:
        x, y = 10, 30
    
    # Draw background rectangle
    padding = 6
    cv2.rectangle(img, 
                  (x - padding, y - text_height - padding), 
                  (x + text_width + padding, y + baseline + padding), 
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font_face, font_scale, color, thickness, cv2.LINE_AA)
    
    # Convert back to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img


def render_model_rollout(
    env,
    meta_env_params,
    transitions,
    total_steps,
    video_path="model_rollout.mp4",
    fps=4,
):
    # Create visualization of the model's behavior
    print("Rendering model rollout frames...")
    
    # Pre-compute ALL episode and step information in O(n) time
    print("Pre-computing episode information...")
    episode_indices = []  # Which episode each step belongs to
    step_in_episode = []  # Step number within the current episode
    cumulative_rewards = []  # Cumulative reward up to each step
    wins_up_to_step = []  # Number of wins up to each step
    
    current_episode = 0
    current_step_in_ep = 1
    cumulative_reward = 0.0
    total_wins = 0
    
    for i in range(total_steps):
        episode_indices.append(current_episode)
        step_in_episode.append(current_step_in_ep)
        
        # Update cumulative reward
        cumulative_reward += float(transitions.reward[i])
        cumulative_rewards.append(cumulative_reward)
        
        # Check if episode ended
        if transitions.step_type[i] == 2:  # LAST step (episode ended)
            # Check if this episode was a win (positive reward in the episode)
            episode_reward = 0.0
            start_idx = 0 if current_episode == 0 else next(j for j in range(i-1, -1, -1) if transitions.step_type[j] == 2) + 1
            for j in range(start_idx, i + 1):
                episode_reward += float(transitions.reward[j])
            
            if episode_reward > 0:
                total_wins += 1
            
            # Reset for next episode
            current_episode += 1
            current_step_in_ep = 1
        else:
            current_step_in_ep += 1
        
        wins_up_to_step.append(total_wins)
    
    # Try to import cv2 for faster rendering, fallback to PIL if not available
    use_cv2 = True
    try:
        import cv2
    except ImportError:
        use_cv2 = False
        print("OpenCV not available, using PIL (slower). Install opencv-python for faster rendering.")
    
    # Render frames with overlays - now O(n) complexity!
    model_images = []
    for i in trange(total_steps, desc="Rendering frames"):
        model_timestep = jtu.tree_map(lambda x: x[i], transitions)
        frame = env.render(meta_env_params, model_timestep)
        
        # Get pre-computed values - O(1) lookup!
        current_ep_idx = episode_indices[i]
        current_ep_step = step_in_episode[i]
        cumulative_reward = cumulative_rewards[i]
        wins_count = wins_up_to_step[i]
        
        # Add episode info overlay
        episode_text = f"Ep {current_ep_idx + 1} | Step {current_ep_step} | Reward: {cumulative_reward:.1f} | Wins: {wins_count}"
        
        if use_cv2:
            frame = add_text_overlay_fast(frame, episode_text, position="top_left", 
                                         font_size=12, color=(255, 255, 255), bg_color=(0, 0, 0))
        else:
            # Fallback to simple text overlay without heavy PIL operations
            frame = add_simple_text_overlay(frame, episode_text)
        
        model_images.append(frame)

    # Standard quality encoding
    writer = imageio.get_writer(
        video_path, 
        fps=fps, 
        codec='libx264',
        quality=6,  # Standard quality
        pixelformat='yuv420p',
        ffmpeg_params=['-crf', '23', '-preset', 'medium']
    )
    
    for frame in model_images:
        writer.append_data(frame)
    writer.close()
    
    print(f"Model rollout saved as {video_path}")

    # Display some key statistics - now using optimized pre-computed data
    total_reward = jnp.sum(transitions.reward)
    episode_lengths = []
    episode_outcomes = {"WIN": 0, "TIMEOUT": 0, "FAIL": 0}
    current_length = 0

    for i in range(total_steps):
        current_length += 1
        if transitions.step_type[i] == 2:  # LAST step (episode ended)
            episode_lengths.append(current_length)
            
            # Determine episode outcome based on reward and length
            episode_reward = 0.0
            start_idx = 0 if len(episode_lengths) == 1 else sum(episode_lengths[:-1])
            for j in range(start_idx, i + 1):
                episode_reward += float(transitions.reward[j])
            
            if episode_reward > 0:
                outcome = "WIN"
            elif current_length > 50:  # Arbitrary threshold for "long" episode
                outcome = "TIMEOUT"
            else:
                outcome = "FAIL"
            
            episode_outcomes[outcome] += 1
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
    )


if __name__ == "__main__":
    run()  # type: ignore
