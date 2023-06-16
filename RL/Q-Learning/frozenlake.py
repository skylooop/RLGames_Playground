import gymnasium
from absl import app, flags
import wandb


def main(_):
    frozenlake = gymnasium.make("FrozenLake-v1", is_slippery=False,
                                map_name="4x4", render_mode="rgb_array")
    
    state_space = frozenlake.observation_space.n
    action_space = frozenlake.action_space.n
    print(f"State space dimensionality: {state_space}")
    print(f"Action space dimensionality: {action_space}")
if __name__ == "__main__":
    app.run(main)