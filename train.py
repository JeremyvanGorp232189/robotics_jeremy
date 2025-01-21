import os
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from clearml import Task
from wrapper_jeremy import CustomEnv  # Adjusted to match the actual filename

# Configure environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['WANDB_API_KEY'] = '7fabefdeafaca23d0547b1d4dc39d59511b55c9f'

# Initialize ClearML Task
task = Task.init(project_name='Mentor Group M/Group 2', task_name='Name')
task.set_base_docker('deanis/2023y2b-rl:latest')

# Initialize WandB
wandb_project_name = "custom_wandb_project"
wandb_run_name = "experiment_run"
wandb_run = wandb.init(project=wandb_project_name, name=wandb_run_name, sync_tensorboard=True)

# Define environment
env = CustomEnv()
check_env(env)

# Create directories for saving models
save_dir = f"saved_models/{wandb_run.id}"
os.makedirs(save_dir, exist_ok=True)

# Parse arguments for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for PPO optimizer")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per update")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for training")
parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
parser.add_argument("--policy", type=str, default="MlpPolicy", help="Policy architecture")
parser.add_argument("--clip_range", type=float, default=0.15, help="Clip range")
parser.add_argument("--value_coefficient", type=float, default=0.5, help="Value function coefficient")
args = parser.parse_args()

# Initialize PPO model
model = PPO(
    args.policy, 
    env, 
    verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    tensorboard_log=f"wandb_runs/{wandb_run.id}"
)

# Define WandB callback
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=save_dir,
    verbose=2
)

# Training parameters
total_timesteps = 5_000_000

# Train the model
model.learn(
    total_timesteps=total_timesteps,
    callback=wandb_callback,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"wandb_runs/{wandb_run.id}"
)

# Save final model
final_model_path = f"{save_dir}/{total_timesteps}_final_model"
model.save(final_model_path)

# Save model to WandB
wandb.save(final_model_path)

# End WandB run
wandb_run.finish()

# Example CLI Usage:
# python train.py --learning_rate 0.0001 --batch_size 32 --n_steps 2048 --n_epochs 10 --gamma 0.98 --policy MlpPolicy --clip_range 0.15 --value_coefficient 0.5
# python train.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --gamma 0.99 --policy MlpPolicy --clip_range 0.2 --value_coefficient 0.5
