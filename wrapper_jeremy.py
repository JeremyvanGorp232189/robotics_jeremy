import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

# Created by Jeremy, Student Number: 232189

class CustomEnv(gym.Env):  # Updated class name to match the expected name in train.py
    def __init__(self, enable_visualization=False, max_steps=1000):
        super().__init__()
        self.enable_visualization = enable_visualization
        self.max_steps = max_steps

        # Initialize simulation with agent configuration
        self.simulation = Simulation(agent_count=1, render=self.enable_visualization)

        # Define boundaries for the environment
        self.env_limits = {
            "x_range": (-0.2, 0.25),
            "y_range": (-0.18, 0.22),
            "z_range": (0.12, 0.29)
        }

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.array([self.env_limits["x_range"][0], self.env_limits["y_range"][0], self.env_limits["z_range"][0]]),
                -np.array([self.env_limits["x_range"][1], self.env_limits["y_range"][1], self.env_limits["z_range"][1]])
            ], axis=0, dtype=np.float32),
            high=np.concatenate([
                np.array([self.env_limits["x_range"][1], self.env_limits["y_range"][1], self.env_limits["z_range"][1]]),
                np.array([self.env_limits["x_range"][1], self.env_limits["y_range"][1], self.env_limits["z_range"][1]])
            ], axis=0, dtype=np.float32),
            dtype=np.float32
        )

        self.current_step = 0

    def reset(self, seed=None):
        if seed:
            np.random.seed(seed)

        # Generate a new goal position within the defined limits
        self.target_position = np.array([
            np.random.uniform(*self.env_limits["x_range"]),
            np.random.uniform(*self.env_limits["y_range"]),
            np.random.uniform(*self.env_limits["z_range"])
        ])

        initial_state = self.simulation.reset(agent_count=1)

        # Combine the agent's initial position with the target position for observation
        observation = np.concatenate([
            self.simulation.get_agent_position(self.simulation.robotIds[0]),
            self.target_position
        ]).astype(np.float32)

        self.current_step = 0
        return observation, {}

    def step(self, action):
        # Augment action array for simulation compatibility
        action_with_extra_dim = np.append(action, 0)
        
        # Execute the action in the simulation
        self.simulation.run([action_with_extra_dim])
        agent_position = self.simulation.get_agent_position(self.simulation.robotIds[0])

        # Observation combines the current agent position and target position
        observation = np.concatenate([agent_position, self.target_position], dtype=np.float32)

        # Reward calculation based on distance to the target
        distance_to_target = np.linalg.norm(agent_position - self.target_position)
        reward = -distance_to_target - 0.01  # Penalize for each step

        # Check termination conditions
        reached_target = distance_to_target <= 0.001
        if reached_target:
            reward += 50  # Bonus for achieving the goal

        max_steps_exceeded = self.current_step >= self.max_steps
        self.current_step += 1

        return observation, reward, reached_target, max_steps_exceeded, {}

    def capture_frame(self):
        return self.simulation.get_plate_image()

    def render(self, mode='human'):
        if self.enable_visualization:
            print("Rendering the environment...")

    def close(self):
        self.simulation.close()
