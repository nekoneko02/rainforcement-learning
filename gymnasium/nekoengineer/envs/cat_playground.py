from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
import math

class CatPlaygroundEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, speed_max=5):
        self.window_size = 512
        self.pixel_size = 512

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "cat": spaces.Box(0, self.pixel_size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.pixel_size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 2 continuous actions, corresponding to moving speed, angle
        self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([speed_max, math.pi]), shape=(2,), dtype=float)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"cat": self._cat_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._cat_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._cat_location = self.np_random.integers(0, self.pixel_size - 1, size=2, dtype=int)

        # We will sample the target's location randomly until it does not
        # coincide with the agent's location
        self._target_location = self._cat_location
        while np.array_equal(self._target_location, self._cat_location):
            self._target_location = self.np_random.integers(0, self.pixel_size - 1, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        speed, angle = action
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction(speed, angle)
        # We use `np.clip` to make sure we don't leave the grid
        self._cat_location = np.clip(
            self._cat_location + direction, 0, self.pixel_size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    def _action_to_direction(self, speed, angle):
        # The direction is a 2D vector, so we need to compute its components
        return np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=int)
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode != "rgb_array":  # rgb_array
            return
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.pixel_size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the cat
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._cat_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

    def close(self):
        pass
