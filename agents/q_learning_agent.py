import numpy as np
from random import randint
from agents import BaseAgent


class QLearningAgent(BaseAgent):
    def __init__(
            self,
            agent_number,
            alpha=0.1,
            gamma=0.9,
            epsilon=1.0,
            training=True
    ):
        """
            Q-Learning agent for grid cleaning.
            Args:
                agent_number: The index of the agent in the environment.
                alpha: Learning rate (default: 0.1).
                gamma: Discount factor (default: 0.9).
                epsilon: Exploration rate (default: 0).
                training: Whether agent is in training mode (default: True)
        """
        super().__init__(agent_number)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        self.training = training
        self.already_visited = set()
        self.cleaned_tiles = set()
        self.grid_state = None
        self.last_state = None
        self.second_last_state = None
        self.epsilon_min = 0.1

    def process_reward(
            self,
            observation: np.ndarray,
            reward: float,
            info: dict,
            state: tuple,
            action: int,
            old_state: tuple
    ):
        """
            Process reward based on given reward
            Args:
                observation: Observation corresponding to reward.
                reward: Reward gained.
                info: Info corresponding to reward.
                state: State corresponding to reward.
                action: Action that reward was earned on.
                old_state: Old state before reward.
        """
        agent_pos = info['agent_pos'][self.agent_number]

        # If not making a move, give bad reward
        if action == 4:
            reward = -1000
            return reward

        # If returning to previous state, give bad reward
        if not self.second_last_state:
            self.second_last_state = old_state
            self.last_state = state
        else:
            if state == self.second_last_state and \
                    info['agent_moved'][self.agent_number]:
                self.second_last_state = old_state
                self.last_state = state
                reward = -4
                return reward
            elif not info['agent_moved'][self.agent_number]:
                self.second_last_state = old_state
                self.last_state = state
                return reward
            self.second_last_state = old_state
            self.last_state = state

        # If moving to state that agent has already been in, give bad reward
        if state in self.already_visited and \
                info['agent_moved'][self.agent_number]:
            reward = -2
            return reward

        # If dirt cleaned with move, update grid_state and add tile to
        # cleaned tiles
        if reward == 20:
            self.cleaned_tiles.add(agent_pos)
            self.grid_state[agent_pos[0]][agent_pos[1]] = 3

        return reward

    def take_action(
            self,
            observation: np.ndarray,
            info: dict
    ) -> int:
        """
            Return the action based on value in q table or randomly if
            epsilon
            Args:
                observation: Observation to compute action on.
                info: Current info to compute action on.
        """
        state = self.get_state_from_info(observation, info)
        self.already_visited.add(state)

        if np.random.uniform() < self.epsilon and self.training:
            # Exploration: Select a random action
            action = randint(0, 8)
        else:
            # Exploitation: Select the action with the highest Q-value
            action = self._get_best_action(state)
        return action

    def decay_epsilon(self, iters):
        """
            Decay the epsilon based on the amount of iterations
            Args:
                iters: Amount of iterations to base decay on.
        """
        if self.epsilon > self.epsilon_min and self.training:
            self.epsilon -= (1/iters)

    def update_q_values(
            self,
            state: tuple,
            action: int,
            reward: float,
            next_state: tuple
    ) -> None:
        """
            Update q values in q table for given parameters.
            Args:
                state: Old state used to compute new q values.
                action: Action that is done to compute q values.
                reward: Reward corresponding to action in old state.
                next_state: New state after performing action in old state.
        """
        q_value = self.q_table.get((state, action), 0.0)
        max_q_value = max(self.q_table.get((next_state, a), 0.0)
                          for a in range(9))
        new_q_value = q_value + self.alpha * \
            (reward + self.gamma * max_q_value - q_value)

        self.q_table[(state, action)] = new_q_value

    def get_state_from_info(
            self,
            observation: np.ndarray,
            info: dict
    ) -> tuple:
        """
            Get state from given observation and info.
            Args:
                observation: Observation to compute state from.
                info: Info to compute position from.
        """
        # Extract agent position from info
        agent_pos = info['agent_pos'][self.agent_number]

        # Initialize grid state if it is None
        if self.grid_state is None:
            self.grid_state = self.get_dirtless_grid(observation)

        # Flatten the grid state such that it can be used as a key in
        # q table.
        grid_state = tuple(self.grid_state.flatten())

        state = tuple(np.concatenate((grid_state, agent_pos)))

        return state

    def reset_parameters(self) -> None:
        """
            Reset agent parameters for when grid is completed.
        """
        self.already_visited = set()
        self.cleaned_tiles = set()
        self.last_state = None
        self.second_last_state = None
        self.grid_state = None

    def _get_best_action(
            self,
            state: tuple
    ) -> int:
        """
            Get best action from Q table.
            Args:
                state: State to get best action for.
        """
        # Get the action with the highest Q-value for the given state
        q_values = [self.q_table.get((state, a), 0.0) for a in range(9)]

        # If all values for a given state are 0, argmax always takes first
        # index, which equals a move down, so we want to randomly choose the
        # move if everything is 0.
        if all(v == 0.0 for v in q_values):
            return randint(0, 8)
        return int(np.argmax(q_values))

    def get_dirtless_grid(
            self,
            observation
    ) -> np.array:
        """
            Get dirtless grid from given observation.
            Args:
                observation: Grid to compute dirtless grid from.
        """
        grid = observation.copy()
        dirt_mask = (grid == 3)
        grid[dirt_mask] = 0
        random_obstacle_mask = (grid == -1)
        grid[random_obstacle_mask] = 0
        return grid
