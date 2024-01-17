import random
from collections import deque, namedtuple

import torch

from state.gametypes import Gametype

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "is_final", "allowed_targets"))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        self.memory: dict[Gametype, deque[Transition]] = {}
        for game_type in Gametype:
            initial_transitions: list[Transition] = []
            self.memory[game_type] = deque(initial_transitions, maxlen=capacity)

    def push(self, game_type: Gametype, *args: Transition):
        self.memory[game_type].append(Transition(*args))

    def get(self, game_type: Gametype, index: int):
        return self.memory[game_type][index]

    def update(self, game_type: Gametype, index: int, *args: Transition):
        self.memory[game_type][index] = Transition(*args)

    def sample(self, game_type: Gametype, batch_size: int):
        return random.sample(self.memory[game_type], batch_size)

    def length(self, game_type: Gametype):
        return len(self.memory[game_type])


class DQLProcessor:
    def __init__(
        self,
        policy_model: torch.nn.Module,
        target_model: torch.nn.Module,
        gamma: float,
        lr: float,
        batch_size: int,
        tau: float,
    ):
        self.__gamma = gamma
        self.__lr = lr
        self.__batch_size = batch_size
        self.__tau = tau
        self.policy_model = policy_model
        self.target_model = target_model
        self.target_model.load_state_dict(policy_model.state_dict())
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__memory = ReplayMemory(10000)

    def memoize_state(
        self,
        game_type: Gametype,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        is_final: torch.Tensor,
        allowed_targets: torch.Tensor,
    ):
        self.__memory.push(game_type, state, action, next_state, reward, is_final, allowed_targets)

    def optimize_model(self, game_type: Gametype):
        if self.__memory.length(game_type) < self.__batch_size:
            return

        # create optimizer
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.__lr, amsgrad=True
        )
        # sample transitions based on replay memory
        transitions = self.__memory.sample(game_type, self.__batch_size)
        # create a batch of transitions
        batch = Transition(*zip(*transitions))

        # create a boolean mask for each next_state if it is None (terminating) or not
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.__device,
            dtype=torch.bool,
        )
        # filter out elements where the next state is None (the element represents a terminating transition)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        is_final_batch = torch.cat(batch.is_final)
        allowed_targets_batch = torch.cat(batch.allowed_targets)

        

        # Compute state action value (q-value) based on policy_net,
        # and gather the q-value of the action our agent did
        #
        # input of net: the state
        # output of net: the q-value for every possible action
        state_action_values = torch.softmax(
            self.policy_model(state_batch), dim=1
        ).gather(1, action_batch)

        # Now compute the expected optimal q*(s,a) values approximated by our target_net and based on a deterministic policy (arg-max)
        next_state_values = torch.zeros(self.__batch_size, device=self.__device)
        with torch.no_grad():
            # compute the optimal state value (v-value) by picking the reward of the action with the largest reward
            # v*(s) = max(q*(s, a)) where q*(s, a) is approximated by our target_net
            target_values = torch.softmax(
                self.target_model(non_final_next_states), dim=1
            )
            next_state_values = target_values.max(1)[0]
        next_state_values = torch.where(is_final_batch == 1.0, torch.tensor(1.0), next_state_values)

        # Compute expected optimal state action values (q-values) based on state values (v-value) of next state, gamma (discount rate) and the reward
        # q*(s, a) = reward + gamma * v*(s)
        expected_state_action_values = reward_batch + (self.__gamma * next_state_values)

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad.clip_grad_value_(self.policy_model.parameters(), 100)
        optimizer.step()
        # print("Loss %f" % loss.item())
        return loss.item()

    def update_network(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.__tau + target_net_state_dict[key] * (1 - self.__tau)
        self.target_model.load_state_dict(target_net_state_dict)
