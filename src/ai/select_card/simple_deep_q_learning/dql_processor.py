import random
from collections import deque, namedtuple

import torch

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity: int):
        initial_transitions: list[Transition] = []
        self.memory = deque(initial_transitions, maxlen=capacity)

    def push(self, *args: Transition):
        self.memory.append(Transition(*args))

    def get(self, index: int):
        return self.memory[index]

    def update(self, index: int, *args: Transition):
        self.memory[index] = Transition(*args)

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
        self.__memory = ReplayMemory(1000)

    def memoize_state(
        self, state: list[int], action: int, reward: float, next_state: list[int]
    ):
        self.__memory.push(
            torch.tensor([state], dtype=torch.float, device=self.__device),
            torch.tensor([[action]], dtype=torch.int64, device=self.__device),
            torch.tensor([next_state], dtype=torch.float, device=self.__device),
            torch.tensor([reward], dtype=torch.float, device=self.__device),
        )

    def optimize_model(self):
        if len(self.__memory) < self.__batch_size:
            return

        # create optimizer
        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=self.__lr, amsgrad=True
        )
        # sample transitions based on replay memory
        transitions = self.__memory.sample(self.__batch_size)
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

        # Compute state action value (q-value) based on policy_net,
        # and gather the q-value of the action our agent did
        #
        # input of net: the state
        # output of net: the q-value for every possible action
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Now compute the expected optimal q*(s,a) values approximated by our target_net and based on a deterministic policy (arg-max)
        next_state_values = torch.zeros(self.__batch_size, device=self.__device)
        with torch.no_grad():
            # compute the optimal state value (v-value) by picking the reward of the action with the largest reward
            # v*(s) = max(q*(s, a)) where q*(s, a) is approximated by our target_net
            next_state_values[non_final_mask] = self.target_model(
                non_final_next_states
            ).max(1)[0]
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

    def update_network(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.__tau + target_net_state_dict[key] * (1 - self.__tau)
        self.target_model.load_state_dict(target_net_state_dict)
