from dataclasses import dataclass
from typing import Any
import numpy as np
import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ai.select_card.models.shared.encoding import get_action_from_card, pick_highest_valid_card
from state.card import Card

# Detect anomalies like "inf" tensor values
torch.autograd.set_detect_anomaly(True)

class Agent(torch.nn.Module):
    def __init__(self, actor: torch.nn.Module, critic: torch.nn.Module ):
        super().__init__()
        self.critic = critic
        self.actor = actor

    def get_value(self, x) -> torch.Tensor:
        # In this algorithm the critic doesn't compute the q-value but instead the v-value. This is valid, 
        # since the the policy always takes the action with the highest probability, see also
        # get_action_and_value()::action = probs.sample()
        # In other words, the q_value based on state + action can be replaced by the v_value based on the next state?
        return self.critic(x)

    def get_action_and_value(self, x, playable_cards: list[Card], action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            card = pick_highest_valid_card(logits, playable_cards)
            action = torch.tensor(get_action_from_card(card))
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

@dataclass
class PpoProcessorConfig:
    # Environment specific arguments
    observation_space_shape: tuple[int]
    """the shape of the observation space, e.g. (2,) for two observation fields [a, b]"""
    action_space_shape: tuple[int]
    """the shape of the action space, e.g. (2,) for two action options [a, b]"""

    # Agent
    agent: Agent
    """The agent containing both, the actor and the critic"""

    # Algorithm specific arguments
    num_steps: int
    """the number of steps to run in each environment per policy rollout"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""


# Aˆt = δt + (γλ)δt+1 + ··· + ··· + (γλ)T −t+1δT − 1
# advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam


# Inspired by https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
# See also PPO-Paper https://arxiv.org/pdf/1707.06347.pdf
class PpoProcessor:
    def __init__(self, config: PpoProcessorConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.config.num_steps, 1)
            + self.config.observation_space_shape
        ).to(self.device)
        self.next_obs = torch.zeros(
            (self.config.num_steps, 1)
            + self.config.observation_space_shape
        ).to(self.device)
        self.actions = torch.zeros(
            (self.config.num_steps, 1)
            + self.config.action_space_shape
        ).to(self.device)
        self.logprobs = torch.zeros((self.config.num_steps, 1)).to(
            self.device
        )
        self.rewards = torch.zeros((self.config.num_steps, 1)).to(
            self.device
        )
        self.dones = torch.zeros((self.config.num_steps, 1)).to(
            self.device
        )
        self.next_dones = torch.zeros((self.config.num_steps, 1)).to(
            self.device
        )
        self.values = torch.zeros((self.config.num_steps, 1)).to(
            self.device
        )

        self.combined_agent = config.agent

        self.optimizer = optim.Adam(
            self.combined_agent.parameters(), lr=self.config.learning_rate, eps=1e-5
        )

        self.config.batch_size = int(self.config.num_steps)
        # self.config.minibatch_size = int(self.config.batch_size // self.config.num_minibatches)
        self.config.minibatch_size = 1

    def memoize_timestep(
        self,
        step: int,
        initial_done: bool,
        next_done: bool,
        initial_obs: Any,
        next_obs: Any,
        action: int,
        critic_value: float,
        reward: float,
        logprob: float,
    ):
        """
        Parameters:
        initial_done (bool): if the env is initially done for that step
        initial_obs (Any): the initial observation for that step
        action (int): The index of the action chosen by the actor in that step
        critic_value (float): The value of the critic for the action the actor has made
        reward (float): The reward for the action the actor has made
        logprob (float): The gradient (logarithmic probabilities?) for the action the actor has made
        """
        self.dones[step] = initial_done
        self.next_dones[step] = next_done
        self.obs[step] = initial_obs
        self.next_obs[step] = next_obs
        self.actions[step] = action
        self.values[step] = critic_value
        self.rewards[step] = reward
        self.logprobs[step] = logprob

    # TODO: Step wird nicht richtig geressetet?

    def compute_advantages_and_returns(self):
        with torch.no_grad():
            next_value = self.combined_agent.get_value(self.next_obs[-1]).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            for t in reversed(range(self.config.num_steps)):
                if t == self.config.num_steps - 1:
                    nextnonterminal = 1.0 - self.next_dones[-1]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                delta = (
                    self.rewards[t]
                    + self.config.gamma * nextvalues * nextnonterminal
                    - self.values[t]
                )
                # normally an array for each env, let's think of it as only one value
                # Since we iterate through the time steps in reverse, we start at the end and 
                # compute the advantages based on the last one in reverse
                # A_t = δt + (γλ)δt+1 + ··· + ··· + (γλ)T −t+1δT − 1
                # A_t = R_t - b(s_t)
                advantages[t] = lastgaelam = (
                    delta
                    + self.config.gamma # Discount
                    * self.config.gae_lambda # ??
                    * nextnonterminal # Either 0 (terminal) or 1 (non-terminal)
                    * lastgaelam # The last advantage
                )
            returns = advantages + self.values

            return (advantages, returns)

    def optimize_actor_critic(self, advantages: torch.Tensor, returns: torch.Tensor):
        """
        Parameters:
        advantages (Tensor): The advantage value (float) for each step
        returns (Tensor): The return value (float) for each step
        """

        # flatten the batch
        b_obs = self.obs.reshape((-1,) + self.config.observation_space_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.config.action_space_shape)
        b_values = self.values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(self.config.batch_size)
        clipfracs = []
        for epoch in range(self.config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.config.batch_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = self.combined_agent.get_action_and_value(b_obs[mb_inds], playable_cards=[], action=b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                # print("logratio", logratio)
                # TODO: Check why logratio gets so big (> 100). .exp() is e^logratio. with such a huge logratio it results in something like 10^27 or even larger (cannot be handled ("inf")).
                # TODO: Temp fix: clamp it at -1000 and 1000. Maybe other values are more suitable or even better: fix possible bug leading to this issue
                ratio = torch.clamp(logratio.exp(), -1000, 1000)
                # print("ratio", ratio)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                # ratio = r_t(θ)
                # ratio = logratio.exp() = log(PI(...))
                # 
                # Loss1 = E[log(PI(a_t | s_t, theta)) * A_t]
                pg_loss1 = -mb_advantages * ratio
                # Loss2 = clip(r_t(θ), 1 - ϵ, 1 + ϵ) * A_t
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                # .mean() = Expectation E[...]
                # torch.max(..., ...) => run max on negated values: 
                #      min(pg_loss1, pg_loss2) = max(-pg_loss1, pg_loss2)
                #
                # Loss = E[min(pg_loss1, pg_loss2)]
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # b(s_t) is baseline estimation (predicted value function)
                    # b(s_t) = value of critic
                    # ValueLoss = ||b(s_t) - R_t||^2
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                

                entropy_loss = entropy.mean()
                # The entropy which describes how unpredictable the outcome is
                # Maximizing the entropy pushes the policy to behave a bit more randomly to cover a wide range of options
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef


                self.optimizer.zero_grad()
                loss.backward()
                # Optimize actor and critic (= all parameters of agent)
                torch.nn.utils.clip_grad_norm_(self.combined_agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            if self.config.target_kl is not None and approx_kl > self.config.target_kl:
                break