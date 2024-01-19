# Select Card

This directory contains all components to make a decision which card to play next during a game. Basically, there are 
agents and agent trainer. The agent is used during the final game to represent some kind of ai. The agent trainer trains 
the ai. Under the hood, the agent and the agent trainer use the a model. The model is the neuronal network, where 
different types of models exist. Some just differ by the number of layers or neurons, other are used in different kinds
of RL algorithms like PPO or Deep-Q-Learning, where different requirements need to be met.

The agents and agent trainer use inheritence to share functionality. The `rl_agent` and `rl_agent_trainer` build the 
foundation.

In the directory `models` are the models mentioned above. 
