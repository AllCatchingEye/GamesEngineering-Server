# Controllers

In this directory are all controllers which are representing some player. This player can be either a real human 
or some ai. 

## AI Controller

In case of ai controller, there are two mayer types: 

1. RL trained AIs
2. Handcrafted AI

For RL trained AIs there are multiple versions of controller, each using some different version/algorithm. The actual Algorithm is represented by a so called agent or agent-trainer (see also [`select_card`](../ai/select_card/) and [`select_game_type`](../ai/select_game_type/) directories). The agent's and agent-trainer's logic (e.g. RL-Training) is mainly based on the events received and redirected by the Controller. To structure/order actions based on events, the AI based Controller invoke for one event three event listeners: 

1. `on_pre_game_event` (e.g. for initialization)
2. `on_game_event` (e.g. for actual event processing)
3. `on_post_game_event` (e.g. for cleanup)
