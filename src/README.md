# Entry Points

In this directory are a lot of entry points for running the game, performing RL training etc.

## RL Training

In case of RL training, different training scripts are created for different purposes. With some scripts different 
learing rates are tested, with other different net configurations.

Each training script calls start_training in `drl_training_base.py` and provides some configuration settings how to train.