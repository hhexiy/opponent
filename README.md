# Opponent Modeling in Deep Reinforcement Learning
Code for the paper "Opponent Modeling in Deep Reinforcement Learning" published in ICML 2016. 
The main goal is to **learn adpative strategies against different opponents** in the deep reinforcement learning framework (Deep Q-Network in particular).
Currently it's tested only on Linux with CPU.

## Dependencies
- [Torch](https://github.com/torch). See installation instructions [here](http://torch.ch/docs/getting-started.html).
- [Glove](http://nlp.stanford.edu/projects/glove/) word vectors. Can also be downloaded by `make dat/glove/glove.840B.300d.txt`.

## Data
Please email hhe@umiacs.umd.edu for the quiz bowl dataset with human buzzes.

## Experiments
Please look at the targets `run_qb` and `run_soccer` in the `Makefile`.
To run the quiz bowl experiments, first we need to train a content model (produce the answers) on a separate dataset. See `train_content` in `Makefile`. The models will be written to `checkpoint_dir` and you want to change it to your path.

## TODO
- Currently some targets in the `Makefile` is more like "notes" and the dependencies need to be fixed.
- Test on GPUs.
