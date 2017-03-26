# A3C
Deep reinforcement learning using an asynchronous advantage actor-critic (A3C) model written in [TensorFlow](https://www.tensorflow.org/). 

This AI does not rely on hand-engineered rules or features. Instead, it masters the environment by looking at raw pixels and learning from experience, just as humans do.

## Dependencies
* NumPy
* OpenAI Gym 0.8
* Pillow
* SciPy
* TensorFlow 1.0

## Learning Environment
Uses environments provided by [OpenAI Gym](https://gym.openai.com/).

## Preprocessing
Each frame is transformed into a 47×47 grayscale image with 32-bit float values between 0 and 1. No image cropping is performed. Reward signals are restricted to -1, 0 and 1.

## Network Architecture
The input layer consists of a 47×47 grayscale image.

Four convolutional layers follow, each with 32 filters of size 3×3 and stride 2 and each applying the rectifier nonlinearity.

A recurrent layer follows, consisting of 256 LSTM units.

Lastly, the network diverges into two output layers – one is a probability distribution over actions (represented as logits), the other is a single linear output representing the value function.

## Acknowledgements
Implementation inspired by the [OpenAI Universe](https://universe.openai.com/) reference agent.

Heavily influenced by DeepMind's seminal paper ['Asynchronous Methods for Deep Reinforcement Learning' (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783).
