## Project Statement

Blackjack is one of the very few casino games I actually knew the rules of. Basically, you try to beat the dealer through a series of "hitting" and "standing" actions, doing your best to obtain a total higher than the dealer's but not over 21 (this is a bust). The casino always has the winning edge, but this project aims to see how far we, as the players, can get to even the playfield a little.

Sutton and Barto's book on reinforcement learning was my main source of inspiration to build this model, specifically all the logic behind it. Essentially, I assume several conditions, simulate the blackjack environment, and see how much better the trained model performs than a random baseline.

## Installation

First, clone the repository:

```bash
git clone git@github.com:BorgXQ/blackjack-optimizer.git
cd blackjack-optimizer
```
Second, depending on your operating system, activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows, use: .venv\Scripts\activate
```

Finally, install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Play Blackjack

If you're simply trying to play blackjack against a computer, simply run the following:

```bash
python blackjack_env.py
```

### Run Simulation

The bread and butter of this project is the simulated blackjack environment that runs for many episodes while the reinforcement learning model learns the "best" outcomes. To execute this, simply run the following:

```bash
python run_simulation.py
```

## Results

| **Metric** | **Trained Agent** | **Random Baseline** | **Improvement (%)** |
|-----------|------|-------------|---------|
| **Winrate** | 0.4218 | 0.2794 | 50.94 |
| **Avg Return** | -0.0669 | -0.4407 | 84.83 |

## Future Work

The rest of the project is under construction. Eventually, the project will simulate episodes of blackjack games! All you need to do in the end is enter the game, then cheat and ask the model which action to take, and drown in wins (probably still not statistically possible). Stay tuned!
