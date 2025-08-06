## Project Statement

Blackjack is one of the very few casino games I actually knew the rules of. Basically, you try to beat the dealer through a series of "hitting" and "standing" actions, doing your best to obtain a total higher than the dealer's but not over 21 (this is a bust). The casino always has the winning edge, but this project aims to see how far we, as the players, can get to even the playfield a little.

Sutton and Barto's book on reinforcement learning was my main source of inspiration to build this model, specifically all the logic behind it. Essentially, I assume several conditions, simulate the blackjack environment, and see how much better the trained model performs compared to the mathematically derived basic strategy and a random baseline.

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
python blackjack_game.py
```

### Run Simulation

The bread and butter of this project is the simulated blackjack environment that runs for many episodes while the reinforcement learning model learns the "best" outcomes. To execute this, simply run the following:

```bash
python run_simulation.py
```

## Results

| **Metric** | **Trained Agent** | **Basic Strat Baseline** | **Random Baseline** |
|-----------|------|-------------|---------|
| Winrate | 0.4185 | 0.4276 | 0.2874 |
| Avg Return | -0.0711 | -0.0555 | -0.4209 |

The trained agent observes a 45.6% improvement compared to the random baseline, but not as much as the 48.8% improvement by implementing the basic strategy in terms of win rate. For average rewards, the basic strategy still somewhat beats the trained agent, reaching an 86.8% improvement compared to the random baseline, with the trained model sitting at an 83.1% improvement.

<img src="raw/training_progress.png" alt="Plots for rewards, winrate, and epsilon decay over 1 mil episodes" width="600" style="text-align: center;"> <br>

The win rate over the 1,000,000 episodes remain somewhat consistent, floating in between 41.0% and 43.0%, though not converging to a specific value. The average reward dramatically improves in the first 50,000 episodes, slowly climbs over the next 350,000 episodes, but then stagnates during the rest of the training period.

## Discussion

The discussion portion of the project is under construction. Stay tuned!

## Conclusion

You have practically near-zero edge in blackjack even when your strategy is heavily optimized. But if you're going to anyway, here's a motivation:

[St. Petersburg paradox](https://en.wikipedia.org/wiki/St._Petersburg_paradox) says that expected values in these games should be infinite, but in actuality, it isn't because it accounts for extremely rare events (like *absurdly impossible* events). But who's to say you're not going to be that one in a quintillion gambler who wins a thousand times in a row? After all, highly improbable ≠ impossible.