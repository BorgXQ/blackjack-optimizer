## Project Statement

This project aims to close the gap between the house edge and player's expected probability of winning. To do this, we look at the mathematically derived best way to play Blackjack: the basic strategy. This strategy outlines the best actions (out of hitting, standing, splitting, and doubling down) the player should take given their total hand value, whether they have aces, whether they can double, and the dealer's up-card. Then, we train a better model to play more optimally than this strategy. Therefore, the overarchiing hypothesis for this project is as follows:

*There is a better defined strategy than the basic strategy that improves its win rate and average returns.*

## Technical Details

### Blackjack Environment

Basic Blackjack rules apply. Since tables and casinos may have varying rules for Blackjack, it is important that we lay out the rules the simulation in this project follows:
1. The playing deck starts with six card decks (each containing 52 distinct cards) and is reshuffled and reset after two decks' worth of cards are drawn
2. The dealer stands on soft 17
3. The player can double down a hand at any point as long as they have exactly two cards in that hand
4. The player cannot split a hand that has been split previously

In total, without including the restrictions set by the basic strategy model, there are 600 unique combinations of states: `player_sum`, `dealer_visible`, `usable_ace`, `can_split`, and `can_double`.

We consider the environment *Markovian*—meaning the current state contains all necessary information to make optimal decisions without requiring knowledge of previous states. This formulation creates a *Markov Decision Process* where future outcomes depend only on the current state and chosen action, not the sequence of past events that led to the current situation. Intuitively, this is true for Blackjack.

### Trained Agent

To validate the hypothesis, we implement an agent that represents a **model-free, on-policy Monte Carlo (MC) method** for learning optimal Blackjack strategy. The agent implements the **every-visit MC** method for policy evaluation and improvement that updates *Q-values* for every occurrence of a state-action pair within an episode. The fundamental equation being approximated is

$$
Q\left(s,a\right)\approx E\left[G_t|S_t=s,A_t=a\right]
$$

where $G_t$ is the return (i.e., cumulative discounted reward) from time $t$ onward.

As defined in `blackjack_rl_env.py`, the agent uses a 5-tuple state representation (`player_sum`, `dealer_visible`, `usable_ace`, `can_split`, and `can_double`) to capture the relevant information for decision-making. A Q-table stores the Q-values, which are the expected total returns for each state-action combination. As the agent progresses through episodes, these Q-values are continuously refined through temporal backpropagation of returns. This process works by analyzing completed episodes backwards in time, accumulating rewards ($G=G\gamma+r$) to calculate the total return from each decision point. Here, $\gamma=1.0$, meaning no discounting is applied since Blackjack is an episodic game where all rewards within an episode are equally important regardless of timing. These returns are then used to update the Q-values through sample averaging, gradually improving the agent’s understanding of action quality.

For action selection, the agent employs **$\epsilon$-greedy exploration** combined with **action masking**. During training, $\epsilon$-greedy strategically balances exploration (i.e., discovering potentially superior actions) with exploitation of current knowledge embedded in the optimized Q-values. The $\epsilon$ parameter *decays over time*, gradually shifting focus from exploration to exploitation. Meanwhile, action masking ensures the agent only considers valid moves for each state, preventing illegal actions that would result in negative rewards.

<div style="text-align: center;">
    <img src="raw/trained_vs_bs_box.png" alt="Plots for winrate and average reward over 25 runs with 1 mil episodes each" width="600">
</div>

The trained agent, basic strategy model, and a random baseline model are run over 25 stochastic simulations, each with 1 million episodes. The distribution of the averaged winrates and rewards for each of the runs for the trained agent and the basic strategy model are compared above. It is clear that the trained agent plays Blackjack statistically more optimally than the basic strategy model.

### Combined Agent

Although the hypothesis has been broadly validated, we now turn to a deeper analysis of the agents. Out of the 600 possible state combinations, we focus on where the trained agent and the basic strategy model agree on the same action. To explore this, we construct a combined agent that selects the optimal action on a state-by-state basis, recording which model each chosen action originates from.

| **Metric** | **Combined Agent** | **Trained Agent** | **Basic Strat Baseline** | **Random Baseline** |
|-----------|------|------|-------------|---------|
| Winrate | 0.4081 | 0.3987 | 0.3891 | 0.2696 |
| Avg Return | -0.1097 | -0.1265 | -0.1512 | -0.4589 |

The combined agent consistently outperforms all baselines, achieving the highest win rate and the best (least negative) average return. Compared to the trained agent alone, the combined approach yields a 0.94 percentage point gain in win rate and a 13.3% relative improvement in average return. Against the basic strategy model, these improvements grow to 1.90 percentage points and 27.4%, respectively.

<div style="text-align: center;">
    <img src="raw/ca_source_pie.png" alt="Pie plot for comparison between sources in combined agent" width="600">
</div>

The pie chart highlights where these gains come from. Rather than relying solely on the trained agent, the combined approach leverages the strengths of both strategies—selecting the best action for each state. Interestingly, just 2.8% of the combined agent’s actions come from the basic strategy model, yet these contributions are precisely what drive the observed performance boost.

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

If you're simply trying to play Blackjack against a computer, simply run the following:

```bash
python blackjack_game.py
```

### Run Simulation and Analysis

The second part of this project is the simulated Blackjack environment that runs for many episodes while the reinforcement learning model learns the "best" outcomes. The following code generates multiple instances of each full training and evaluation cycles:

```bash
python batch_runner.py --runs 25
```

Running this is necessary to proceed with running the cells in `analysis.ipynb`. However, before that, you must also run the following script, which initializes a CSV file containing all the supposed possible combinations given the constraints put forth in the project scope:

```bash
python total_combinations.py
```

After running the two CLI commands above, you can get run all the cells in `analysis.ipynb`. Aside from analyses, the script aggregates the best action-state combinations between the trained agent and basic strategy, creating a new CSV file with which you can perform optimal Blackjack action predictions with.

### Run Prediction

Once you have run `analysis.ipynb`, you can run predictions using the following format:

```bash
# Ex1: Your cards are [9, 9] and the dealer's up-card is a 8
python prediction.py --hand 9 9 --dealer 8

# Ex2: Your cards are [7, 9] after splitting and the dealer's up-card is a 6
python prediction.py --hand 7 9 --dealer 6 --split
```
After running the first example, the output should be look like the following:

```
------------------------------------------------------------------
     player_sum  dealer_visible  usable_ace  can_split  can_double
497          18               8       False       True        True
------------------------------------------------------------------
Best Action: Split
EV: 0.10632
Source: basic, trained
```

According to the agent, in general, splitting 9s against a dealer's 8 leads to a positive return of 0.10632 per unit bet.

## Results

The trained agent demonstrates a 0.96 percentage point improvement in win rate compared to the basic strategy baseline while achieving a substantially better average return with a 16.3% relative improvement in losses. Both metrics significantly outperform the random baseline, with the trained agent showing 12.91 percentage points higher win rate and 72.4% better average return than random play.



These results demonstrate that while the trained agent successfully learns to outperform traditional heuristics, the greatest gains emerge from intelligently combining learned strategies with established baseline approaches, suggesting that hybrid methodologies can effectively leverage the strengths of both machine learning and conventional strategic frameworks.

<img src="raw/training_progress.png" alt="Plots for rewards, winrate, and epsilon decay over 1 mil episodes" width="600" style="text-align: center;"> <br>

The win rate over the 1,000,000 episodes remained somewhat consistent with a range between 0.3898 and 0.4095. The average reward dramatically improved in the first 200,000 episodes but then stagnated for the remainder of the training period.

## Conclusion

You have practically negative edge in Blackjack even when your strategy is heavily optimized. A risk-averse and risk-neutral person would never prefer to engage in the game. However, if you still choose to play anyway, remember to understand bankroll management and don't gamble what you can't afford to lose.