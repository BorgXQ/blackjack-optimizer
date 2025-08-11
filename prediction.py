"""
Blackjack State Evaluator
-------------------------
Takes a player's hand, the dealer's visible card, and split status, then looks up the
best action, expected value (EV), and source from a merged results CSV file.
"""

import numpy as np
import pandas as pd
import argparse
from typing import List, Tuple


def state_representation(df: pd.DataFrame, player_values: List[int], dealer_visible: int, have_split: bool) -> Tuple[str, str, str]:
    for i in player_values:
        if i > 11:
            print("----------------------------")
            print("Card values cannot exceed 11")
            print("----------------------------")
            return ("NO_DATA", "-999", "NO_DATA")
        elif i < 1:
            print("---------------------------------")
            print("Card values cannot be less than 1")
            print("---------------------------------")
            return ("NO_DATA", "-999", "NO_DATA")
        
    player_sum = np.sum(player_values)
    usable_ace = False
    can_double = False
    can_split = False

    for i in player_values:
        if i == 1 or i == 11:
            usable_ace = True
            player_sum += 10 if player_sum + 10 <= 21 else 0

    if len(player_values) == 2:
        can_double = True
        if not have_split:
            if player_values[0] == player_values[1]:
                can_split = True
            elif (player_values[0] == 1 or player_values[0] == 11) and (player_values[1] == 1 or player_values[1] == 11):
                can_split = True

    matched_row = df[
        (df["player_sum"] == player_sum) &
        (df["dealer_visible"] == dealer_visible) &
        (df["usable_ace"] == usable_ace) &
        (df["can_split"] == can_split) &
        (df["can_double"] == can_double)
    ]
    
    if not matched_row.empty:
        print("------------------------------------------------------------------")
        print(f'{matched_row[["player_sum", "dealer_visible", "usable_ace", "can_split", "can_double"]]}')
        print("------------------------------------------------------------------")
        return matched_row[["best_action", "ev", "source"]].values[0]
    else:
        print("--------------------------------")
        print("State combination does not exist")
        print("--------------------------------")
        return ("NO_DATA", "-999", "NO_DATA")
    

def main():
    parser = argparse.ArgumentParser(description="Evaluate Blackjack hand state and return best action.")
    parser.add_argument("--csv", type=str, default="merged_results.csv", help="Path to merged results CSV.")
    parser.add_argument("--hand", type=int, nargs="+", required=True, help="Your hand card values, e.g., --hand 1 2 9")
    parser.add_argument("--dealer", type=int, required=True, help="Dealer's visible card value.")
    parser.add_argument("--split", action="store_true", help="Flag indicating you have already split.")

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    action, ev, source = state_representation(df, args.hand, args.dealer, args.split)
    print(f"Best Action: {action}\nEV: {ev}\nSource: {source}\n")


if __name__ == "__main__":
    main()
