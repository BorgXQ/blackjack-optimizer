import csv

# Range of dealer_visible values
dealer_visible_values = list(range(1, 11))

# Predefined valid (player_sum, usable_ace, can_split, can_double) combinations
combinations = []

# player_sum = 2
combinations.append((2, True, True, True))

# player_sum = 3
combinations.append((3, True, False, True))
combinations.append((3, False, False, False))

# player_sum = 4
combinations.append((4, False, True, True))
combinations.append((4, True, False, True))
combinations.append((4, False, False, False))

# player_sum = 5
combinations.append((5, True, False, True))
combinations.append((5, False, False, True))
combinations.append((5, False, False, False))

# Even player_sum from 6 to 20 (5 combinations each)
for s in range(6, 21, 2):
    combinations.append((s, False, True, True))   # Same value pair
    combinations.append((s, False, False, True))  # Different 2-card no ace
    combinations.append((s, True, False, True))   # 2-card with ace
    combinations.append((s, True, False, False))  # >2-card with ace
    combinations.append((s, False, False, False)) # >2-card no ace

# Odd player_sum from 7 to 19 (4 combinations each)
for s in range(7, 20, 2):
    combinations.append((s, True, False, True))   # 2-card with ace
    combinations.append((s, False, False, True))  # 2-card no ace
    combinations.append((s, True, False, False))  # >2-card with ace
    combinations.append((s, False, False, False)) # >2-card no ace

# player_sum = 21
combinations.append((21, True, False, True))
combinations.append((21, True, False, False))
combinations.append((21, False, False, False))

# Expand each combination with all 10 dealer_visible values
final_rows = []
for combo in combinations:
    for dv in dealer_visible_values:
        player_sum, usable_ace, can_split, can_double = combo
        final_rows.append({
            'player_sum': player_sum,
            'dealer_visible': dv,
            'usable_ace': usable_ace,
            'can_split': can_split,
            'can_double': can_double
        })

# Write to CSV
csv_filename = "./csv/blackjack_states.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['player_sum', 'dealer_visible', 'usable_ace', 'can_split', 'can_double'])
    writer.writeheader()
    writer.writerows(final_rows)

print(f"CSV file '{csv_filename}' created with {len(final_rows)} rows.")
