import csv

# Range of dealer_visible values
dealer_visible_values = list(range(1, 11))

# Predefined valid (player_sum, usable_ace, can_split, can_double) combinations
# Remember that ace values are ALWAYS 11 unless player_sum busts if 11 then 1
combinations = []

# player_sum = 4
combinations.append((4, False, True, True))

# player_sum = 5
combinations.append((5, False, False, True))

# Even player_sum from 6 to 10 (3 combinations each)
for s in range(6, 11, 2):
    combinations.append((s, False, True, True))   # Same value pair
    combinations.append((s, False, False, True))  # Different 2-card no ace
    combinations.append((s, False, False, False)) # >2-card no ace

# Odd player_sum from 7 to 11 (2 combinations each)
for s in range(7, 12, 2):
    combinations.append((s, False, False, True))  # 2-card no ace
    combinations.append((s, False, False, False)) # >2-card no ace

# Even player_sum from 12 to 20 (5 combinations each)
for s in range(12, 21, 2):
    combinations.append((s, False, True, True))   # Same value pair
    combinations.append((s, False, False, True))  # Different 2-card no ace
    combinations.append((s, True, False, True))   # 2-card with ace
    combinations.append((s, True, False, False))  # >2-card with ace
    combinations.append((s, False, False, False)) # >2-card no ace

# Odd player_sum from 13 to 19 (4 combinations each)
for s in range(13, 20, 2):
    combinations.append((s, True, False, True))   # 2-card with ace
    combinations.append((s, False, False, True))  # 2-card no ace
    combinations.append((s, True, False, False))  # >2-card with ace
    combinations.append((s, False, False, False)) # >2-card no ace

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
