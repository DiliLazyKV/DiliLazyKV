import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the JSON file
# IMPORTANT: Replace with the correct path to your JSON file.
# For example, if it\'s in the same directory as the script:
# json_file_path = 'Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json'
# Or if it\'s in the Important_Head/head_score/ directory relative to your workspace root:
# json_file_path = 'Important_Head/head_score/Meta-Llama-3-8B-Instruct_retrieval_reasoning_heads.json'
# Or the path from your log:
json_file_path = '/content/head_infscore.json' # Using the path from your log

print(f"Attempting to load JSON from: {json_file_path}")

try:
    with open(json_file_path, 'r') as f:
        data_from_file = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file_path}")
    print("Please ensure the path is correct and the file exists in the CWD or via the absolute/relative path provided.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_file_path}")
    exit()

print(f"Successfully loaded JSON. Top-level keys found: {list(data_from_file.keys())[:5]}...")

# --- MODIFICATION START ---
# Get the actual dictionary of scores, which is a value of one of the top-level keys
actual_scores_dict = data_from_file.get('average_infscores_weighted')

if not isinstance(actual_scores_dict, dict):
    print(f"Error: Key 'average_infscores_weighted' not found in JSON or is not a dictionary.")
    print(f"Available top-level keys: {list(data_from_file.keys())}")
    exit()
# --- MODIFICATION END ---


infscores_processed = {} # Renamed to avoid confusion with previous user script's 'infscores' variable name
max_layer = -1
max_head = -1

# --- MODIFICATION START: Iterate over the correct dictionary ---
for key, score_value in actual_scores_dict.items():
    try:
        layer_str, head_str = key.split('-')
        layer = int(layer_str)
        head = int(head_str)

        if not isinstance(score_value, (int, float)):
            print(f"Warning: Key {key} has non-numeric value {score_value}. Skipping.")
            continue

        infscores_processed[key] = float(score_value)
        max_layer = max(max_layer, layer)
        max_head = max(max_head, head)
    except ValueError:
        print(f"Warning: Key {key} is not in 'layer-head' format or score is not a valid number. Skipping.")
        continue
    except Exception as e:
        print(f"Warning: Error processing key {key} with value {score_value}: {e}. Skipping.")
        continue
# --- MODIFICATION END ---

print(f"Processed infscores_processed dictionary. Number of entries: {len(infscores_processed)}")
if len(infscores_processed) < 5:
    print(f"Sample processed entries: {list(infscores_processed.items())[:5]}")


if max_layer == -1 or max_head == -1 or not infscores_processed:
    print("Error: Could not determine valid layer/head dimensions or no scores were processed.")
    print(f"max_layer: {max_layer}, max_head: {max_head}, number of scores: {len(infscores_processed)}")
    exit()

print(f"Determined max_layer: {max_layer}, max_head: {max_head}")

# Create a matrix to store InfScores
# Matrix dimensions: (num_heads, num_layers)
infscore_matrix = np.full((max_head + 1, max_layer + 1), np.nan) # Initialize with NaN for missing values

# Fill the matrix with InfScore values
filled_entries = 0
for key, value in infscores_processed.items():
    try:
        layer_str, head_str = key.split('-')
        layer = int(layer_str)
        head = int(head_str)
        infscore_matrix[head, layer] = value
        filled_entries += 1
    except Exception as e:
        print(f"Error placing score for {key} in matrix: {e}")
print(f"Filled {filled_entries} entries into infscore_matrix.")

if filled_entries == 0:
    print("Error: No entries were successfully placed in the heatmap matrix.")
    exit()

# Diagnostic: Print some info about the matrix
print(f"infscore_matrix shape: {infscore_matrix.shape}")
print(f"Number of NaNs in matrix: {np.isnan(infscore_matrix).sum()}")
print(f"Min score in matrix (ignoring NaNs): {np.nanmin(infscore_matrix) if not np.all(np.isnan(infscore_matrix)) else 'All NaNs'}")
print(f"Max score in matrix (ignoring NaNs): {np.nanmax(infscore_matrix) if not np.all(np.isnan(infscore_matrix)) else 'All NaNs'}")
print(f"Mean score in matrix (ignoring NaNs): {np.nanmean(infscore_matrix) if not np.all(np.isnan(infscore_matrix)) else 'All NaNs'}")


# Create the heatmap
plt.figure(figsize=(12, 7)) # Adjusted figsize slightly

# If the plot is still monochrome, you might need to manually set vmin/vmax
# based on the printed min/max values if they are very close.
sns.heatmap(infscore_matrix, cmap="viridis", annot=False, cbar_kws={'label': 'InfScore'}) # Added cbar label
plt.title(f'InfScore Heatmap from {json_file_path}')
plt.xlabel('Layer Index')
plt.ylabel('Head Index')
plt.tight_layout()
plt.show()
print("Script finished. If plot is not displayed, ensure your environment supports GUI popups or save the plot to a file.")

# --- Code to sum raw scores for Layer 0, 1, 2 for verification ---
# (Assuming infscores_processed is populated correctly)
raw_layer_sums = {}
for i in range(min(3, max_layer + 1)): # Check layers 0, 1, 2 if they exist
    current_layer_sum = 0
    count = 0
    for head_idx in range(max_head + 1):
        key = f"{i}-{head_idx}"
        if key in infscores_processed:
            current_layer_sum += infscores_processed[key]
            count +=1
    if count > 0:
        raw_layer_sums[f'Layer {i}'] = current_layer_sum
    else:
        raw_layer_sums[f'Layer {i}'] = 'No heads found or all scores zero'

print("--- Raw Layer Sums (from script's processing of JSON) ---")
for layer_name, total_score in raw_layer_sums.items():
    print(f"{layer_name} - Total Original Score: {total_score}")

