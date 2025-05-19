import json
import os

def transform_json_data(file_path):
    """
    Loads a JSON file, wraps the numerical scores under 'average_infscores_weighted'
    into lists, and saves the modified data back to the file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    if 'average_recalls_weighted' not in data or not isinstance(data['average_recalls_weighted'], dict):
        print("Error: 'average_recalls_weighted' key not found or its value is not a dictionary.")
        return

    inner_dict = data['average_recalls_weighted']
    transformed_inner_dict = {}

    for key, value in inner_dict.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool): # Check if it's a number but not a boolean
            transformed_inner_dict[key] = [value]
        elif isinstance(value, list): # If it's already a list, keep it as is
             transformed_inner_dict[key] = value
        else: # For other types, keep as is, or decide on a specific handling
            print(f"Warning: Value for key '{key}' is of type {type(value)}, not a number. Keeping original value.")
            transformed_inner_dict[key] = value


    data['average_recalls_weighted'] = transformed_inner_dict

    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully transformed and saved data to {file_path}")
    except IOError:
        print(f"Error: Could not write to file {file_path}")

if __name__ == '__main__':
    # Assuming the script is in the workspace root, and the target file is relative to that
    # Adjust the path if the script is located elsewhere or the target file path is different
    target_file = "/home/casl/KVC/HeadKV/Important_Head/results/meta-llama_Meta-Llama-3-8B-Instruct/Copy-Paste/head_recall.json"
    
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path for the target file
    absolute_target_file_path = os.path.join(script_dir, target_file)
    
    # Normalize the path to resolve any ".." or "." components and ensure correct separators
    normalized_target_file_path = os.path.normpath(absolute_target_file_path)

    transform_json_data(normalized_target_file_path) 