import json

# Define multiple file paths
file_paths = [
    "./FactCheck/covid-misconceptions_data.json", 
    "./FactCheck/fakenews_data.json",
    "./FactCheck/scicheck_data.json",
    "./FactCheck/LIAR-RAW.json",
    "./FactCheck/FactCheck&LIAR-RAW.json",
    "./PolitFact/converted.json",
    "./merged_all.json"
]

# Loop through each file path
for file_path in file_paths:
    try:
        # Open and load the JSON file
        with open(file_path, "r", encoding="utf-8") as file:
            dataset = json.load(file)

        # Check if the dataset is a list
        if isinstance(dataset, list):
            sample_count = len(dataset)
        else:
            print(f"The structure of {file_path} is not a list and may need additional handling.")
            continue

        # Print the number of samples in the dataset
        print(f"{file_path} contains {sample_count} samples.")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found. Please check the file path.")
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An error occurred while reading file {file_path}: {e}")
