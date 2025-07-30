import json

def convert_and_merge_datasets(dataset1_path, dataset2_path, output_path):
    # Load the first dataset
    with open(dataset1_path, 'r', encoding='utf-8') as f:
        dataset1 = json.load(f)

    # Load the second dataset
    with open(dataset2_path, 'r', encoding='utf-8') as f:
        dataset2 = json.load(f)

    # Merge the datasets
    merged_dataset = dataset1 + dataset2

    # Save the merged dataset
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_dataset, f, ensure_ascii=False, indent=4)

    print(f"Merged dataset has been saved to {output_path}")

dataset1_path = './FactCheck/FactCheck&LIAR-RAW.json'
dataset2_path = './PolitFact/converted.json'
output_path = 'merged_all.json'

convert_and_merge_datasets(dataset1_path, dataset2_path, output_path)
