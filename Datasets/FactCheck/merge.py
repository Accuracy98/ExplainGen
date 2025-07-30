import json

def merge_json_files(json_files):
    """Merge multiple JSON files"""
    merged_data = []
    
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    print(f"Warning: {file} does not contain a list of JSON objects, skipping.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return merged_data

def convert_and_merge_datasets(dataset1_path, dataset2_data):
    """Convert and merge two datasets"""
    # Read the first dataset
    with open(dataset1_path, 'r', encoding='utf-8') as f:
        dataset1 = json.load(f)

    # Convert the format of the second dataset
    converted_dataset2 = []
    for item in dataset2_data:
        if 'title' in item and 'full_story' in item:
            converted_item = {
                "text": f"[INST]Please evaluate the following claim \"{item['title']}\". [/INST] {item['full_story']}"
            }
            converted_dataset2.append(converted_item)

    # Return the merged data
    return dataset1 + converted_dataset2

def clean_text_field(data):
    """Clean the content of the 'text' field"""
    for entry in data:
        if 'text' in entry and isinstance(entry['text'], str):
            entry['text'] = entry['text'].replace("Therefore, the claim is true", "").replace("Therefore, the claim is false", "").strip()
    return data

if __name__ == "__main__":
    # Step 1: Merge multiple JSON files
    json_files = ['covid-misconceptions_data.json', 'scicheck_data.json', 'fakenews_data.json']
    merged_factcheck_data = merge_json_files(json_files)

    # Step 2: Convert and merge the LIAR-RAW dataset
    liar_raw_path = 'LIAR-RAW.json'
    merged_all_data = convert_and_merge_datasets(liar_raw_path, merged_factcheck_data)

    # Step 3: Clean the 'text' field
    final_data = clean_text_field(merged_all_data)

    # Save final result
    output_file = 'FactCheck&LIAR-RAW.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"Processing completed. Final dataset saved to {output_file}")
