import json
import re

# Input and output files
INPUT_FILE = "results.json"
OUTPUT_FILE = "converted.json"


def clean_text(text):
    """Remove '(Screenshot from ...)' and 'We rate ...' sentences; handle None values"""
    if not isinstance(text, str):  # Ensure text is a string
        return ""
    text = re.sub(r'\(Screenshot from .*?\)', '', text).strip()  # Remove screenshot notes
    text = re.sub(r'We rate .*?$', '', text).strip()  # Remove sentences starting with "We rate ..."
    return text

def convert_data(input_file, output_file):
    # Read the JSON file
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted_data = []
    for item in data:
        # Ensure both 'title' and 'claims' fields exist
        if "title" not in item or "claims" not in item:
            continue  # Skip invalid entries

        # Clean 'title' and 'claims'
        cleaned_title = clean_text(item.get("title", ""))
        cleaned_claims = [clean_text(claim) for claim in item.get("claims", [])]

        # Skip entries with an empty title
        if not cleaned_title:
            continue

        # Construct new text in instruction format
        inst_text = f"[INST]Please evaluate the following claim {cleaned_title}[/INST]  "
        claims_text = " ".join(cleaned_claims)
        new_text = inst_text + claims_text  # Concatenate

        # Add to the new dataset
        converted_data.append({"text": new_text})

    # Save the converted JSON data
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"Conversion complete! New data saved to {output_file}")

# Run the conversion
if __name__ == "__main__":
    convert_data(INPUT_FILE, OUTPUT_FILE)
