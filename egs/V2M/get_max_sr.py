import json

def get_max_sample_rate(jsonl_path):
    max_rate = 0
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'sample_rate' in data:
                    max_rate = max(max_rate, data['sample_rate'])
            except json.JSONDecodeError:
                continue  # Skip malformed lines
    return max_rate

# Example usage
path_to_file = "/work/users/t/i/tis/VidMuse/egs/V2M20K/data.jsonl"
print("Max sample rate:", get_max_sample_rate(path_to_file))