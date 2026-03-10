import json
import os

json_path = 'benchmarks/results.json'
output_path = 'LEADERBOARD.md'

if not os.path.exists(json_path):
    print(f"❌ Error: {json_path} not found!")
else:
    with open(json_path, 'r') as f:
        data = json.load(f)

    markdown_content = "# 🏆 Munajjam Performance Leaderboard\n\n"
    markdown_content += "This leaderboard is automatically generated from the benchmark results.\n\n"
    markdown_content += "| Model Name | Accuracy | Inference Speed | Test Date |\n"
    markdown_content += "| :--- | :--- | :--- | :--- |\n"

    for entry in data:
        markdown_content += f"| **{entry['model']}** | {entry['accuracy']}% | {entry['speed']} | {entry['date']} |\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"✅ Success! {output_path} has been generated.")