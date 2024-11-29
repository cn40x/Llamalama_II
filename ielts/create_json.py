import json
import os

def parse_text_to_jsonl(input_folder, output_file):
    with open(output_file, 'w') as jsonl_file:
        # Iterate over each .txt file in the input folder
        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                
                with open(file_path, 'r') as file:
                    lines = file.readlines()
                
                data = {
                    "text": "",
                    "questions": [],
                    "choices": [],
                    "answers": []
                }
                
                section = "text"
                for line in lines:
                    line = line.strip()
                    
                    if line == "#questions":
                        section = "questions"
                    elif line == "#choices":
                        section = "choices"
                    elif line == "#answers":
                        section = "answers"
                    else:
                        if section == "text":
                            data["text"] += line + " "
                        elif section == "questions":
                            data["questions"].append(line)
                        elif section == "choices":
                            data["choices"].append(line)
                        elif section == "answers":
                            question_id, answer = line.split()
                            data["answers"].append({"question_id": question_id, "answer": answer})

                # Write parsed data as a JSON object in JSONL format
                jsonl_file.write(json.dumps(data) + "\n")

# Usage:
# Specify the folder containing 1.txt, 2.txt, etc. and the output .jsonl file
parse_text_to_jsonl("/Users/kunkerdthaisong/Llamalama_II/ielts/datasets/IELST_example/IELST_reading/", "output_1.jsonl")
