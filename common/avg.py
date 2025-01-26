import re

# Path to your .tex file containing the three tables
latex_file = "common/tables.txt"

# Dictionary to store all numeric values by augmentation
augmentations = {
    "SpecAugment": {"acc": [], "map": [], "f1": []},
    "SpecMix": {"acc": [], "map": [], "f1": []},
    "DiffRes": {"acc": [], "map": [], "f1": []},
    "FrameMixer": {"acc": [], "map": [], "f1": []},
}

# Regex to match floats, possibly wrapped in \textcolor{...}{...}
# e.g., \textcolor{blue}{54.25} or 54.25
float_pattern = re.compile(r'(?:\\textcolor\{[^}]*\}\{)?(\d+\.\d+)(?:\})?')

with open(latex_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        
        # Determine which augmentation this line corresponds to
        if "SpecAugment\\cite{Park2019SpecAugmentAS}" in line or "& SpecAugment" in line:
            aug_key = "SpecAugment"
        elif "SpecMix\\cite{Kim2021SpecMixA}" in line or "& SpecMix" in line:
            aug_key = "SpecMix"
        elif "DiffRes\\cite{Liu2022LearningTR}" in line or "& DiffRes" in line:
            aug_key = "DiffRes"
        elif "\\textbf{FrameMixer (ours)}" in line or "& FrameMixer (ours)" in line:
            aug_key = "FrameMixer"
        else:
            # Not one of the four augmentations we care about
            continue
        
        # Extract floating-point values (Accuracy, mAP, F1 Score)
        matches = float_pattern.findall(line)
        
        # We expect exactly 3 values in each relevant line: Accuracy, mAP, F1
        if len(matches) >= 3:
            acc_val = float(matches[0])
            map_val = float(matches[1])
            f1_val  = float(matches[2])
            
            augmentations[aug_key]["acc"].append(acc_val)
            augmentations[aug_key]["map"].append(map_val)
            augmentations[aug_key]["f1"].append(f1_val)

# Now compute and print the average for each augmentation
for aug in augmentations:
    acc_list = augmentations[aug]["acc"]
    map_list = augmentations[aug]["map"]
    f1_list  = augmentations[aug]["f1"]

    if len(acc_list) == 0:
        # No data found for this augmentation
        print(f"{aug}: No data found.")
        continue
    
    avg_acc = sum(acc_list) / len(acc_list)
    avg_map = sum(map_list) / len(map_list)
    avg_f1  = sum(f1_list)  / len(f1_list)

    print(f"{aug} Averages across all tables:")
    print(f"  Accuracy = {avg_acc:.2f}")
    print(f"  mAP      = {avg_map:.2f}")
    print(f"  F1 Score = {avg_f1:.2f}\n")
