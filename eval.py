
import os
import glob
import numpy as np
import torch
from eval_data_loader import load_txt_as_data_eval
from Gattnetconv import GATTNetConvHybrid

# ----- Configuration Variables -----
input_folder = "eval_filestxt"         
output_folder = "PC_eval"  
checkpoint_path = "evalgatnet/model_best.pth"  # Path to the saved model checkpoint

graph_type = "fixed"    # "fixed" for KNN, or "dynamic" for radius-based graph
k_value = 16            # Number of neighbors for KNN
radius_value = 0.16     # Radius for dynamic graph 

features_str = "ALL"    # all features

# Create output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model with in_channels=13 
model = GATTNetConvHybrid(
    in_channels=13,
    hidden_channels=64,
    out_channels=3,
    k=k_value,
    dropout=0.2
).to(device)

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()
print("Model loaded and set to evaluation mode.")

# Get list of all TXT files in the input folder
txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
txt_files.sort()
print("Number of test files found:", len(txt_files))

if len(txt_files) == 0:
    print("No test files found. Please check the folder path and file extensions.")
    exit()

# Process each TXT file
for file in txt_files:
    print("Evaluating file:", file)
    
    # Load the data using the evaluation-specific data loader
    
    data = load_txt_as_data_eval(file, k=k_value, features=features_str,
                             graph_type=graph_type, radius=radius_value,
                             max_points=1024)

    if data.x is None or data.x.size(0) == 0:
        print("Warning: No points loaded from", file)
        continue
    print("Data.x shape:", data.x.shape)
    data = data.to(device)
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1).cpu().numpy()
    
    # Load the original TXT file data
    original_data = np.loadtxt(file)
    if original_data.ndim == 1:
        original_data = original_data.reshape(1, -1)
    
    # Append the predictions as a new column
    evaluated_data = np.hstack((original_data, preds.reshape(-1, 1)))
    
    # Save the evaluated data to a new TXT file 
    base_name = os.path.basename(file)
    output_file = os.path.join(output_folder, base_name)
    np.savetxt(output_file, evaluated_data, fmt="%.6f")
    print("Saved evaluated file to:", output_file)
