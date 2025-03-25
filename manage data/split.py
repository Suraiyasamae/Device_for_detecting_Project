import os
import pandas as pd
import shutil

def split_csv_files(input_root_dir, output_root_dir):
    # Create output root directory if it doesn't exist
    os.makedirs(output_root_dir, exist_ok=True)

    # Walk through the input directory
    for root, dirs, files in os.walk(input_root_dir):
        # Skip if no files
        if not files:
            continue

        # Determine relative path from input root
        relative_path = os.path.relpath(root, input_root_dir)
        
        # Create corresponding output directory
        output_dir = os.path.join(output_root_dir, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        # Process CSV files
        for file in files:
            if file.endswith('.csv'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, file)

                # Read the CSV file
                df = pd.read_csv(input_file_path)

                # Take first 20 rows plus header (total 21 rows)
                df_truncated = df.head(20)

                # Save the truncated DataFrame
                df_truncated.to_csv(output_file_path, index=False)

                print(f"Processed: {input_file_path} -> {output_file_path}")

# Set input and output directories
input_root = 'convert_Type_abnormal_data'
output_root = 'split_abnormal_data'

# Run the CSV splitting process
split_csv_files(input_root, output_root)