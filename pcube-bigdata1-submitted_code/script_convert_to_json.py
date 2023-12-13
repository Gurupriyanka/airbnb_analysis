import os
import pandas as pd
import zipfile
import sys

# Directory containing CSV files
csv_folder = sys.argv[1]

# Output folder for JSON and ZIP files (relative to the script's directory)
output_folder_relative = sys.argv[2]

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Form the absolute path for the output folder
output_folder = os.path.join(script_dir, output_folder_relative)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate through each CSV file in the folder
for file_name in os.listdir(csv_folder):
    
    if file_name.endswith(".csv.gz"):
        print(file_name)
        # Extract the city name from the filename (assuming filenames are in the format "city_data.csv")
        city_name = file_name.split('.')[0]  # Adjust this based on your filename pattern

        # Form the paths for CSV and JSON files
        csv_path = os.path.join(csv_folder, file_name)
        json_output_path = os.path.join(output_folder, f"{file_name.split('.')[0]}.json")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_path)

        # Add a new column 'city' with the filename (without extension) as the value
        df['city'] = city_name

        # Convert DataFrame to JSON
        df.to_json(json_output_path, orient="records", lines=True, date_format="iso", double_precision=15)

# Create a zip file containing all JSON files
with zipfile.ZipFile(os.path.join(output_folder, "output.zip"), 'w') as zipf:
    for file_name in os.listdir(output_folder):
        if file_name.endswith(".json"):
            json_file_path = os.path.join(output_folder, file_name)
            zipf.write(json_file_path, os.path.basename(json_file_path))

# Cleanup: Remove individual JSON files after zipping
for file_name in os.listdir(output_folder):
    if file_name.endswith(".json"):
        json_file_path = os.path.join(output_folder, file_name)
        os.remove(json_file_path)

print("Conversion and zipping completed.")
