import os
import pandas as pd

# Define the path to the folder containing the .wav files
database_path = "/root/data/rawboost/unlabeled_data"
output_csv_path = "/root/data/rawboost/unlabeled_data.csv"

# List all .wav files in the directory
wav_files = [f for f in os.listdir(database_path)]

# Create a DataFrame with id and path
df = pd.DataFrame({
    'id': [f[:-4] for f in wav_files],
    'path': ["/root/data/rawboost/unlabeled_data/" + f for f in wav_files]
})

# Save the DataFrame to a CSV file
df.to_csv(output_csv_path, index=False)

df.head()  # Display the first few rows of the DataFrame to ensure it's correct
