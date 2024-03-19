import os
import requests
import zipfile
import shutil

dirs = ['data', 'data/raw', 'data/processed/train', 'data/processed/val', 'data/processed/test']
for dir in dirs:
  os.makedirs(dir, exist_ok=True)

# URL of the file to download
url = 'http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip'

# Directory to save the file
save_dir = 'data/raw'

# File name to save as
file_name = os.path.join(save_dir, 'MachineLearningCVE.zip')

# Download the file
print("Loading...")
response = requests.get(url)
with open(file_name, 'wb') as f:
    f.write(response.content)

print(f"File saved to: {file_name}")
print("Unziping...")

# Check if the downloaded file is a valid ZIP file
try:
  with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(save_dir)
  print(f"File unzipped to: {save_dir}")

  os.remove(file_name)

  # Move files up one level
  extracted_files = os.listdir(os.path.join('data/raw', 'MachineLearningCVE'))
  for file in extracted_files:
    src = os.path.join(save_dir, 'MachineLearningCVE', file)
    dst = os.path.join(save_dir, file)
    shutil.move(src, dst)
  print("Moved extracted files up one level.")

  # Remove the now empty extracted directory
  os.rmdir(os.path.join(save_dir, 'MachineLearningCVE'))
except zipfile.BadZipFile:
  print("The downloaded file is not a valid ZIP file.")