import os
import kagglehub
import bz2
import glob

def download_chess_data(dataset_name="ironicninja/raw-chess-games-pgn", download_path="data/input/"):
    """
    Download chess dataset from Kaggle using kagglehub.

    Args:
        dataset_name (str): Name of the Kaggle dataset (e.g., 'ironicninja/raw-chess-games-pgn')
        download_path (str): Local path to save the downloaded data (not used with kagglehub)
    """
    print(f"Downloading dataset '{dataset_name}' from Kaggle using kagglehub...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download(dataset_name)
        print(f"Dataset downloaded to: {path}")
        
        # Find the bz2 file
        bz2_files = glob.glob(os.path.join(path, "*.bz2"))
        if bz2_files:
            bz2_file = bz2_files[0]  # Assume first bz2 file
            print(f"Found bz2 file: {bz2_file}")
            
            # Copy to our data/input and extract
            os.makedirs(download_path, exist_ok=True)
            extracted_file = os.path.join(download_path, os.path.basename(bz2_file).replace('.bz2', ''))
            
            print(f"Extracting {bz2_file} to {extracted_file}...")
            with bz2.open(bz2_file, 'rb') as f_in, open(extracted_file, 'wb') as f_out:
                f_out.write(f_in.read())
            print(f"Extraction complete: {extracted_file}")
        else:
            print("No bz2 file found in downloaded dataset.")
            # List files
            files = os.listdir(path)
            print(f"Downloaded files: {files}")
        
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have internet connection and the dataset name is correct.")
        return False

if __name__ == "__main__":
    success = download_chess_data()
    if not success:
        print("Download failed.")