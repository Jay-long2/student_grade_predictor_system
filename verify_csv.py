# verify_csv.py
import pandas as pd
import os

def verify_csv_file(file_path):
    """Verify that a CSV file can be read properly"""
    try:
        print(f"Checking file: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
        
        if os.path.getsize(file_path) == 0:
            print("❌ File is empty!")
            return False
        
        # Try to read the file
        df = pd.read_csv(file_path)
        print("✅ CSV file read successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 3 rows:")
        print(df.head(3))
        return True
        
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False

# Test the file
if __name__ == "__main__":
    file_path = "test_data.csv"  # Change this to your file path
    verify_csv_file(file_path)