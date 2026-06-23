import csv
import numpy as np
import os

class DataLoader:
    """Handles loading data from disk."""
    def __init__(self, delimiter=';'):
        self.delimiter = delimiter

    def load(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found at {file_path}")
        
        print(f"Loading data from {file_path}...")
        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            header = next(reader)
            for row in reader:
                data.append(row)
        return np.array(data), header
