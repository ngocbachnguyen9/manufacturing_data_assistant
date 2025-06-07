import pandas as pd
import os
from typing import Dict, List


class DataLoader:
    """Handles loading of manufacturing data from CSV files."""

    def __init__(self, base_path: str = "data/manufacturing_base"):
        self.base_path = base_path
        self.data_files = [
            "location_data.csv",
            "machine_log.csv",
            "relationship_data.csv",
            "worker_data.csv",
        ]

    def load_base_data(self) -> Dict[str, pd.DataFrame]:
        """Loads the original, unmodified manufacturing data."""
        data = {}
        print(f"Loading base data from: {self.base_path}")
        for file_name in self.data_files:
            try:
                file_path = os.path.join(self.base_path, file_name)
                key = file_name.replace(".csv", "")
                # Use specific dtypes to prevent pandas from misinterpreting IDs
                data[key] = pd.read_csv(
                    file_path, dtype=str, keep_default_na=False
                )
                print(f"  - Loaded {file_name} into '{key}'")
            except FileNotFoundError:
                print(f"  - WARNING: {file_name} not found in {self.base_path}")
                data[key] = pd.DataFrame()
        return data