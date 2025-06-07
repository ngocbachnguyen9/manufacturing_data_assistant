import pandas as pd
import re
from typing import Dict, List


class ManufacturingValidator:
    """
    Provides reusable validation functions for checking manufacturing data
    integrity against predefined rules.
    """

    def __init__(self):
        self.patterns = {
            "worker_rfid": r"^***REMOVED***d{10}$",
            "printer": r"^Printer_***REMOVED***d+$",
            "gear": r"^3DOR***REMOVED***d{5,6}$",
            "order": r"^ORBOX***REMOVED***d+",
            "material": r"^[A-Z]{4}***REMOVED***d{4}$",
        }

    def validate_barcode_formats(self, df: pd.DataFrame, column: str) -> pd.Series:
        """
        Validates a DataFrame column against known barcode patterns.
        Returns a boolean Series indicating compliance.
        """
        # Find the pattern key that matches the column name or content
        pattern_key = None
        if "worker" in column:
            pattern_key = "worker_rfid"
        elif "printer" in column:
            pattern_key = "printer"
        elif "gear" in column:
            pattern_key = "gear"
        elif "order" in column:
            pattern_key = "order"
        elif "material" in column:
            pattern_key = "material"

        # Dynamically determine pattern based on content if no key matches
        if not pattern_key:
            if df[column].str.match(self.patterns["gear"]).any():
                pattern_key = "gear"
            elif df[column].str.match(self.patterns["order"]).any():
                pattern_key = "order"

        if pattern_key:
            return df[column].astype(str).str.match(self.patterns[pattern_key])

        # Return True for all if no pattern applies
        return pd.Series([True] * len(df), index=df.index)