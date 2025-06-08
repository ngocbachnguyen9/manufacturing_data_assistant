from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class BaseTool(ABC):
    """Abstract base class for all manufacturing tools."""

    def __init__(self, datasets: Dict[str, pd.DataFrame]):
        """
        Initializes the tool with all available data sources.

        Args:
            datasets: A dictionary mapping table names to pandas DataFrames.
        """
        self.datasets = datasets

    @abstractmethod
    def run(self, tool_input: Any) -> Any:
        """The main execution method for the tool."""
        pass