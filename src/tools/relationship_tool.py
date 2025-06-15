from typing import Dict, Any, List
import pandas as pd
from .base_tool import BaseTool
from difflib import SequenceMatcher
import re


class RelationshipTool(BaseTool):
    """Tool for traversing parent-child relationships with optional fuzzy matching."""

    def run(self, entity_id: str, **kwargs) -> List[Dict]:
        """
        Finds all direct relationships for an entity ID with fuzzy matching support.

        Args:
            entity_id: The ID to find relationships for.
            **kwargs: Can accept 'fuzzy_enabled' and 'threshold'.
        """
        df = self.datasets.get("relationship_data")
        if df is None or df.empty:
            return [{"error": "Relationship data not available."}]

        fuzzy_enabled = kwargs.get("fuzzy_enabled", False)
        threshold = kwargs.get("threshold", 0.8)

        # Try exact match first
        result_df = df[
            (df["parent"] == entity_id) | (df["child"] == entity_id)
        ]

        # If no exact matches and fuzzy is enabled, try fuzzy matching
        if result_df.empty and fuzzy_enabled:
            print(f"  - No exact match for '{entity_id}', attempting fuzzy matching...")
            fuzzy_matches = self._find_fuzzy_matches(df, entity_id, threshold)
            if fuzzy_matches:
                print(f"  - Found {len(fuzzy_matches)} fuzzy matches")
                # Combine all fuzzy match results
                result_dfs = []
                for match_id, confidence in fuzzy_matches:
                    match_df = df[
                        (df["parent"] == match_id) | (df["child"] == match_id)
                    ]
                    # Add confidence score to results
                    match_df = match_df.copy()
                    match_df["fuzzy_match_confidence"] = confidence
                    match_df["original_query"] = entity_id
                    match_df["matched_id"] = match_id
                    result_dfs.append(match_df)

                if result_dfs:
                    result_df = pd.concat(result_dfs, ignore_index=True)

        if result_df.empty:
            return [{"error": f"No relationships found for ID {entity_id}"}]

        return result_df.to_dict("records")

    def _find_fuzzy_matches(self, df: pd.DataFrame, query_id: str, threshold: float) -> List[tuple]:
        """
        Find fuzzy matches for an entity ID in the relationship data.

        Args:
            df: The relationship dataframe
            query_id: The ID to find fuzzy matches for
            threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (matched_id, confidence_score) tuples
        """
        all_ids = set()
        all_ids.update(df["parent"].dropna().unique())
        all_ids.update(df["child"].dropna().unique())

        matches = []
        for candidate_id in all_ids:
            if candidate_id == query_id:
                continue  # Skip exact matches (already tried)

            # Calculate similarity using multiple methods
            similarity = self._calculate_similarity(query_id, candidate_id)

            if similarity >= threshold:
                matches.append((candidate_id, similarity))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:3]  # Return top 3 matches to avoid too many false positives

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using multiple methods.
        """
        if not str1 or not str2:
            return 0.0

        # Method 1: Sequence matching (handles character removal/insertion)
        seq_similarity = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

        # Method 2: Pattern-based matching for known ID formats
        pattern_similarity = self._pattern_similarity(str1, str2)

        # Method 3: Whitespace normalization (handles space injection)
        normalized_str1 = re.sub(r'***REMOVED***s+', '', str1)
        normalized_str2 = re.sub(r'***REMOVED***s+', '', str2)
        whitespace_similarity = SequenceMatcher(None, normalized_str1.lower(), normalized_str2.lower()).ratio()

        # Take the maximum similarity from all methods
        return max(seq_similarity, pattern_similarity, whitespace_similarity)

    def _pattern_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity based on known manufacturing ID patterns.
        """
        # Extract patterns for different ID types
        patterns = {
            'gear': r'(3DOR)(***REMOVED***d+)',
            'order': r'(ORBOX)(***REMOVED***d+)',
            'printer': r'(Printer)_?(***REMOVED***d+)',
            'material': r'([A-Z]+)(***REMOVED***d+)'
        }

        for pattern_name, pattern in patterns.items():
            match1 = re.match(pattern, str1, re.IGNORECASE)
            match2 = re.match(pattern, str2, re.IGNORECASE)

            if match1 and match2:
                # Both match the same pattern, compare components
                prefix1, num1 = match1.groups()
                prefix2, num2 = match2.groups()

                if prefix1.lower() == prefix2.lower() and num1 == num2:
                    return 0.95  # Very high confidence for same pattern + number
                elif prefix1.lower() == prefix2.lower():
                    return 0.7   # Medium confidence for same pattern, different number

        return 0.0