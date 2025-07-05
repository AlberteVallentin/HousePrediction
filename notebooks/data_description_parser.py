"""
Data Description Parser for Ames Housing Dataset

This module provides functions to parse and display the data_description.txt file
in a structured format for use in Jupyter notebooks.
"""

import pandas as pd
import re
from typing import List, Optional


def parse_data_description(file_path: str) -> pd.DataFrame:
    """
    Parse the data_description.txt file into a structured DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the data_description.txt file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: Feature, Description, Type, Categories
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    features_data = []
    lines = content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Check if line contains a feature definition (has colon)
        if ':' in line and not line.startswith('       '):
            # Extract feature name and description
            feature_name, description = line.split(':', 1)
            feature_name = feature_name.strip()
            description = description.strip()
            
            # Look ahead for categories/values
            categories = {}
            i += 1
            
            # Skip empty line after feature definition
            while i < len(lines) and not lines[i].strip():
                i += 1
            
            # Parse categorical values (indented lines)
            while i < len(lines):
                current_line = lines[i]
                
                # Check if this is an indented category line
                if current_line.startswith('       ') and current_line.strip():
                    # Parse category line
                    category_content = current_line.strip()
                    
                    # Split by tab or multiple spaces to separate code and description
                    parts = re.split(r'\t+|\s{2,}', category_content, 1)
                    
                    if len(parts) >= 2:
                        code = parts[0].strip()
                        desc = parts[1].strip()
                        categories[code] = desc
                    elif len(parts) == 1:
                        # Single value without description
                        code = parts[0].strip()
                        categories[code] = code
                
                # Stop if we hit a non-indented line or empty line followed by feature
                elif not current_line.startswith('       '):
                    break
                    
                i += 1
            
            # Determine feature type
            if categories:
                feature_type = "Categorical"
                categories_str = str(categories)
            else:
                feature_type = "Numerical"
                categories_str = "None"
            
            features_data.append({
                'Feature': feature_name,
                'Description': description,
                'Type': feature_type,
                'Categories': categories_str,
                'Category_Count': len(categories) if categories else 0
            })
            
            # Don't increment i here as it's already been incremented in the while loop
            continue
        else:
            i += 1
    
    # Create DataFrame
    df = pd.DataFrame(features_data)
    return df


def search_feature(df: pd.DataFrame, feature_name: str) -> Optional[pd.Series]:
    """
    Search for a specific feature in the parsed data description.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Parsed data description DataFrame
    feature_name : str
        Name of the feature to search for
        
    Returns:
    --------
    pd.Series or None
        Feature information if found, None otherwise
    """
    
    # Exact match first
    exact_match = df[df['Feature'] == feature_name]
    if not exact_match.empty:
        return exact_match.iloc[0]
    
    # Partial match
    partial_match = df[df['Feature'].str.contains(feature_name, case=False, na=False)]
    if not partial_match.empty:
        return partial_match.iloc[0]
    
    return None


def display_feature_info(df: pd.DataFrame, feature_name: str) -> None:
    """
    Display detailed information about a specific feature.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Parsed data description DataFrame
    feature_name : str
        Name of the feature to display
    """
    
    feature_info = search_feature(df, feature_name)
    
    if feature_info is None:
        print(f"Feature '{feature_name}' not found.")
        return
    
    print(f"Feature: {feature_info['Feature']}")
    print(f"Description: {feature_info['Description']}")
    print(f"Type: {feature_info['Type']}")
    
    if feature_info['Type'] == 'Categorical' and feature_info['Categories'] != 'None':
        print("\nCategories:")
        # Parse categories string back to dict for display
        try:
            categories = eval(feature_info['Categories'])
            for code, desc in categories.items():
                print(f"  {code}: {desc}")
        except:
            print(f"  {feature_info['Categories']}")
    
    print("-" * 60)


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of all categorical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Parsed data description DataFrame
        
    Returns:
    --------
    List[str]
        List of categorical feature names
    """
    
    return df[df['Type'] == 'Categorical']['Feature'].tolist()


def get_numerical_features(df: pd.DataFrame) -> List[str]:
    """
    Get list of all numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Parsed data description DataFrame
        
    Returns:
    --------
    List[str]
        List of numerical feature names
    """
    
    return df[df['Type'] == 'Numerical']['Feature'].tolist()


def display_summary_table(df: pd.DataFrame, max_rows: int = 20) -> None:
    """
    Display a summary table of features with truncated categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Parsed data description DataFrame
    max_rows : int
        Maximum number of rows to display
    """
    
    # Create display DataFrame with truncated categories
    display_df = df.copy()
    
    # Truncate long descriptions and categories
    display_df['Description'] = display_df['Description'].apply(
        lambda x: x[:60] + "..." if len(x) > 60 else x
    )
    
    # For categorical features, show count instead of full categories
    display_df['Categories_Summary'] = display_df.apply(
        lambda row: f"{row['Category_Count']} categories" if row['Type'] == 'Categorical' else 'Numerical',
        axis=1
    )
    
    # Select columns for display
    display_cols = ['Feature', 'Type', 'Description', 'Categories_Summary']
    
    print("Feature Summary Table:")
    print("=" * 80)
    print(display_df[display_cols].head(max_rows).to_string(index=False))
    
    if len(display_df) > max_rows:
        print(f"\n... and {len(display_df) - max_rows} more features")
    
    print(f"\nTotal features: {len(display_df)}")
    print(f"Categorical: {len(get_categorical_features(df))}")
    print(f"Numerical: {len(get_numerical_features(df))}")


# Example usage functions for notebooks
def load_feature_descriptions(data_path: str = "../docs/data_description.txt") -> pd.DataFrame:
    """
    Convenience function to load feature descriptions in notebooks.
    
    Parameters:
    -----------
    data_path : str
        Path to data_description.txt file
        
    Returns:
    --------
    pd.DataFrame
        Parsed feature descriptions
    """
    
    return parse_data_description(data_path)


def quick_feature_lookup(feature_name: str, df: pd.DataFrame = None) -> None:
    """
    Quick lookup function for notebooks.
    
    Parameters:
    -----------
    feature_name : str
        Feature to look up
    df : pd.DataFrame, optional
        Parsed descriptions DataFrame. If None, will load from default path.
    """
    
    if df is None:
        df = load_feature_descriptions()
    
    display_feature_info(df, feature_name)


