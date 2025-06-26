"""
Example usage of data_description_parser in notebooks

This shows how to integrate the feature description parser into your analysis notebooks.
Copy the relevant code sections into your actual notebooks.
"""

# Example notebook cell - Data Description Setup
"""
# Load and parse feature descriptions
from data_description_parser import load_feature_descriptions, quick_feature_lookup, display_summary_table

# Load the parsed feature descriptions
feature_descriptions = load_feature_descriptions()

print("Dataset Feature Overview:")
display_summary_table(feature_descriptions, max_rows=15)
"""

# Example notebook cell - Feature Investigation
"""
# Investigate specific features during analysis
quick_feature_lookup('OverallQual', feature_descriptions)
quick_feature_lookup('Neighborhood', feature_descriptions)
quick_feature_lookup('GrLivArea', feature_descriptions)
"""

# Example notebook cell - Categorical Features Analysis
"""
from data_description_parser import get_categorical_features, get_numerical_features

# Get feature lists for analysis
categorical_features = get_categorical_features(feature_descriptions)
numerical_features = get_numerical_features(feature_descriptions)

print(f"Categorical features ({len(categorical_features)}):")
for feature in categorical_features[:10]:  # Show first 10
    print(f"  - {feature}")

print(f"\nNumerical features ({len(numerical_features)}):")
for feature in numerical_features[:10]:  # Show first 10
    print(f"  - {feature}")
"""

# Example notebook cell - Feature-Specific Analysis
"""
# When analyzing a specific feature, show its description
feature_name = 'OverallQual'
quick_feature_lookup(feature_name, feature_descriptions)

# Then perform your analysis
print(f"\n{feature_name} value counts:")
print(df[feature_name].value_counts().sort_index())
"""

# Example notebook cell - Missing Data Analysis with Context
"""
# Analyze missing data with feature context
missing_features = df.isnull().sum()
missing_features = missing_features[missing_features > 0].sort_values(ascending=False)

print("Missing Data Analysis with Feature Context:")
print("=" * 60)

for feature in missing_features.head(10).index:
    missing_count = missing_features[feature]
    missing_pct = (missing_count / len(df)) * 100
    
    print(f"\n{feature}: {missing_count} missing ({missing_pct:.1f}%)")
    
    # Show feature description for context
    feature_info = search_feature(feature_descriptions, feature)
    if feature_info is not None:
        print(f"  Description: {feature_info['Description']}")
        if feature_info['Type'] == 'Categorical':
            print(f"  Type: {feature_info['Type']} ({feature_info['Category_Count']} categories)")
    print("-" * 40)
"""

# Example notebook cell - Encoding Strategy with Context
"""
# Develop encoding strategy based on feature descriptions
from data_description_parser import search_feature

def analyze_categorical_feature_for_encoding(feature_name, df, feature_descriptions):
    '''Analyze a categorical feature to determine best encoding strategy'''
    
    print(f"Encoding Analysis for {feature_name}:")
    print("=" * 50)
    
    # Get feature description
    feature_info = search_feature(feature_descriptions, feature_name)
    if feature_info is not None:
        print(f"Description: {feature_info['Description']}")
        print(f"Categories: {feature_info['Category_Count']}")
    
    # Analyze data distribution
    value_counts = df[feature_name].value_counts()
    print(f"\nValue distribution:")
    print(value_counts.head(10))
    
    # Suggest encoding strategy
    unique_values = df[feature_name].nunique()
    if unique_values <= 5:
        strategy = "One-hot encoding (low cardinality)"
    elif unique_values <= 10:
        strategy = "Ordinal encoding if ordered, otherwise one-hot"
    else:
        strategy = "Target encoding (high cardinality)"
    
    print(f"\nSuggested encoding: {strategy}")
    print("-" * 50)

# Example usage
# analyze_categorical_feature_for_encoding('Neighborhood', df, feature_descriptions)
"""

print("Example usage code has been prepared!")
print("Copy the relevant sections into your notebooks as needed.")
print("\nKey functions available:")
print("- load_feature_descriptions(): Load and parse the data description file")
print("- quick_feature_lookup(feature_name): Display detailed feature information")
print("- display_summary_table(): Show overview of all features")
print("- get_categorical_features(): Get list of categorical features")
print("- get_numerical_features(): Get list of numerical features")