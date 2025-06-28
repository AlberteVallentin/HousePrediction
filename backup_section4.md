# Notebook 02: Data Preprocessing

Systematic preprocessing of Ames Housing dataset based on exploratory findings to prepare clean data for machine learning model development.

---

## 1. Data Loading and Initial Processing

Load datasets and implement parser-guided missing data treatment strategies identified during exploratory analysis.

### 1.1 Dataset Import and Validation

```python
# Load required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load the datasets
df_train = pd.read_csv('../data/raw/train.csv')
df_test = pd.read_csv('../data/raw/test.csv')

print("Dataset Import Validation:")
print(f"Training data: {df_train.shape}")
print(f"Test data: {df_test.shape}")

# Create combined dataset for consistent feature processing
df_combined = pd.concat([
    df_train.drop('SalePrice', axis=1),
    df_test
], ignore_index=True)
df_combined['dataset_source'] = ['train']*len(df_train) + ['test']*len(df_test)

print(f"Combined dataset: {df_combined.shape}")
print(f"Features to process: {df_combined.shape[1] - 1}")

# Validate against Notebook 01 findings
expected_missing_features = 34
actual_missing_features = df_combined.drop('dataset_source', axis=1).isnull().any().sum()
print(f"\nMissing data validation:")
print(f"Expected features with missing data: {expected_missing_features}")
print(f"Actual features with missing data: {actual_missing_features}")
print(f"Validation: {'✓ PASS' if actual_missing_features == expected_missing_features else '✗ FAIL'}")
```

Dataset validation successful - all expected characteristics confirmed from Notebook 01 analysis.

### 1.2 Parser Integration Setup

```python
# Setup data description parser for domain knowledge
from data_description_parser import (
    load_feature_descriptions,
    quick_feature_lookup,
    display_summary_table,
    get_categorical_features,
    get_numerical_features
)

# Load official documentation
feature_descriptions = load_feature_descriptions()
print("Parser Integration Setup:")
print("✓ Official real estate documentation loaded successfully")

# Get feature classifications for preprocessing
categorical_features = get_categorical_features(feature_descriptions)
numerical_features = get_numerical_features(feature_descriptions)

print(f"✓ Categorical features identified: {len(categorical_features)}")
print(f"✓ Numerical features identified: {len(numerical_features)}")

# Verify critical misclassified features from Notebook 01
misclassified_features = ['OverallQual', 'OverallCond', 'MSSubClass']
print(f"\nMisclassified ordinal features to correct:")
for feature in misclassified_features:
    feature_type = 'Categorical' if feature in categorical_features else 'Numerical'
    pandas_type = df_train[feature].dtype
    print(f"  {feature}: Parser={feature_type}, Pandas={pandas_type}")
```

Parser integration confirmed 46 categorical and 33 numerical features with 3 misclassified ordinal features requiring correction.

---

## 2. Feature Classification Correction

Correct misclassified ordinal features identified in Notebook 01 before missing data treatment to ensure proper data types.

### 2.1 Ordinal Feature Correction

```python
# Correct misclassified ordinal features identified in Notebook 01
ordinal_features = ['OverallQual', 'OverallCond', 'MSSubClass']

print("Ordinal Feature Correction:")
print("Converting integer-stored ordinal features to proper categorical types")

# Show current state before correction
print(f"\nBefore correction:")
for feature in ordinal_features:
    dtype = df_combined[feature].dtype
    unique_vals = sorted(df_combined[feature].unique())
    print(f"  {feature}: {dtype} with {len(unique_vals)} unique values: {unique_vals}")

# Convert to ordered categorical for combined dataset
print(f"\nApplying corrections:")
for feature in ordinal_features:
    if feature == 'MSSubClass':
        # MSSubClass: dwelling type categories
        print(f"  {feature}: Converting to unordered categorical (dwelling types)")
        df_combined[feature] = df_combined[feature].astype('category')
    else:
        # OverallQual and OverallCond: 1-10 quality scales
        print(f"  {feature}: Converting to ordered categorical (quality scale)")
        df_combined[feature] = pd.Categorical(df_combined[feature],
                                            categories=sorted(df_combined[feature].unique()),
                                            ordered=True)

print(f"\nAfter correction:")
for feature in ordinal_features:
    dtype = df_combined[feature].dtype
    is_ordered = hasattr(df_combined[feature], 'cat') and df_combined[feature].cat.ordered
    print(f"  {feature}: {dtype} (ordered: {is_ordered})")

    # Show categories for verification
    if hasattr(df_combined[feature], 'cat'):
        categories = list(df_combined[feature].cat.categories)
        print(f"    Categories: {categories}")
```

Successfully converted 3 misclassified features to proper categorical types. OverallQual and OverallCond now preserve ordinal relationships (1-10 scales), while MSSubClass represents dwelling type categories.

---

## 3. Missing Data Treatment

Systematic analysis and treatment of 34 missing data features using parser consultation for domain-guided decisions.

### 3.1 Missing Data Analysis

```python
# Get all features with missing data from combined dataset
missing_data = df_combined.drop('dataset_source', axis=1).isnull().sum()
missing_features = missing_data[missing_data > 0].sort_values(ascending=False)

print("Missing Data Overview:")
print(f"Total features with missing data: {len(missing_features)}")
print(f"Total missing values: {missing_features.sum()}")
print(f"Dataset completeness: {((len(df_combined) * 80 - missing_features.sum()) / (len(df_combined) * 80)) * 100:.1f}%")

print(f"\nTop 10 features with missing data:")
for feature, count in missing_features.head(10).items():
    pct = (count / len(df_combined)) * 100
    print(f"  {feature}: {count} missing ({pct:.1f}%)")

print(f"\nAll missing features for systematic treatment:")
for feature, count in missing_features.items():
    pct = (count / len(df_combined)) * 100
    print(f"  {feature}: {count} ({pct:.1f}%)")
```

Confirmed 34 features with 15,707 missing values across combined dataset. Clear patterns emerge: high-impact amenity features (>50% missing) and coordinated feature groups (garage ~5.4%, basement ~2.8%) indicating systematic architectural absence.

### 3.2 Parser-Guided Feature Analysis

```python
# Parser consultation for ALL missing features - discovery phase
print(f"\nSystematic Parser Consultation for ALL Missing Features:")
print("="*70)

# Get all features with missing data (not predefined lists)
all_missing_features = missing_features.index.tolist()  # From section 3.1
print(f"Total features to analyze: {len(all_missing_features)}")

# Process each feature individually
for i, feature in enumerate(all_missing_features, 1):
    missing_count = df_combined[feature].isnull().sum()
    missing_pct = (missing_count / len(df_combined)) * 100

    print(f"\n{i}. {feature}")
    print("-" * 50)
    print(f"Missing: {missing_count} values ({missing_pct:.1f}%)")

    # Parser consultation
    quick_feature_lookup(feature, feature_descriptions)

    # Show feature context and distribution
    if df_combined[feature].dtype == 'object':
        # Categorical feature
        unique_values = df_combined[feature].dropna().unique()
        print(f"Data type: Categorical ({len(unique_values)} unique values)")

        # Value distribution - show all values
        value_counts = df_combined[feature].value_counts()
        print(f"Value distribution:")
        for value, count in value_counts.items():
            pct = (count / value_counts.sum()) * 100
            print(f"  {value}: {count} ({pct:.1f}%)")
    else:
        # Numerical feature
        train_mask = df_combined['dataset_source'] == 'train'
        train_data = df_combined[train_mask][feature].dropna()
        print(f"Data type: Numerical")
        print(f"Range: {train_data.min():.1f} - {train_data.max():.1f}")
        print(f"Stats: Mean={train_data.mean():.1f}, Median={train_data.median():.1f}")

        # Check for zero values
        zero_count = (train_data == 0).sum()
        if zero_count > 0:
            zero_pct = (zero_count / len(train_data)) * 100
            print(f"Zero values: {zero_count} ({zero_pct:.1f}%)")

    # Pause every 5 features for readability
    if i % 5 == 0 and i < len(all_missing_features):
        print(f"\n{'='*20} Processed {i}/{len(all_missing_features)} features {'='*20}")

print(f"\n✓ Parser consultation completed for all {len(all_missing_features)} features")
print("Next: Categorize features by treatment strategy based on parser findings")
```

Based on parser consultation, categorize features into treatment strategies.

### 3.3 Features Known to be Zero (Complete Structural Absence)

```python
# SECTION 3.3: Features Known to be Zero (Complete Structural Absence)
# Logic: When structures don't exist, related measurements should be 0

print("=== SECTION 3.3: Features Known to be Zero ===")

# These features represent measurements of structures that may not exist
# If the structure doesn't exist, the measurement should be 0, not estimated

# GROUP 1: Amenity/Optional Structure Features - Missing = Structure Doesn't Exist
amenity_none_features = [
    'PoolQC',        # 99.7% missing - most houses don't have pools
    'MiscFeature',   # 96.4% missing - most houses don't have elevators/tennis courts
    'Alley',         # 93.2% missing - most houses don't have alley access
    'Fence',         # 80.4% missing - many houses don't have fences
    'FireplaceQu'    # 48.6% missing - many houses don't have fireplaces
]

# GROUP 2: Coordinated Structure Features - Missing = No Garage/Basement Exists
garage_none_features = [
    'GarageFinish',  # 5.4% missing - no garage = no finish to evaluate
    'GarageQual',    # 5.4% missing - no garage = no quality to rate
    'GarageCond',    # 5.4% missing - no garage = no condition to assess
    'GarageType'     # 5.4% missing - no garage = no type to classify
]

basement_none_features = [
    'BsmtExposure',  # 2.8% missing - no basement = no exposure to evaluate
    'BsmtCond',      # 2.8% missing - no basement = no condition to assess
    'BsmtQual',      # 2.8% missing - no basement = no quality to rate
    'BsmtFinType2',  # 2.7% missing - no basement = no finished area types
    'BsmtFinType1'   # 2.7% missing - no basement = no finished area types
]

# GROUP 3: Masonry Special Case - Has explicit "None" category
masonry_none_features = ['MasVnrType']  # Has "None: None" category for houses without masonry

# Combine all "None" replacement features
all_none_features = amenity_none_features + garage_none_features + basement_none_features + masonry_none_features

print(f"Applying 'None' replacement for {len(all_none_features)} categorical features:")

# Apply "None" replacement for structural absence
for feature in all_none_features:
    before_missing = df_combined[feature].isnull().sum()
    df_combined[feature] = df_combined[feature].fillna('None')
    print(f"✓ {feature}: {before_missing} missing → 'None'")

print(f"\n✓ Completed 'None' replacement for absent structures")
```

Applied 'None' replacement to 15 categorical features representing absent structures (pools, garages, basements, amenities).

### 3.4 Geographic Features (Neighborhood-Based Treatment)

```python
# SECTION 3.4: Geographic Features (Neighborhood-Based Treatment)
# Logic: These features cluster geographically within neighborhoods

print("=== SECTION 3.4: Geographic Features (Neighborhood Mode) ===")

# Geographic numerical features - use neighborhood context
geographic_numerical = ['LotFrontage']  # Street frontage varies by neighborhood development patterns

# Geographic categorical features - use neighborhood mode
geographic_categorical = {
    'MSZoning': 'zoning clusters geographically',
    'Exterior1st': 'architectural styles cluster',
    'Exterior2nd': 'architectural styles cluster',
    'SaleType': 'construction patterns vary by area'
}

# System/standard categorical features - also use neighborhood mode
system_categorical = {
    'Electrical': 'electrical standards may vary by neighborhood development era',
    'Utilities': 'utility access patterns cluster geographically',
    'Functional': 'functionality standards cluster by neighborhood',
    'KitchenQual': 'kitchen quality varies by neighborhood characteristics'
}

# Process geographic numerical features
print("Geographic numerical features:")
for feature in geographic_numerical:
    if df_combined[feature].isnull().sum() > 0:
        missing_count = df_combined[feature].isnull().sum()
        df_combined[feature] = df_combined.groupby('Neighborhood')[feature].transform(
            lambda x: x.fillna(x.median())
        )
        print(f"✓ {feature}: {missing_count} missing → neighborhood median")

# Process all categorical features with neighborhood mode
all_categorical = {**geographic_categorical, **system_categorical}

print(f"\nCategorical features (neighborhood mode):")
for feature, reason in all_categorical.items():
    if df_combined[feature].isnull().sum() > 0:
        missing_count = df_combined[feature].isnull().sum()

        # Calculate neighborhood mode
        neighborhood_mode = df_combined.groupby('Neighborhood')[feature].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else None
        )

        # Fill missing values with neighborhood mode
        for neighborhood in df_combined['Neighborhood'].unique():
            if neighborhood in neighborhood_mode.index and pd.notna(neighborhood_mode[neighborhood]):
                mask = (df_combined['Neighborhood'] == neighborhood) & df_combined[feature].isnull()
                df_combined.loc[mask, feature] = neighborhood_mode[neighborhood]

        # Fallback: use overall mode for any remaining missing values
        if df_combined[feature].isnull().sum() > 0:
            overall_mode = df_combined[feature].mode()[0]
            df_combined[feature].fillna(overall_mode, inplace=True)

        print(f"✓ {feature}: {missing_count} missing → neighborhood mode ({reason})")

print(f"\n✓ Completed neighborhood-based treatment for geographic features")
```

Geographic and system features treated using neighborhood mode for 9 categorical and 1 numerical feature.

### 3.5 Coordinated Numerical Features (Manual Review Required)

```python
# SECTION 3.5: Coordinated Numerical Features (Manual Review Required)
# Logic: Check if ALL values are missing (structure absent) → set to 0
#        If SOME values are missing → print house IDs for manual decision

print("=== SECTION 3.5: Coordinated Numerical Features ===")

# Define coordinated numerical features that should be analyzed together
coordinated_features = {
    'GarageYrBlt': 'garage construction year',
    'GarageArea': 'garage square footage',
    'GarageCars': 'garage capacity',
    'TotalBsmtSF': 'basement total square footage',
    'BsmtUnfSF': 'basement unfinished square footage',
    'BsmtFinSF1': 'basement finished area 1',
    'BsmtFinSF2': 'basement finished area 2',
    'BsmtFullBath': 'basement full bathrooms',
    'BsmtHalfBath': 'basement half bathrooms',
    'MasVnrArea': 'masonry veneer area'
}

# Special handling for GarageYrBlt (coordinate with house construction)
if 'GarageYrBlt' in coordinated_features and df_combined['GarageYrBlt'].isnull().sum() > 0:
    missing_count = df_combined['GarageYrBlt'].isnull().sum()
    print(f"\nGarageYrBlt analysis:")
    print(f"Missing values: {missing_count}")

    # Check coordination with GarageType
    garage_exists_mask = df_combined['GarageType'] != 'None'
    garage_missing_mask = df_combined['GarageYrBlt'].isnull()

    # Set to 0 for houses without garages
    no_garage_missing_year = ~garage_exists_mask & garage_missing_mask
    has_garage_but_missing_year = garage_exists_mask & garage_missing_mask

    if no_garage_missing_year.sum() > 0:
        df_combined.loc[no_garage_missing_year, 'GarageYrBlt'] = 0
        print(f"✓ Set {no_garage_missing_year.sum()} houses without garages to 0")

    if has_garage_but_missing_year.sum() > 0:
        missing_house_ids = df_combined[has_garage_but_missing_year]['Id'].tolist()
        print(f"→ {has_garage_but_missing_year.sum()} houses with garages missing GarageYrBlt")
        print(f"  House IDs requiring manual review: {missing_house_ids}")

# Analyze remaining coordinated features
remaining_features = {k: v for k, v in coordinated_features.items() if k != 'GarageYrBlt'}

print(f"\nAnalyzing {len(remaining_features)} coordinated numerical features:")

for feature, description in remaining_features.items():
    missing_count = df_combined[feature].isnull().sum()
    total_count = len(df_combined)
    missing_pct = (missing_count / total_count) * 100

    print(f"\n{feature} ({description}):")
    print(f"  Missing: {missing_count}/{total_count} ({missing_pct:.1f}%)")

    if missing_count == 0:
        print(f"  ✓ No missing values")
    elif missing_count == total_count:
        print(f"  → ALL missing - setting to 0 (structure doesn't exist)")
        df_combined[feature] = 0
    else:
        # Handle coordinated features with structure existence logic
        if 'Garage' in feature:
            # Check garage existence
            structure_exists_mask = df_combined['GarageType'] != 'None'
            missing_mask = df_combined[feature].isnull()

            # Set to 0 for houses without garages
            no_structure = ~structure_exists_mask & missing_mask
            has_structure_missing = structure_exists_mask & missing_mask

            if no_structure.sum() > 0:
                df_combined.loc[no_structure, feature] = 0
                print(f"  ✓ Set {no_structure.sum()} houses without garage to 0")

            if has_structure_missing.sum() > 0:
                missing_house_ids = df_combined[has_structure_missing]['Id'].tolist()
                print(f"  → {has_structure_missing.sum()} houses with garages missing {feature}")
                print(f"  House IDs requiring manual review: {missing_house_ids}")

        elif 'Bsmt' in feature:
            # Check basement existence
            structure_exists_mask = df_combined['BsmtQual'] != 'None'
            missing_mask = df_combined[feature].isnull()

            # Set to 0 for houses without basements
            no_structure = ~structure_exists_mask & missing_mask
            has_structure_missing = structure_exists_mask & missing_mask

            if no_structure.sum() > 0:
                df_combined.loc[no_structure, feature] = 0
                print(f"  ✓ Set {no_structure.sum()} houses without basement to 0")

            if has_structure_missing.sum() > 0:
                missing_house_ids = df_combined[has_structure_missing]['Id'].tolist()
                print(f"  → {has_structure_missing.sum()} houses with basements missing {feature}")
                print(f"  House IDs requiring manual review: {missing_house_ids}")

        elif 'MasVnr' in feature:
            # Check masonry existence
            structure_exists_mask = df_combined['MasVnrType'] != 'None'
            missing_mask = df_combined[feature].isnull()

            # Set to 0 for houses without masonry
            no_structure = ~structure_exists_mask & missing_mask
            has_structure_missing = structure_exists_mask & missing_mask

            if no_structure.sum() > 0:
                df_combined.loc[no_structure, feature] = 0
                print(f"  ✓ Set {no_structure.sum()} houses without masonry to 0")

            if has_structure_missing.sum() > 0:
                missing_house_ids = df_combined[has_structure_missing]['Id'].tolist()
                print(f"  → {has_structure_missing.sum()} houses with masonry missing {feature}")
                print(f"  House IDs requiring manual review: {missing_house_ids}")
        else:
            # Fallback for other features
            missing_mask = df_combined[feature].isnull()
            missing_house_ids = df_combined[missing_mask]['Id'].tolist()
            print(f"  → PARTIAL missing - requires manual review")
            print(f"  House IDs with missing {feature}: {missing_house_ids}")

print(f"\n⚠️  Manual review required for features with partial missing values")
print(f"   Review house IDs above and decide treatment strategy")
```

Coordinated feature analysis identified 2 houses requiring manual review for garage timing and measurement features.

### 3.6 Manual Review Analysis for Missing Coordinated Features

```python
# Manual review analysis for houses with missing coordinated features
print("=== MANUAL REVIEW: Missing Coordinated Features ===")

# Houses requiring manual review (actual house IDs from dataset)
garage_timing_missing = [2127, 2577]  # Missing GarageYrBlt
garage_measurement_missing = [2577]   # Missing GarageArea, GarageCars

print(f"Houses missing GarageYrBlt: {garage_timing_missing}")
print(f"Houses missing garage measurements: {garage_measurement_missing}")

# Analyze each house's characteristics
review_houses = list(set(garage_timing_missing + garage_measurement_missing))

for house_id in review_houses:
    print(f"\n--- House ID {house_id} Analysis ---")

    # Get house data using ID lookup
    house_data = df_combined[df_combined['Id'] == house_id].iloc[0]

    # Garage features
    garage_features = ['GarageType', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageYrBlt', 'GarageArea', 'GarageCars']
    print("Garage features:")
    for feature in garage_features:
        if feature in df_combined.columns:
            value = house_data[feature]
            is_missing = pd.isna(value)
            print(f"  {feature}: {value} {'(MISSING)' if is_missing else ''}")

    # House construction timing
    timing_features = ['YearBuilt', 'YearRemodAdd']
    print("Construction timing:")
    for feature in timing_features:
        if feature in df_combined.columns:
            value = house_data[feature]
            print(f"  {feature}: {value}")

    # Additional context
    context_features = ['OverallQual', 'OverallCond', 'Neighborhood']
    print("Additional context:")
    for feature in context_features:
        if feature in df_combined.columns:
            value = house_data[feature]
            print(f"  {feature}: {value}")

print(f"\n=== Manual Decision Required ===")
print("Based on analysis above, decide treatment strategy for each missing feature:")
print("- GarageYrBlt: Set to YearBuilt, YearRemodAdd, or specific year?")
print("- GarageArea/GarageCars: Use neighborhood median or specific value?")
```

Analysis identified 2 houses requiring individual assessment for garage feature consistency and data quality decisions.

### 3.7 Manual Decision Implementation

```python
# Apply manual decisions for missing coordinated features
print("=== MANUAL DECISION IMPLEMENTATION ===")

# House 2127: Garage exists with measurements, only missing year
print("House 2127: Setting GarageYrBlt to YearRemodAdd (1983)")
house_2127_idx = df_combined[df_combined['Id'] == 2127].index[0]
df_combined.loc[house_2127_idx, 'GarageYrBlt'] = df_combined.loc[house_2127_idx, 'YearRemodAdd']

# House 2577: All garage features missing/None, treat as no functional garage
print("House 2577: Setting all garage features to None/0 (no functional garage)")
house_2577_idx = df_combined[df_combined['Id'] == 2577].index[0]
df_combined.loc[house_2577_idx, 'GarageType'] = 'None'
df_combined.loc[house_2577_idx, 'GarageYrBlt'] = 0
df_combined.loc[house_2577_idx, 'GarageArea'] = 0
df_combined.loc[house_2577_idx, 'GarageCars'] = 0

# Validate changes
print(f"\nValidation:")
for house_id in [2127, 2577]:
    house_idx = df_combined[df_combined['Id'] == house_id].index[0]
    garage_features = ['GarageType', 'GarageYrBlt', 'GarageArea', 'GarageCars']
    values = [df_combined.loc[house_idx, feature] for feature in garage_features]
    print(f"House {house_id}: {dict(zip(garage_features, values))}")

# Final missing data check for coordinated features
coordinated_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF',
                       'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

remaining_missing = df_combined[coordinated_features].isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

print(f"\nRemaining missing values in coordinated features:")
if len(remaining_missing) == 0:
    print("✓ No missing values remaining")
else:
    for feature, count in remaining_missing.items():
        print(f"  {feature}: {count}")

print(f"\n✓ Manual decisions implemented for garage feature consistency")
```

Manual decisions applied: House 2127 garage timing corrected, House 2577 treated as no functional garage for data consistency.

### 3.8 Final Missing Data Validation

```python
# Final comprehensive missing data check
print("=== FINAL MISSING DATA VALIDATION ===")

# Check all features for remaining missing values
all_missing = df_combined.drop('dataset_source', axis=1).isnull().sum()
remaining_missing = all_missing[all_missing > 0].sort_values(ascending=False)

print(f"Missing data summary:")
print(f"Total features: {len(all_missing)}")
print(f"Features with missing values: {len(remaining_missing)}")
print(f"Total missing values: {remaining_missing.sum()}")

if len(remaining_missing) == 0:
    print("✓ SUCCESS: No missing values remaining in dataset")
else:
    print(f"\n⚠️ Remaining missing values:")
    for feature, count in remaining_missing.items():
        pct = (count / len(df_combined)) * 100
        print(f"  {feature}: {count} ({pct:.2f}%)")

# Validate specific coordinated features are clean
coordinated_check = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'BsmtUnfSF',
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']

coordinated_missing = df_combined[coordinated_check].isnull().sum()
coordinated_remaining = coordinated_missing[coordinated_missing > 0]

print(f"\nCoordinated features validation:")
if len(coordinated_remaining) == 0:
    print("✓ All coordinated numerical features complete")
else:
    print("⚠️ Issues with coordinated features:")
    for feature, count in coordinated_remaining.items():
        print(f"  {feature}: {count} missing")

# Sample validation for manual decision houses
print(f"\nManual decision validation:")
for house_id in [2127, 2577]:
    house_idx = df_combined[df_combined['Id'] == house_id].index[0]
    garage_yr = df_combined.loc[house_idx, 'GarageYrBlt']
    garage_type = df_combined.loc[house_idx, 'GarageType']
    print(f"House {house_id}: GarageType={garage_type}, GarageYrBlt={garage_yr}")

print(f"\n✓ Missing data treatment pipeline completed successfully")
```

Missing data treatment completed for 34 features using structure-aware logic, achieving zero missing values with coordinated feature consistency.

---

---

## 4. Categorical Feature Encoding

Manual integer mapping for ordinal features and one-hot encoding for nominal features to ensure optimal ML algorithm compatibility.

### 4.1 Ordinal Encoding

```python
# Manual integer mapping for ordinal features ensures predictable relationships for ML algorithms
print("Ordinal Feature Encoding:")

# Standard quality/condition features with consistent 5-point scale
ordinal_1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 
             'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for feature in ordinal_1:
    if feature in df_combined.columns:
        if 'None' in df_combined[feature].values:
            df_combined[feature] = df_combined[feature].map({"None":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')
        else:
            df_combined[feature] = df_combined[feature].map({"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')
        print(f"✓ {feature}: Encoded to 0-5 scale")

# Individual ordinal features with custom mappings
ordinal_2 = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'Fence']

df_combined['BsmtExposure'] = df_combined['BsmtExposure'].map({"None":0, "No":1,"Mn":2,"Av":3,"Gd":4}).astype('int')
df_combined['BsmtFinType1'] = df_combined['BsmtFinType1'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df_combined['BsmtFinType2'] = df_combined['BsmtFinType2'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df_combined['GarageFinish'] = df_combined['GarageFinish'].map({"None":0,"Unf":1,"RFn":2,"Fin":3}).astype('int')
df_combined['Fence'] = df_combined['Fence'].map({"None":0, "MnWw":1,"GdWo":2,"MnPrv":3,"GdPrv":4}).astype('int')

for feature in ordinal_2:
    print(f"✓ {feature}: Custom ordinal mapping applied")

# Get list of all encoded ordinal features
ordinal_encoded_features = ordinal_1 + ordinal_2
print(f"\n✓ Total ordinal features encoded: {len(ordinal_encoded_features)}")
```

### 4.2 Ordinal Encoding Implementation

```python
# Apply manual integer mapping to ordinal features - ensures predictable relationships for ML algorithms
print("Applying Manual Integer Mapping to Ordinal Features:")

# Standard quality/condition features with consistent 5-point scale
standard_ordinal = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                   'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

for feature in standard_ordinal:
    if feature in df_combined.columns:
        if 'None' in df_combined[feature].values:
            df_combined[feature] = df_combined[feature].map({"None":0,"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')
        else:
            df_combined[feature] = df_combined[feature].map({"Po":1,"Fa":2,"TA":3,"Gd":4,"Ex":5}).astype('int')
        print(f"✓ {feature}: 5-point scale → integers 0-5")

# Individual ordinal features with custom mappings
print(f"\nCustom ordinal mappings:")
df_combined['BsmtExposure'] = df_combined['BsmtExposure'].map({"None":0, "No":1,"Mn":2,"Av":3,"Gd":4}).astype('int')
df_combined['BsmtFinType1'] = df_combined['BsmtFinType1'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df_combined['BsmtFinType2'] = df_combined['BsmtFinType2'].map({"None":0, "Unf":1,"LwQ":2,"Rec":3,"BLQ":4,"ALQ":5,"GLQ":6}).astype('int')
df_combined['GarageFinish'] = df_combined['GarageFinish'].map({"None":0,"Unf":1,"RFn":2,"Fin":3}).astype('int')
df_combined['Fence'] = df_combined['Fence'].map({"None":0, "MnWw":1,"GdWo":2,"MnPrv":3,"GdPrv":4}).astype('int')

ordinal_encoded_features = standard_ordinal + ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageFinish', 'Fence']
print(f"✓ {len(ordinal_encoded_features)} ordinal features encoded as integers")
```

### 4.3 Nominal Categorical Analysis

```python
# Identify remaining categorical features for one-hot encoding
categorical_features = df_combined.select_dtypes(include=['object']).columns.tolist()

# Remove ordinal features and special cases
nominal_features = [f for f in categorical_features
                   if f not in ordinal_encoded_features
                   and f not in special_ordinal
                   and f != 'dataset_source']

print(f"Nominal Categorical Features Analysis:")
print(f"Total categorical features: {len(categorical_features)}")
print(f"Ordinal features (already encoded): {len(ordinal_encoded_features)}")
print(f"Nominal features for one-hot encoding: {len(nominal_features)}")

# Analyze cardinality for encoding strategy
high_cardinality_features = []
print(f"\nNominal feature cardinality:")
for feature in nominal_features:
    unique_count = df_combined[feature].nunique()
    print(f"  {feature}: {unique_count} categories")
    if unique_count > 10:
        high_cardinality_features.append(feature)

print(f"\nHigh cardinality features (>10 categories): {len(high_cardinality_features)}")
for feature in high_cardinality_features:
    print(f"  {feature}: {df_combined[feature].nunique()} categories")
```

### 4.4 One-Hot Encoding Implementation

```python
# Apply one-hot encoding to remaining nominal categorical features
print("One-Hot Encoding Implementation:")

# Get remaining categorical features (exclude ordinal and special cases)
categorical_features = df_combined.select_dtypes(include=['object']).columns.tolist()
nominal_features = [f for f in categorical_features
                   if f not in ordinal_encoded_features
                   and f not in ['OverallQual', 'OverallCond', 'MSSubClass', 'dataset_source']]

print(f"Nominal features for one-hot encoding: {len(nominal_features)}")
for feature in nominal_features:
    categories = df_combined[feature].nunique()
    print(f"  {feature}: {categories} categories")

# Apply one-hot encoding with drop_first to prevent multicollinearity
if len(nominal_features) > 0:
    print(f"\nApplying one-hot encoding (drop_first=True prevents multicollinearity):")

    # Create one-hot encoded features
    df_encoded = pd.get_dummies(df_combined[nominal_features], drop_first=True, dtype=int)

    # Drop original categorical columns and add encoded columns
    df_combined = df_combined.drop(columns=nominal_features)
    df_combined = pd.concat([df_combined, df_encoded], axis=1)

    print(f"✓ {len(nominal_features)} nominal features → {df_encoded.shape[1]} binary columns")
    print(f"✓ Dataset shape after encoding: {df_combined.shape}")

print(f"\n✓ Categorical encoding completed using manual integer mapping and one-hot encoding")
```

Categorical encoding completed using manual integer mapping for ordinal features and one-hot encoding for nominal features, ensuring optimal ML algorithm compatibility.

---

## 5. Variable Transformations

Apply transformations to improve distribution characteristics for modeling optimization.

### 5.1 Target Variable Transformation

```python
# Apply log transformation to SalePrice based on Notebook 01 findings
print("Target Variable Transformation:")
print("Applying log transformation to address right skewness (1.88)")

# Calculate original statistics
original_skewness = df_train['SalePrice'].skew()
original_kurtosis = df_train['SalePrice'].kurtosis()

print(f"Original SalePrice skewness: {original_skewness:.4f}")
print(f"Original SalePrice kurtosis: {original_kurtosis:.4f}")

# Apply log transformation
df_train['SalePrice_log'] = np.log1p(df_train['SalePrice'])

# Calculate transformed statistics
log_skewness = df_train['SalePrice_log'].skew()
log_kurtosis = df_train['SalePrice_log'].kurtosis()

print(f"Log-transformed skewness: {log_skewness:.4f}")
print(f"Log-transformed kurtosis: {log_kurtosis:.4f}")
print(f"Skewness improvement: {abs(original_skewness) - abs(log_skewness):.4f}")

# Store transformation info for inverse transformation
print("\n✓ Log transformation applied - use np.expm1() for inverse transformation")
```

### 5.2 Numerical Feature Distribution Analysis

```python
# Systematic skewness analysis for all numerical features
numerical_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in ['Id', 'SalePrice', 'SalePrice_log']]

print("Numerical Feature Skewness Analysis:")
print(f"Analyzing {len(numerical_cols)} numerical features")

skewness_analysis = {}
for feature in numerical_cols:
    skewness = df_train[feature].skew()
    skewness_analysis[feature] = skewness

# Sort by absolute skewness
sorted_skewness = sorted(skewness_analysis.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\nTop 15 most skewed numerical features:")
for feature, skew in sorted_skewness[:15]:
    print(f"  {feature}: {skew:.4f}")

# Identify candidates for log transformation (|skewness| > 0.75)
log_transform_candidates = [feature for feature, skew in sorted_skewness if abs(skew) > 0.75]

print(f"\nLog transformation candidates (|skewness| > 0.75): {len(log_transform_candidates)}")
for feature in log_transform_candidates[:10]:  # Show top 10
    skew = skewness_analysis[feature]
    print(f"  {feature}: {skew:.4f}")

# Apply log transformation to highly skewed features
transformed_features = []
for feature in log_transform_candidates:
    if df_train[feature].min() >= 0:  # Only transform non-negative features
        # Apply log1p transformation
        df_train[f'{feature}_log'] = np.log1p(df_train[feature])
        df_test[f'{feature}_log'] = np.log1p(df_test[feature])
        df_combined[f'{feature}_log'] = np.log1p(df_combined[feature])

        # Check improvement
        original_skew = skewness_analysis[feature]
        new_skew = df_train[f'{feature}_log'].skew()
        improvement = abs(original_skew) - abs(new_skew)

        if improvement > 0.1:  # Keep transformation if significant improvement
            transformed_features.append(feature)
            print(f"  ✓ {feature}: {original_skew:.4f} → {new_skew:.4f} (improvement: {improvement:.4f})")

print(f"\n✓ Applied log transformation to {len(transformed_features)} numerical features")
```

---

## 6. Outlier Treatment

Remove identified data quality outliers while preserving legitimate market extremes.

### 6.1 Outlier Removal

```python
# Remove data quality outliers identified in Notebook 01
outlier_ids = [524, 1299]  # Partial sales of incomplete luxury properties

print("Outlier Removal Strategy:")
print("Removing only clear data quality violations based on Notebook 01 analysis")
print(f"Outlier IDs to remove: {outlier_ids}")

# Show outlier characteristics before removal
outlier_analysis = df_train[df_train['Id'].isin(outlier_ids)]
print(f"\nOutlier characteristics:")
for idx, row in outlier_analysis.iterrows():
    print(f"  ID {row['Id']}: {row['GrLivArea']:.0f} sqft, ${row['SalePrice']:,}, OverallQual={row['OverallQual']}")

# Remove outliers from training data
before_count = len(df_train)
df_train_clean = df_train[~df_train['Id'].isin(outlier_ids)].reset_index(drop=True)
after_count = len(df_train_clean)

print(f"\nOutlier removal impact:")
print(f"  Before: {before_count} samples")
print(f"  After: {after_count} samples")
print(f"  Removed: {before_count - after_count} samples ({((before_count - after_count) / before_count) * 100:.2f}%)")

# Update working dataset
df_train = df_train_clean
print("✓ Outlier removal completed - conservative approach preserving market diversity")
```

### 6.2 Impact Assessment

```python
# Assess impact of outlier removal on key statistics
print("Outlier Removal Impact Assessment:")

# Compare SalePrice statistics
print(f"SalePrice statistics after outlier removal:")
print(f"  Mean: ${df_train['SalePrice'].mean():,.0f}")
print(f"  Median: ${df_train['SalePrice'].median():,.0f}")
print(f"  Std: ${df_train['SalePrice'].std():,.0f}")
print(f"  Skewness: {df_train['SalePrice'].skew():.4f}")

# Compare with log-transformed target
if 'SalePrice_log' in df_train.columns:
    print(f"Log-transformed SalePrice skewness: {df_train['SalePrice_log'].skew():.4f}")

print("✓ Impact assessment completed - clean dataset ready for modeling")
```

---

## 7. Data Export and Validation

Export preprocessed datasets for model development and validate preprocessing pipeline integrity.

### 7.1 Preprocessed Dataset Export

```python
# Export clean datasets for model development
print("Exporting Preprocessed Datasets:")

# Split preprocessed combined dataset back to train/test
print("Splitting Combined Dataset:")
train_mask = df_combined['dataset_source'] == 'train'
test_mask = df_combined['dataset_source'] == 'test'

df_train_processed = df_combined[train_mask].drop('dataset_source', axis=1).reset_index(drop=True)
df_test_processed = df_combined[test_mask].drop('dataset_source', axis=1).reset_index(drop=True)

print(f"✓ Processed training data: {df_train_processed.shape}")
print(f"✓ Processed test data: {df_test_processed.shape}")

# Verify feature consistency
print(f"\nFeature consistency check:")
print(f"Training features: {df_train_processed.shape[1]}")
print(f"Test features: {df_test_processed.shape[1]}")
print(f"Feature match: {set(df_train_processed.columns) == set(df_test_processed.columns)}")

# Add target variable back to training data
df_train_processed['SalePrice'] = df_train['SalePrice']
if 'SalePrice_log' in df_train.columns:
    df_train_processed['SalePrice_log'] = df_train['SalePrice_log']

# Export datasets
df_train_processed.to_csv('../data/processed/processed_train.csv', index=False)
df_test_processed.to_csv('../data/processed/processed_test.csv', index=False)
df_combined.drop('dataset_source', axis=1).to_csv('../data/processed/processed_combined.csv', index=False)

print(f"✓ Exported processed_train.csv: {df_train_processed.shape}")
print(f"✓ Exported processed_test.csv: {df_test_processed.shape}")
print(f"✓ Exported processed_combined.csv: {df_combined.shape[0]} x {df_combined.shape[1]-1}")
```

### 7.2 Pipeline Validation

```python
# Validate preprocessing pipeline integrity
print("Preprocessing Pipeline Validation:")

# Check for remaining missing values
train_missing = df_train_processed.isnull().sum().sum()
test_missing = df_test_processed.isnull().sum().sum()

print(f"Missing values in processed datasets:")
print(f"  Training: {train_missing}")
print(f"  Test: {test_missing}")

# Validate feature consistency
train_features = set(df_train_processed.columns) - {'SalePrice', 'SalePrice_log'}
test_features = set(df_test_processed.columns)
feature_mismatch = train_features.symmetric_difference(test_features)

print(f"Feature consistency:")
print(f"  Training features: {len(train_features)}")
print(f"  Test features: {len(test_features)}")
print(f"  Mismatched features: {len(feature_mismatch)}")

if len(feature_mismatch) > 0:
    print(f"  Mismatch details: {feature_mismatch}")

# Summary of preprocessing transformations
print(f"\nPreprocessing Summary:")
print(f"✓ Ordinal feature correction: 3 features")
print(f"✓ Missing data treatment: 34 features")
print(f"✓ Log transformations applied: {len(transformed_features) + 1} features")  # +1 for SalePrice
print(f"✓ Outliers removed: 2 samples")
print(f"✓ Final dataset: {df_train_processed.shape[0]} train samples, {df_test_processed.shape[0]} test samples")

print("\n✓ Preprocessing pipeline completed successfully")
```
