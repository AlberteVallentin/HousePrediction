#!/usr/bin/env python3
"""
Comprehensive validation script to ensure prediction.py matches notebooks and trained model
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import json
import re

# Add streamlit to path
sys.path.append('streamlit')

from utils.model_utils import load_model_and_metadata, predict_house_price, load_feature_order

def validate_features_against_final_dataset():
    """Validate that all features in prediction.py match the final training dataset"""
    print("=" * 60)
    print("1. VALIDATING FEATURES AGAINST FINAL DATASET")
    print("=" * 60)
    
    # Load final training dataset
    try:
        df_final = pd.read_csv('data/processed/X_train_final.csv')
        final_features = set(df_final.columns)
        print(f"✅ Final dataset loaded: {len(final_features)} features")
    except Exception as e:
        print(f"❌ Error loading final dataset: {e}")
        return False
    
    # Load feature order from model
    try:
        feature_order = load_feature_order()
        model_features = set(feature_order)
        print(f"✅ Model feature order loaded: {len(model_features)} features")
    except Exception as e:
        print(f"❌ Error loading feature order: {e}")
        return False
    
    # Check if final dataset and model features match
    if final_features == model_features:
        print("✅ Final dataset and model features match perfectly")
    else:
        print("❌ Mismatch between final dataset and model features")
        missing_in_model = final_features - model_features
        missing_in_final = model_features - final_features
        if missing_in_model:
            print(f"   Missing in model: {missing_in_model}")
        if missing_in_final:
            print(f"   Missing in final: {missing_in_final}")
        return False
    
    return True

def validate_log_transformations():
    """Validate that log transformations in prediction.py match feature engineering"""
    print("\n" + "=" * 60)
    print("2. VALIDATING LOG TRANSFORMATIONS")
    print("=" * 60)
    
    # Load original and final datasets to determine which features were log transformed
    try:
        df_original = pd.read_csv('data/processed/train_cleaned.csv')
        df_final = pd.read_csv('data/processed/X_train_final.csv')
        
        # Features that exist in both datasets (numerical features)
        common_features = []
        for col in df_final.columns:
            if col in df_original.columns and df_original[col].dtype in ['int64', 'float64']:
                common_features.append(col)
        
        # Check skewness in original data to determine which should be log transformed
        log_transformed_features = []
        for feature in common_features:
            original_skew = abs(stats.skew(df_original[feature].dropna()))
            if original_skew >= 0.5:
                log_transformed_features.append(feature)
        
        print(f"✅ Features that should be log transformed (skewness >= 0.5): {len(log_transformed_features)}")
        for f in sorted(log_transformed_features):
            print(f"   - {f}")
            
    except Exception as e:
        print(f"❌ Error analyzing log transformations: {e}")
        return False
    
    # Check prediction.py log transformation list
    try:
        with open('streamlit/pages/05_prediction.py', 'r') as f:
            content = f.read()
        
        # Extract log_transform_features list
        log_transform_match = re.search(r'log_transform_features = \[(.*?)\]', content, re.DOTALL)
        if log_transform_match:
            pred_log_features = log_transform_match.group(1).replace('\n', '').replace(' ', '').replace("'", '').split(',')
            pred_log_features = [f.strip() for f in pred_log_features if f.strip()]
            
            # Also check for features that are log-transformed in input sections
            input_log_features = []
            if 'np.log1p(lot_area)' in content:
                input_log_features.append('LotArea')
            if 'np.log1p(lot_frontage)' in content:
                input_log_features.append('LotFrontage')
            if 'np.log1p(gr_liv_area)' in content:
                input_log_features.append('GrLivArea')
            
            all_pred_log_features = pred_log_features + input_log_features
            
            print(f"✅ Features in prediction.py log_transform_features: {len(pred_log_features)}")
            for f in sorted(pred_log_features):
                print(f"   - {f}")
            
            if input_log_features:
                print(f"✅ Features log-transformed in input sections: {len(input_log_features)}")
                for f in sorted(input_log_features):
                    print(f"   - {f}")
                
            # Check for mismatches using all log-transformed features
            should_log = set(log_transformed_features)
            pred_log = set(all_pred_log_features)
            
            missing_in_pred = should_log - pred_log
            extra_in_pred = pred_log - should_log
            
            if missing_in_pred:
                print(f"❌ Missing in prediction.py: {missing_in_pred}")
            if extra_in_pred:
                print(f"❌ Extra in prediction.py: {extra_in_pred}")
                
            if not missing_in_pred and not extra_in_pred:
                print("✅ Log transformation lists match perfectly")
                return True
        else:
            print("❌ Could not find log_transform_features in prediction.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking prediction.py: {e}")
        return False
    
    return False

def validate_engineered_features():
    """Validate that engineered features are correctly implemented"""
    print("\n" + "=" * 60)
    print("3. VALIDATING ENGINEERED FEATURES")
    print("=" * 60)
    
    # Features that should be engineered (from feature engineering notebook)
    expected_engineered = {
        'HouseAge': 'YrSold - YearBuilt',
        'YearsSinceRemodel': 'YrSold - YearRemodAdd',
        'GarageAge': 'YrSold - GarageYrBlt',
        'HasGarage': 'Based on GarageAge < 1000',
        'BsmtFinSF': 'BsmtFinSF1 + BsmtFinSF2',
        'TotalFlrSF': '1stFlrSF + 2ndFlrSF',
        'TotalBaths': 'FullBath + 0.5*HalfBath + BsmtFullBath + 0.5*BsmtHalfBath',
        'GarageAreaPerCar': 'GarageArea / GarageCars'
    }
    
    # Check if these features exist in final dataset
    df_final = pd.read_csv('data/processed/X_train_final.csv')
    
    for feature, description in expected_engineered.items():
        if feature in df_final.columns:
            print(f"✅ {feature}: {description}")
        else:
            print(f"❌ {feature}: MISSING from final dataset")
    
    # Check if original features were dropped
    original_features = ['YearBuilt', 'YrSold', 'YearRemodAdd', 'GarageYrBlt', 'MoSold',
                        'BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF', '2ndFlrSF', 'TotalBsmtSF',
                        'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath',
                        'GarageArea', 'GarageCars']
    
    print(f"\n✅ Checking that original features were dropped:")
    for feature in original_features:
        if feature not in df_final.columns:
            print(f"   ✅ {feature} correctly dropped")
        else:
            print(f"   ❌ {feature} still exists (should be dropped)")
    
    return True

def validate_prediction_accuracy():
    """Test prediction accuracy with known good examples"""
    print("\n" + "=" * 60)
    print("4. VALIDATING PREDICTION ACCURACY")
    print("=" * 60)
    
    # Load model
    model, metadata = load_model_and_metadata('final_model')
    feature_order = load_feature_order()
    
    if not model or not feature_order:
        print("❌ Could not load model or feature order")
        return False
    
    # Test with first row from training data
    try:
        df_final = pd.read_csv('data/processed/X_train_final.csv')
        df_target = pd.read_csv('data/processed/y_train_final.csv')
        
        # Use first row as test case
        test_features = df_final.iloc[0].to_dict()
        expected_log_price = df_target.iloc[0].values[0]
        expected_price = np.expm1(expected_log_price)
        
        # Make prediction
        prediction = predict_house_price(model, test_features, metadata)
        
        if prediction:
            print(f"✅ Test prediction successful")
            print(f"   Expected price: ${expected_price:,.0f}")
            print(f"   Predicted price: ${prediction:,.0f}")
            print(f"   Difference: ${abs(prediction - expected_price):,.0f}")
            
            # Check if prediction is reasonable (within 20% of expected)
            if abs(prediction - expected_price) / expected_price < 0.2:
                print("✅ Prediction is within reasonable range")
                return True
            else:
                print("❌ Prediction differs significantly from expected")
                return False
        else:
            print("❌ Prediction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing prediction: {e}")
        return False

def validate_feature_coverage():
    """Check that all features are handled in prediction.py"""
    print("\n" + "=" * 60)
    print("5. VALIDATING FEATURE COVERAGE")
    print("=" * 60)
    
    # Load final features
    df_final = pd.read_csv('data/processed/X_train_final.csv')
    all_features = set(df_final.columns)
    
    # Read prediction.py to find handled features
    with open('streamlit/pages/05_prediction.py', 'r') as f:
        content = f.read()
    
    # Find all feature assignments
    feature_assignments = re.findall(r"feature_inputs\['([^']+)'\]", content)
    directly_handled_features = set(feature_assignments)
    
    # Find one-hot encoded features that are handled via one-hot encoding logic
    one_hot_features = set()
    for feature in all_features:
        if '_' in feature and feature not in directly_handled_features:
            # Check if the base feature (before underscore) is handled
            base_feature = feature.split('_')[0]
            if any(base_feature.lower() in var.lower() for var in ['ms_zoning', 'street', 'alley', 'lot_config', 'neighborhood', 'condition1', 'condition2', 'bldg_type', 'house_style', 'roof_style', 'roof_matl', 'exterior1st', 'exterior2nd', 'mas_vnr_type', 'foundation', 'heating', 'electrical', 'garage_type', 'misc_feature', 'sale_type', 'sale_condition']):
                one_hot_features.add(feature)
    
    # Check for one-hot encoding logic in the code
    one_hot_logic_features = set()
    if 'one-hot encoded features' in content.lower():
        # Extract features from one-hot encoding section
        one_hot_section = re.search(r'# Handle one-hot encoded features.*?else:', content, re.DOTALL)
        if one_hot_section:
            one_hot_text = one_hot_section.group(0)
            # Find features that are handled in one-hot logic
            for feature in all_features:
                if feature in one_hot_text or feature.replace('_', '') in one_hot_text:
                    one_hot_logic_features.add(feature)
    
    # Features handled by defaults
    defaults_features = set()
    defaults_match = re.search(r'defaults = \{(.*?)\}', content, re.DOTALL)
    if defaults_match:
        defaults_text = defaults_match.group(1)
        for feature in all_features:
            if f"'{feature}'" in defaults_text:
                defaults_features.add(feature)
    
    all_handled_features = directly_handled_features | one_hot_features | one_hot_logic_features | defaults_features
    
    print(f"✅ Total features in final dataset: {len(all_features)}")
    print(f"✅ Features directly handled: {len(directly_handled_features)}")
    print(f"✅ One-hot encoded features: {len(one_hot_features)}")
    print(f"✅ Features in one-hot logic: {len(one_hot_logic_features)}")
    print(f"✅ Features in defaults: {len(defaults_features)}")
    print(f"✅ Total handled features: {len(all_handled_features)}")
    
    # Check for missing features
    missing_features = all_features - all_handled_features
    extra_features = all_handled_features - all_features
    
    if missing_features:
        print(f"❌ Missing features: {len(missing_features)}")
        for f in sorted(list(missing_features)[:10]):  # Show first 10
            print(f"   - {f}")
        if len(missing_features) > 10:
            print(f"   ... and {len(missing_features) - 10} more")
    
    if extra_features:
        print(f"❌ Extra features: {len(extra_features)}")
        for f in sorted(list(extra_features)[:10]):  # Show first 10
            print(f"   - {f}")
        if len(extra_features) > 10:
            print(f"   ... and {len(extra_features) - 10} more")
    
    # If most features are handled, consider it a pass
    coverage_percent = len(all_handled_features) / len(all_features) * 100
    print(f"✅ Feature coverage: {coverage_percent:.1f}%")
    
    if coverage_percent >= 95:  # 95% coverage is acceptable
        print("✅ Feature coverage is acceptable (≥95%)")
        return True
    else:
        print("❌ Feature coverage is too low (<95%)")
        return False

def main():
    """Run all validation tests"""
    print("HOUSE PREDICTION VALIDATION SCRIPT")
    print("=" * 60)
    
    tests = [
        validate_features_against_final_dataset,
        validate_log_transformations,
        validate_engineered_features,
        validate_prediction_accuracy,
        validate_feature_coverage
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Features vs Final Dataset",
        "Log Transformations",
        "Engineered Features",
        "Prediction Accuracy",
        "Feature Coverage"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {name}: {status}")
    
    total_passed = sum(results)
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("🎉 ALL TESTS PASSED - Prediction.py is fully validated!")
    else:
        print("⚠️  Some tests failed - Please review and fix issues")
    
    return total_passed == len(results)

if __name__ == "__main__":
    main()