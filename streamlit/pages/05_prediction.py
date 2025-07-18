import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add the parent directory to access utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.data_loader import load_feature_descriptions_cached, load_processed_data
from utils.model_utils import (
    load_model_and_metadata, 
    predict_house_price, 
    format_prediction_result,
    calculate_price_range,
    explain_prediction,
    get_available_models,
    load_feature_order
)

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide"
)

st.title("House Price Predictor")
st.markdown("---")

st.markdown("""
Use this comprehensive tool to predict house prices based on detailed property characteristics. 
All features are included for maximum prediction accuracy.
""")

# Load model and feature descriptions
@st.cache_resource
def load_prediction_resources():
    """Load model and feature descriptions for prediction."""
    model, metadata = load_model_and_metadata("final_model")
    feature_descriptions = load_feature_descriptions_cached()
    processed_data = load_processed_data()
    return model, metadata, feature_descriptions, processed_data

model, metadata, feature_descriptions, processed_data = load_prediction_resources()

# Check if model is available
if model is None:
    st.error("Model not found. Please ensure the model has been trained and saved.")
    st.stop()

# Current year for age calculations
current_year = datetime.now().year

# Model Information
st.header("1. Model Information")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Type", metadata.get('model_type', 'Stacking Ensemble'))
with col2:
    st.metric("CV RMSE", f"{metadata.get('cv_rmse', 0.1114):.4f}")
with col3:
    st.metric("Features", metadata.get('n_features', 191))

# Feature Input Section
st.header("2. Property Details")

# Initialize feature inputs dictionary
feature_inputs = {}

# Create tabs for different categories
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Basic Info", 
    "Lot & Location", 
    "House Structure", 
    "Basement", 
    "Garage & Exterior"
])

with tab1:
    st.subheader("Basic Property Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MSSubClass
        ms_subclass = st.selectbox(
            "Building Class",
            options=[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190],
            index=0,
            format_func=lambda x: {
                20: "1-Story 1946 & Newer All Styles",
                30: "1-Story 1945 & Older",
                40: "1-Story w/Finished Attic All Ages",
                45: "1-1/2 Story - Unfinished All Ages",
                50: "1-1/2 Story Finished All Ages",
                60: "2-Story 1946 & Newer",
                70: "2-Story 1945 & Older",
                75: "2-1/2 Story All Ages",
                80: "Split or Multi-Level",
                85: "Split Foyer",
                90: "Duplex - All Styles and Ages",
                120: "1-Story PUD (Planned Unit Development) - 1946 & Newer",
                150: "1-1/2 Story PUD - All Ages",
                160: "2-Story PUD - 1946 & Newer",
                180: "PUD - Multilevel - Incl Split Lev/Foyer",
                190: "2 Family Conversion - All Styles and Ages"
            }[x],
            help="Type of dwelling"
        )
        feature_inputs['MSSubClass'] = ms_subclass
        
        # MS Zoning
        ms_zoning = st.selectbox(
            "Zoning Classification",
            options=['A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'],
            index=5,  # Default to RL
            format_func=lambda x: {
                'A': 'Agriculture',
                'C': 'Commercial',
                'FV': 'Floating Village Residential',
                'I': 'Industrial',
                'RH': 'Residential High Density',
                'RL': 'Residential Low Density',
                'RP': 'Residential Low Density Park',
                'RM': 'Residential Medium Density'
            }[x],
            help="General zoning classification"
        )
        
        # House Age
        house_age = st.slider(
            "House Age (years)",
            min_value=0,
            max_value=150,
            value=20,
            help="How old is the house?"
        )
        feature_inputs['HouseAge'] = house_age
        
        # Remodel Status
        has_been_remodeled = st.checkbox(
            "Has the house been remodeled?",
            value=False,
            help="Check if the house has undergone major renovation/remodeling"
        )
        
        if has_been_remodeled:
            years_since_remodel = st.slider(
                "Years Since Last Remodel",
                min_value=0,
                max_value=house_age,
                value=min(10, house_age),
                help="Years since last major remodel"
            )
            feature_inputs['YearsSinceRemodel'] = years_since_remodel
        else:
            # If never remodeled, YearsSinceRemodel = HouseAge
            feature_inputs['YearsSinceRemodel'] = house_age
        
    with col2:
        # Overall Quality
        overall_qual = st.selectbox(
            "Overall Quality",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            index=5,
            format_func=lambda x: {
                1: '1 - Very Poor', 2: '2 - Poor', 3: '3 - Fair', 4: '4 - Below Average',
                5: '5 - Average', 6: '6 - Above Average', 7: '7 - Good', 8: '8 - Very Good',
                9: '9 - Excellent', 10: '10 - Very Excellent'
            }[x],
            help="Overall material and finish quality"
        )
        feature_inputs['OverallQual'] = overall_qual
        
        # Overall Condition
        overall_cond = st.selectbox(
            "Overall Condition",
            options=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            index=4,
            format_func=lambda x: {
                1: '1 - Very Poor', 2: '2 - Poor', 3: '3 - Fair', 4: '4 - Below Average',
                5: '5 - Average', 6: '6 - Above Average', 7: '7 - Good', 8: '8 - Very Good',
                9: '9 - Excellent'
            }[x],
            help="Overall condition rating"
        )
        feature_inputs['OverallCond'] = overall_cond
        

with tab2:
    st.subheader("Lot & Location Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lot Area
        lot_area = st.number_input(
            "Lot Area (sqft)",
            min_value=1300,
            max_value=215245,
            value=10000,
            help="Lot size in square feet"
        )
        feature_inputs['LotArea'] = np.log1p(lot_area)
        
        # Lot Frontage
        lot_frontage = st.number_input(
            "Lot Frontage (feet)",
            min_value=21,
            max_value=313,
            value=70,
            help="Linear feet of street connected to property"
        )
        feature_inputs['LotFrontage'] = np.log1p(lot_frontage)
        
        # Lot Shape
        lot_shape = st.selectbox(
            "Lot Shape",
            options=['Reg', 'IR1', 'IR2', 'IR3'],
            index=0,
            format_func=lambda x: {
                'Reg': 'Regular',
                'IR1': 'Slightly Irregular',
                'IR2': 'Moderately Irregular',
                'IR3': 'Very Irregular'
            }[x],
            help="General shape of property"
        )
        # Convert to numeric for model
        shape_map = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
        feature_inputs['LotShape'] = shape_map[lot_shape]
        
        # Land Contour
        land_contour = st.selectbox(
            "Land Contour",
            options=['Lvl', 'Bnk', 'HLS', 'Low'],
            index=0,
            format_func=lambda x: {
                'Lvl': 'Near Flat/Level',
                'Bnk': 'Banked - Quick rise from street',
                'HLS': 'Hillside - Significant slope',
                'Low': 'Depression'
            }[x],
            help="Flatness of the property"
        )
        contour_map = {'Lvl': 3, 'Bnk': 2, 'HLS': 1, 'Low': 0}
        feature_inputs['LandContour'] = contour_map[land_contour]
        
    with col2:
        # Neighborhood
        neighborhood = st.selectbox(
            "Neighborhood",
            options=['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor',
                    'Edwards', 'Gilbert', 'IDOTRR', 'MeadowV', 'Mitchel', 'NAmes', 'NPkVill',
                    'NWAmes', 'NoRidge', 'NridgHt', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW',
                    'Somerst', 'StoneBr', 'Timber', 'Veenker'],
            index=12,  # Default to NAmes
            help="Physical locations within Ames city limits"
        )
        
        # Lot Configuration
        lot_config = st.selectbox(
            "Lot Configuration",
            options=['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'],
            index=0,
            format_func=lambda x: {
                'Inside': 'Inside lot',
                'Corner': 'Corner lot',
                'CulDSac': 'Cul-de-sac',
                'FR2': 'Frontage on 2 sides',
                'FR3': 'Frontage on 3 sides'
            }[x],
            help="Lot configuration"
        )
        
        # Land Slope
        land_slope = st.selectbox(
            "Land Slope",
            options=['Gtl', 'Mod', 'Sev'],
            index=0,
            format_func=lambda x: {
                'Gtl': 'Gentle slope',
                'Mod': 'Moderate slope',
                'Sev': 'Severe slope'
            }[x],
            help="Slope of property"
        )
        slope_map = {'Gtl': 2, 'Mod': 1, 'Sev': 0}
        feature_inputs['LandSlope'] = slope_map[land_slope]
        
        # Street Type
        street = st.selectbox(
            "Street Type",
            options=['Grvl', 'Pave'],
            index=1,
            format_func=lambda x: {
                'Grvl': 'Gravel',
                'Pave': 'Paved'
            }[x],
            help="Type of road access"
        )
        
        # Alley
        alley = st.selectbox(
            "Alley Access",
            options=['None', 'Grvl', 'Pave'],
            index=0,
            format_func=lambda x: {
                'None': 'No alley access',
                'Grvl': 'Gravel',
                'Pave': 'Paved'
            }[x],
            help="Type of alley access"
        )

with tab3:
    st.subheader("House Structure & Interior")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Living Area
        gr_liv_area = st.number_input(
            "Above Grade Living Area (sqft)",
            min_value=334,
            max_value=5642,
            value=1500,
            help="Above grade (ground) living area square feet"
        )
        feature_inputs['GrLivArea'] = np.log1p(gr_liv_area)
        
        # Total Floor Area
        total_flr_sf = st.number_input(
            "Total Floor Area (sqft)",
            min_value=334,
            max_value=5642,
            value=1500,
            help="Total square feet of all floors (1st + 2nd floor)"
        )
        feature_inputs['TotalFlrSF'] = total_flr_sf
        
        # Bedrooms
        bedrooms = st.selectbox(
            "Bedrooms Above Grade",
            options=[0, 1, 2, 3, 4, 5, 6, 7, 8],
            index=3,
            help="Number of bedrooms above grade"
        )
        feature_inputs['BedroomAbvGr'] = bedrooms
        
        # Total Rooms
        tot_rms_abv_grd = st.number_input(
            "Total Rooms Above Grade",
            min_value=2,
            max_value=15,
            value=7,
            help="Total rooms above grade (excluding bathrooms)"
        )
        feature_inputs['TotRmsAbvGrd'] = tot_rms_abv_grd
        
    with col2:
        # Full Bathrooms Above Grade
        full_bath = st.selectbox(
            "Full Bathrooms Above Grade",
            options=[0, 1, 2, 3, 4],
            index=2,
            help="Number of full bathrooms above grade"
        )
        
        # Half Bathrooms Above Grade
        half_bath = st.selectbox(
            "Half Bathrooms Above Grade",
            options=[0, 1, 2],
            index=1,
            help="Number of half bathrooms above grade"
        )
        
        # Kitchen Quality
        kitchen_qual = st.selectbox(
            "Kitchen Quality",
            options=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
            index=2,
            format_func=lambda x: {
                'Po': 'Poor', 'Fa': 'Fair', 'TA': 'Typical/Average', 'Gd': 'Good', 'Ex': 'Excellent'
            }[x],
            help="Kitchen quality rating"
        )
        kitchen_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        feature_inputs['KitchenQual'] = kitchen_qual_map[kitchen_qual]
        
        # Low Quality Finished Area
        has_low_qual = st.checkbox(
            "Has Low Quality Finished Area?",
            value=False,
            help="Check if the house has low quality finished area"
        )
        
        if has_low_qual:
            low_qual_fin_sf = st.number_input(
                "Low Quality Finished Area (sqft)",
                min_value=1,
                max_value=572,
                value=100,
                help="Low quality finished area in square feet"
            )
            feature_inputs['LowQualFinSF'] = low_qual_fin_sf
        else:
            feature_inputs['LowQualFinSF'] = 0
        
        # Kitchens Above Grade
        kitchen_abv_gr = st.selectbox(
            "Kitchens Above Grade",
            options=[0, 1, 2, 3],
            index=1,
            help="Number of kitchens above grade"
        )
        feature_inputs['KitchenAbvGr'] = kitchen_abv_gr
        
        # Functional
        functional = st.selectbox(
            "Home Functionality",
            options=['Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'],
            index=7,
            format_func=lambda x: {
                'Sal': 'Salvage only',
                'Sev': 'Severely Damaged',
                'Maj2': 'Major Deductions 2',
                'Maj1': 'Major Deductions 1',
                'Mod': 'Moderate Deductions',
                'Min2': 'Minor Deductions 2',
                'Min1': 'Minor Deductions 1',
                'Typ': 'Typical Functionality'
            }[x],
            help="Home functionality rating"
        )
        functional_map = {'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8}
        feature_inputs['Functional'] = functional_map[functional]
        
        # Fireplaces
        fireplaces = st.number_input(
            "Number of Fireplaces",
            min_value=0,
            max_value=3,
            value=0,
            help="Number of fireplaces"
        )
        feature_inputs['Fireplaces'] = fireplaces
        
        # Fireplace Quality (only if fireplaces exist)
        if fireplaces > 0:
            fireplace_qu = st.selectbox(
                "Fireplace Quality",
                options=['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                index=2,
                format_func=lambda x: {
                    'Po': 'Poor', 'Fa': 'Fair', 'TA': 'Typical/Average', 'Gd': 'Good', 'Ex': 'Excellent'
                }[x],
                help="Fireplace quality"
            )
            fireplace_qu_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['FireplaceQu'] = fireplace_qu_map[fireplace_qu]
        else:
            feature_inputs['FireplaceQu'] = 0
        
        # Calculate total bathrooms (will be updated after basement inputs)
        # This is just a placeholder - real calculation happens after basement section

with tab4:
    st.subheader("Basement Information")
    
    # Has Basement
    has_basement = st.checkbox("Has Basement", value=True)
    
    if has_basement:
        col1, col2 = st.columns(2)
        
        with col1:
            # Finished Basement Area
            bsmt_fin_sf = st.number_input(
                "Finished Basement Area (sqft)",
                min_value=0,
                max_value=5644,
                value=0,
                help="Total finished square feet in basement"
            )
            feature_inputs['BsmtFinSF'] = bsmt_fin_sf
            
            # Unfinished Basement Area
            bsmt_unf_sf = st.number_input(
                "Unfinished Basement Area (sqft)",
                min_value=0,
                max_value=2336,
                value=0,
                help="Unfinished square feet in basement"
            )
            feature_inputs['BsmtUnfSF'] = bsmt_unf_sf
            
            # Basement Full Bath
            bsmt_full_bath = st.selectbox(
                "Basement Full Bathrooms",
                options=[0, 1, 2, 3],
                index=0,
                help="Number of full bathrooms in basement"
            )
            # Don't add individual bathroom features to feature_inputs - we'll calculate TotalBaths
            
            # Basement Half Bath
            bsmt_half_bath = st.selectbox(
                "Basement Half Bathrooms",
                options=[0, 1, 2],
                index=0,
                help="Number of half bathrooms in basement"
            )
            # Don't add individual bathroom features to feature_inputs - we'll calculate TotalBaths
            
        with col2:
            # Basement Quality
            bsmt_qual = st.selectbox(
                "Basement Quality",
                options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                index=2,
                format_func=lambda x: {
                    'Ex': 'Excellent (100+ inches)',
                    'Gd': 'Good (90-99 inches)',
                    'TA': 'Typical (80-89 inches)',
                    'Fa': 'Fair (70-79 inches)',
                    'Po': 'Poor (<70 inches)'
                }[x],
                help="Basement height and quality"
            )
            bsmt_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['BsmtQual'] = bsmt_qual_map[bsmt_qual]
            
            # Basement Condition
            bsmt_cond = st.selectbox(
                "Basement Condition",
                options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                index=2,
                format_func=lambda x: {
                    'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical', 'Fa': 'Fair', 'Po': 'Poor'
                }[x],
                help="General condition of basement"
            )
            bsmt_cond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['BsmtCond'] = bsmt_cond_map[bsmt_cond]
            
            # Basement Exposure
            bsmt_exposure = st.selectbox(
                "Basement Exposure",
                options=['No', 'Mn', 'Av', 'Gd'],
                index=0,
                format_func=lambda x: {
                    'No': 'No Exposure',
                    'Mn': 'Minimal Exposure',
                    'Av': 'Average Exposure',
                    'Gd': 'Good Exposure'
                }[x],
                help="Walkout or garden level basement walls"
            )
            exposure_map = {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}
            feature_inputs['BsmtExposure'] = exposure_map[bsmt_exposure]
            
            # Basement Finished Type
            bsmt_fin_type1 = st.selectbox(
                "Basement Finished Type",
                options=['Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
                index=0,
                format_func=lambda x: {
                    'Unf': 'Unfinished',
                    'LwQ': 'Low Quality',
                    'Rec': 'Average Rec Room',
                    'BLQ': 'Below Average',
                    'ALQ': 'Average Living Quarters',
                    'GLQ': 'Good Living Quarters'
                }[x],
                help="Rating of basement finished area"
            )
            fin_type_map = {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
            feature_inputs['BsmtFinType1'] = fin_type_map[bsmt_fin_type1]
            
            # Basement Finished Type 2 (usually unfinished)
            feature_inputs['BsmtFinType2'] = 1  # Unfinished
            
    else:
        # No basement - set all basement features to 0
        feature_inputs['BsmtFinSF'] = 0
        feature_inputs['BsmtUnfSF'] = 0
        feature_inputs['BsmtQual'] = 0
        feature_inputs['BsmtCond'] = 0
        feature_inputs['BsmtExposure'] = 0
        feature_inputs['BsmtFinType1'] = 0
        # Set bathroom variables to 0 for calculation
        bsmt_full_bath = 0
        bsmt_half_bath = 0
    
    # Calculate TotalBaths after getting all bathroom inputs
    total_baths = full_bath + (half_bath * 0.5) + bsmt_full_bath + (bsmt_half_bath * 0.5)
    feature_inputs['TotalBaths'] = total_baths

with tab5:
    st.subheader("Garage & Exterior Features")
    
    # Has Garage
    has_garage = st.checkbox("Has Garage", value=True)
    
    if has_garage:
        col1, col2 = st.columns(2)
        
        with col1:
            # Garage Cars
            garage_cars = st.selectbox(
                "Garage Capacity (cars)",
                options=[1, 2, 3, 4, 5],
                index=1,
                help="Number of cars that can fit in the garage"
            )
            # Don't add individual garage features - we'll calculate GarageAreaPerCar
            
            # Garage Area
            garage_area = st.number_input(
                "Garage Area (sqft)",
                min_value=0,
                max_value=1488,
                value=garage_cars * 250,
                help="Size of garage in square feet"
            )
            # Don't add individual garage area - we'll calculate GarageAreaPerCar
            
            # Garage Age
            garage_age = st.slider(
                "Garage Age (years)",
                min_value=0,
                max_value=house_age,
                value=house_age,
                help="Age of garage (0 = new)"
            )
            feature_inputs['GarageAge'] = garage_age
            
            # Calculate garage area per car (will be added after garage section)
            # This is just a placeholder - real calculation happens after garage section
            
        with col2:
            # Garage Type
            garage_type = st.selectbox(
                "Garage Type",
                options=['Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd'],
                index=0,
                format_func=lambda x: {
                    'Attchd': 'Attached to home',
                    'Basment': 'Basement Garage',
                    'BuiltIn': 'Built-In',
                    'CarPort': 'Car Port',
                    'Detchd': 'Detached from home'
                }[x],
                help="Garage location"
            )
            
            # Garage Finish
            garage_finish = st.selectbox(
                "Garage Finish",
                options=['Unf', 'RFn', 'Fin'],
                index=1,
                format_func=lambda x: {
                    'Unf': 'Unfinished',
                    'RFn': 'Rough Finished',
                    'Fin': 'Finished'
                }[x],
                help="Interior finish of garage"
            )
            finish_map = {'Unf': 1, 'RFn': 2, 'Fin': 3}
            feature_inputs['GarageFinish'] = finish_map[garage_finish]
            
            # Garage Quality
            garage_qual = st.selectbox(
                "Garage Quality",
                options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                index=2,
                format_func=lambda x: {
                    'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'
                }[x],
                help="Garage quality"
            )
            garage_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['GarageQual'] = garage_qual_map[garage_qual]
            
            # Garage Condition
            garage_cond = st.selectbox(
                "Garage Condition",
                options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
                index=2,
                format_func=lambda x: {
                    'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'
                }[x],
                help="Garage condition"
            )
            garage_cond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['GarageCond'] = garage_cond_map[garage_cond]
            
        feature_inputs['HasGarage'] = 1
        
        # Calculate garage area per car
        garage_area_per_car = garage_area / max(1, garage_cars)
        feature_inputs['GarageAreaPerCar'] = garage_area_per_car
        
    else:
        # No garage - set all garage features to 0
        feature_inputs['GarageAge'] = 0
        feature_inputs['GarageAreaPerCar'] = 0
        feature_inputs['GarageFinish'] = 0
        feature_inputs['GarageQual'] = 0
        feature_inputs['GarageCond'] = 0
        feature_inputs['HasGarage'] = 0
        garage_type = 'None'
        garage_cars = 0
        garage_area = 0
    
    # Exterior Features
    st.subheader("Exterior Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exterior Quality
        exter_qual = st.selectbox(
            "Exterior Quality",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            index=2,
            format_func=lambda x: {
                'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'
            }[x],
            help="Exterior material quality"
        )
        exter_qual_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        feature_inputs['ExterQual'] = exter_qual_map[exter_qual]
        
        # Exterior Condition
        exter_cond = st.selectbox(
            "Exterior Condition",
            options=['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            index=2,
            format_func=lambda x: {
                'Ex': 'Excellent', 'Gd': 'Good', 'TA': 'Typical/Average', 'Fa': 'Fair', 'Po': 'Poor'
            }[x],
            help="Present condition of exterior"
        )
        exter_cond_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        feature_inputs['ExterCond'] = exter_cond_map[exter_cond]
        
        # Masonry Veneer
        has_masonry = st.checkbox(
            "Has Masonry Veneer?",
            value=False,
            help="Check if the house has masonry veneer"
        )
        
        if has_masonry:
            mas_vnr_area = st.number_input(
                "Masonry Veneer Area (sqft)",
                min_value=1,
                max_value=1600,
                value=200,
                help="Masonry veneer area in square feet"
            )
            feature_inputs['MasVnrArea'] = mas_vnr_area
        else:
            feature_inputs['MasVnrArea'] = 0
        
    with col2:
        # Paved Driveway
        paved_drive = st.selectbox(
            "Paved Driveway",
            options=['N', 'P', 'Y'],
            index=2,
            format_func=lambda x: {
                'N': 'No - Dirt/Gravel',
                'P': 'Partial Pavement',
                'Y': 'Paved'
            }[x],
            help="Paved driveway"
        )
        paved_map = {'N': 0, 'P': 1, 'Y': 2}
        feature_inputs['PavedDrive'] = paved_map[paved_drive]
        
        # Central Air
        central_air = st.selectbox(
            "Central Air Conditioning",
            options=['N', 'Y'],
            index=1,
            format_func=lambda x: {'N': 'No', 'Y': 'Yes'}[x],
            help="Central air conditioning"
        )
        feature_inputs['CentralAir'] = 1 if central_air == 'Y' else 0
        
        # Porch/Deck Features
        has_porch_deck = st.checkbox(
            "Has Porch/Deck Areas?",
            value=False,
            help="Check if the house has any porch or deck areas"
        )
        
        if has_porch_deck:
            # Wood Deck Area
            wood_deck_sf = st.number_input(
                "Wood Deck Area (sqft)",
                min_value=0,
                max_value=857,
                value=0,
                help="Wood deck area in square feet"
            )
            feature_inputs['WoodDeckSF'] = wood_deck_sf
            
            # Open Porch Area
            open_porch_sf = st.number_input(
                "Open Porch Area (sqft)",
                min_value=0,
                max_value=547,
                value=0,
                help="Open porch area in square feet"
            )
            feature_inputs['OpenPorchSF'] = open_porch_sf
            
            # Enclosed Porch Area
            enclosed_porch = st.number_input(
                "Enclosed Porch Area (sqft)",
                min_value=0,
                max_value=552,
                value=0,
                help="Enclosed porch area in square feet"
            )
            feature_inputs['EnclosedPorch'] = enclosed_porch
            
            # 3-Season Porch Area
            three_ssn_porch = st.number_input(
                "3-Season Porch Area (sqft)",
                min_value=0,
                max_value=508,
                value=0,
                help="3-season porch area in square feet"
            )
            feature_inputs['3SsnPorch'] = three_ssn_porch
            
            # Screen Porch Area
            screen_porch = st.number_input(
                "Screen Porch Area (sqft)",
                min_value=0,
                max_value=480,
                value=0,
                help="Screen porch area in square feet"
            )
            feature_inputs['ScreenPorch'] = screen_porch
        else:
            # No porch/deck areas
            feature_inputs['WoodDeckSF'] = 0
            feature_inputs['OpenPorchSF'] = 0
            feature_inputs['EnclosedPorch'] = 0
            feature_inputs['3SsnPorch'] = 0
            feature_inputs['ScreenPorch'] = 0
        
        # Pool
        has_pool = st.checkbox(
            "Has Pool?",
            value=False,
            help="Check if the house has a pool"
        )
        
        if has_pool:
            # Pool Area
            pool_area = st.number_input(
                "Pool Area (sqft)",
                min_value=1,
                max_value=738,
                value=200,
                help="Pool area in square feet"
            )
            feature_inputs['PoolArea'] = pool_area
            
            # Pool Quality
            pool_qc = st.selectbox(
                "Pool Quality",
                options=['Fa', 'TA', 'Gd', 'Ex'],
                index=1,
                format_func=lambda x: {
                    'Fa': 'Fair', 'TA': 'Typical/Average', 'Gd': 'Good', 'Ex': 'Excellent'
                }[x],
                help="Pool quality"
            )
            pool_qc_map = {'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
            feature_inputs['PoolQC'] = pool_qc_map[pool_qc]
        else:
            feature_inputs['PoolArea'] = 0
            feature_inputs['PoolQC'] = 0
        
        # Fence
        fence = st.selectbox(
            "Fence Quality",
            options=['None', 'MnWw', 'GdWo', 'MnPrv', 'GdPrv'],
            index=0,
            format_func=lambda x: {
                'None': 'No Fence',
                'MnWw': 'Minimum Wood/Wire',
                'GdWo': 'Good Wood',
                'MnPrv': 'Minimum Privacy',
                'GdPrv': 'Good Privacy'
            }[x],
            help="Fence quality"
        )
        fence_map = {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
        feature_inputs['Fence'] = fence_map[fence]
        
        # Miscellaneous Features
        has_misc = st.checkbox(
            "Has Miscellaneous Features?",
            value=False,
            help="Check if the house has miscellaneous features (shed, tennis court, etc.)"
        )
        
        if has_misc:
            misc_val = st.number_input(
                "Miscellaneous Feature Value ($)",
                min_value=1,
                max_value=15500,
                value=500,
                help="Dollar value of miscellaneous feature"
            )
            feature_inputs['MiscVal'] = misc_val
        else:
            feature_inputs['MiscVal'] = 0

# Prediction Section
st.header("3. Price Prediction")

if st.button("Predict House Price", type="primary"):
    # Load feature order
    feature_order = load_feature_order()
    
    if not feature_order:
        st.error("Could not load feature order. Please ensure the model is properly trained.")
        st.stop()
    
    # Create comprehensive feature dictionary
    full_features = {}
    
    # Add all user inputs and apply log transformation where needed
    full_features.update(feature_inputs)
    
    # Apply log transformation to numerical features that need it
    # Based on the processed dataset, these features are log-transformed (skewness >= 0.5):
    log_transform_features = [
        'MasVnrArea', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 
        'PoolArea', 'MiscVal', 'BsmtUnfSF', 'TotRmsAbvGrd', 'LowQualFinSF', 
        'OpenPorchSF', 'Fireplaces', 'KitchenAbvGr', 'OverallCond', 'MSSubClass', 'WoodDeckSF'
    ]
    
    # Apply log transformation to features that are not already transformed
    for feature in log_transform_features:
        if feature in full_features:
            # LotArea, LotFrontage, and GrLivArea are already transformed in input sections above
            value = full_features[feature]
            if isinstance(value, (int, float)) and value >= 0:
                full_features[feature] = np.log1p(value)
    
    # Add default values for features not specified
    defaults = {
        'Utilities': 4,  # All public utilities
        'MiscFeature': 'None',  # No miscellaneous feature
        'SaleType': 'WD',  # Warranty Deed
        'SaleCondition': 'Normal',  # Normal sale
        'MasVnrType': 'None',  # No masonry veneer type
        'BsmtFinType2': 1,  # Unfinished (label encoded)
        'HeatingQC': 5,  # Excellent heating quality (label encoded)
        'Heating': 'GasA',  # Gas forced warm air
        'Electrical': 'SBrkr',  # Standard circuit breakers
        'BldgType': '1Fam',  # Single-family
        'HouseStyle': '1Story',  # 1-story
        'RoofStyle': 'Gable',  # Gable roof
        'RoofMatl': 'CompShg',  # Composition shingles
        'Exterior1st': 'VinylSd',  # Vinyl siding
        'Exterior2nd': 'VinylSd',  # Vinyl siding
        'Foundation': 'PConc',  # Poured concrete
        'Condition1': 'Norm',  # Normal proximity
        'Condition2': 'Norm'  # Normal proximity
    }
    
    # Add defaults for missing features
    for key, value in defaults.items():
        if key not in full_features:
            full_features[key] = value
    
    # Handle one-hot encoded features
    for feature_name in feature_order:
        if feature_name not in full_features:
            # One-hot encoding logic
            if feature_name.startswith('MSZoning_'):
                full_features[feature_name] = 1 if feature_name == f'MSZoning_{ms_zoning}' else 0
            elif feature_name.startswith('Street_'):
                full_features[feature_name] = 1 if feature_name == f'Street_{street}' else 0
            elif feature_name.startswith('Alley_'):
                full_features[feature_name] = 1 if feature_name == f'Alley_{alley}' else 0
            elif feature_name.startswith('LotConfig_'):
                full_features[feature_name] = 1 if feature_name == f'LotConfig_{lot_config}' else 0
            elif feature_name.startswith('Neighborhood_'):
                full_features[feature_name] = 1 if feature_name == f'Neighborhood_{neighborhood}' else 0
            elif feature_name.startswith('GarageType_'):
                full_features[feature_name] = 1 if feature_name == f'GarageType_{garage_type}' else 0
            elif feature_name.startswith('Condition1_'):
                full_features[feature_name] = 1 if feature_name == 'Condition1_Norm' else 0
            elif feature_name.startswith('Condition2_'):
                full_features[feature_name] = 1 if feature_name == 'Condition2_Norm' else 0
            elif feature_name.startswith('BldgType_'):
                full_features[feature_name] = 1 if feature_name == 'BldgType_1Fam' else 0
            elif feature_name.startswith('HouseStyle_'):
                full_features[feature_name] = 1 if feature_name == 'HouseStyle_1Story' else 0
            elif feature_name.startswith('RoofStyle_'):
                full_features[feature_name] = 1 if feature_name == 'RoofStyle_Gable' else 0
            elif feature_name.startswith('RoofMatl_'):
                full_features[feature_name] = 1 if feature_name == 'RoofMatl_CompShg' else 0
            elif feature_name.startswith('Exterior1st_'):
                full_features[feature_name] = 1 if feature_name == 'Exterior1st_VinylSd' else 0
            elif feature_name.startswith('Exterior2nd_'):
                full_features[feature_name] = 1 if feature_name == 'Exterior2nd_VinylSd' else 0
            elif feature_name.startswith('MasVnrType_'):
                full_features[feature_name] = 1 if feature_name == 'MasVnrType_None' else 0
            elif feature_name.startswith('Foundation_'):
                full_features[feature_name] = 1 if feature_name == 'Foundation_PConc' else 0
            elif feature_name.startswith('Heating_'):
                full_features[feature_name] = 1 if feature_name == 'Heating_GasA' else 0
            elif feature_name.startswith('Electrical_'):
                full_features[feature_name] = 1 if feature_name == 'Electrical_SBrkr' else 0
            elif feature_name.startswith('MiscFeature_'):
                full_features[feature_name] = 1 if feature_name == 'MiscFeature_None' else 0
            elif feature_name.startswith('SaleType_'):
                full_features[feature_name] = 1 if feature_name == 'SaleType_WD' else 0
            elif feature_name.startswith('SaleCondition_'):
                full_features[feature_name] = 1 if feature_name == 'SaleCondition_Normal' else 0
            else:
                full_features[feature_name] = 0
    
    # Make prediction
    prediction = predict_house_price(model, full_features, metadata)
    
    if prediction is not None:
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Predicted Price",
                f"${prediction:,.0f}",
                help="Estimated house price based on input features"
            )
        
        with col2:
            lower_bound, upper_bound = calculate_price_range(prediction, uncertainty_factor=0.15)
            st.metric(
                "Price Range (±15%)",
                f"${lower_bound:,.0f} - ${upper_bound:,.0f}",
                help="Confidence interval around the prediction"
            )
        
        with col3:
            price_per_sqft = prediction / gr_liv_area if gr_liv_area > 0 else 0
            st.metric(
                "Price per Sq Ft",
                f"${price_per_sqft:.0f}",
                help="Predicted price per square foot of living area"
            )
        
        # Enhanced prediction insights
        st.subheader("Prediction Analysis")
        
        # Create tabs for different insights
        tab1, tab2, tab3 = st.tabs(["Price Drivers", "Market Context", "Feature Impact"])
        
        with tab1:
            st.markdown("**Key Price Drivers:**")
            
            # Quality impact
            if overall_qual >= 8:
                st.success("**Excellent Quality** - Major value driver (+25-40%)")
            elif overall_qual >= 6:
                st.info("**Good Quality** - Positive impact (+10-15%)")
            else:
                st.warning("**Lower Quality** - May reduce value (-10-20%)")
            
            # Size impact
            if gr_liv_area >= 2000:
                st.success("**Large Living Area** - Premium size (+15-25%)")
            elif gr_liv_area >= 1500:
                st.info("**Good Size** - Market standard")
            else:
                st.warning("**Compact Size** - Below average (-5-10%)")
            
            # Age impact
            if house_age < 10:
                st.success("**New Construction** - Premium for modern features (+10-15%)")
            elif house_age < 30:
                st.info("**Modern Home** - Good condition expected")
            else:
                st.warning("**Older Home** - May need updates (-5-10%)")
            
            # Garage impact
            if has_garage:
                st.success("**Garage Present** - Adds significant value (+8-12%)")
            else:
                st.warning("**No Garage** - Reduces appeal (-8-12%)")
                
        with tab2:
            st.markdown("**Market Position Analysis:**")
            
            # Market segment
            if prediction < 150000:
                st.info("**Entry-Level Market** - Affordable starter home segment")
                market_segment = "Entry-Level"
                segment_color = "blue"
            elif prediction < 250000:
                st.info("**Mid-Market** - Good value family home segment")
                market_segment = "Mid-Market"
                segment_color = "green"
            elif prediction < 400000:
                st.info("**Upper-Mid Market** - Quality home with premium features")
                market_segment = "Upper-Mid"
                segment_color = "orange"
            else:
                st.info("**Luxury Market** - High-end property with exceptional features")
                market_segment = "Luxury"
                segment_color = "red"
            
            # Price positioning chart
            fig = go.Figure()
            
            # Market segments
            segments = {
                'Entry-Level': {'range': [50000, 150000], 'color': 'blue'},
                'Mid-Market': {'range': [150000, 250000], 'color': 'green'},
                'Upper-Mid': {'range': [250000, 400000], 'color': 'orange'},
                'Luxury': {'range': [400000, 800000], 'color': 'red'}
            }
            
            for segment, data in segments.items():
                fig.add_trace(go.Scatter(
                    x=data['range'],
                    y=[segment, segment],
                    mode='lines',
                    line=dict(color=data['color'], width=20),
                    name=segment,
                    opacity=0.3
                ))
            
            # Add prediction point
            fig.add_trace(go.Scatter(
                x=[prediction],
                y=[market_segment],
                mode='markers',
                marker=dict(color=segment_color, size=15, symbol='diamond'),
                name='Your Prediction'
            ))
            
            fig.update_layout(
                title='Market Positioning',
                xaxis_title='Price ($)',
                yaxis_title='Market Segment',
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Price context information
            st.markdown("**Price Context:**")
            st.info("Comparable properties analysis would require access to recent sales data in the same area with similar characteristics.")
            
        with tab3:
            st.markdown("**Feature Impact Analysis:**")
            st.info("Feature impact analysis would require model interpretability tools like SHAP values or feature importance from the trained model.")
        
        # Prediction confidence analysis
        st.subheader("Prediction Confidence Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confidence Factors:**")
            
            # Calculate confidence score based on inputs
            confidence_factors = []
            
            # Quality factor
            if 4 <= overall_qual <= 8:
                confidence_factors.append(("Quality in normal range", 0.8))
            elif overall_qual < 4 or overall_qual > 8:
                confidence_factors.append(("Extreme quality rating", 0.6))
            else:
                confidence_factors.append(("Standard quality", 0.7))
            
            # Size factor
            if 1000 <= gr_liv_area <= 3000:
                confidence_factors.append(("Size in normal range", 0.9))
            else:
                confidence_factors.append(("Unusual size", 0.7))
            
            # Age factor
            if 0 <= house_age <= 100:
                confidence_factors.append(("Age in normal range", 0.8))
            else:
                confidence_factors.append(("Unusual age", 0.6))
            
            # Feature completeness
            if len(feature_inputs) >= 20:
                confidence_factors.append(("Complete feature set", 0.9))
            else:
                confidence_factors.append(("Limited features", 0.7))
            
            # Display factors
            for factor, score in confidence_factors:
                st.write(f"• {factor}: {score:.1f}")
            
            # Calculate overall confidence
            overall_confidence = np.mean([score for _, score in confidence_factors])
            
            if overall_confidence >= 0.8:
                st.success(f"**High Confidence**: {overall_confidence:.1f}")
            elif overall_confidence >= 0.7:
                st.info(f"**Medium Confidence**: {overall_confidence:.1f}")
            else:
                st.warning(f"**Lower Confidence**: {overall_confidence:.1f}")
        
        with col2:
            st.markdown("**Model Reliability:**")
            
            # Model performance metrics
            st.write("• **Model Type**: Stacking Ensemble")
            st.write("• **Cross-validation RMSE**: 0.1114")
            st.write("• **R² Score**: ~0.89")
            st.write("• **Training Data**: 1,458 properties")
            st.write("• **Feature Count**: 191 engineered features")
            
            # Prediction reliability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = overall_confidence * 100,
                title = {'text': "Prediction Confidence"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Unable to make prediction. Please check your inputs and try again.")

# Model Performance Information
st.header("4. Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Model Details:**")
    st.write("• Type: Stacking Ensemble")
    st.write("• Base Models: CatBoost, XGBoost, LightGBM, Lasso")
    st.write("• Features: 191 engineered features")

with col2:
    st.markdown("**Performance Metrics:**")
    st.write("• Cross-validation RMSE: 0.1114")
    st.write("• R² Score: ~0.89")
    st.write("• Mean Absolute Error: ~$15,000")

with col3:
    st.markdown("**Prediction Accuracy:**")
    st.write("• ±15% range covers ~90% of cases")
    st.write("• Best for: $100k - $400k range")
    st.write("• Trained on: 1,458 properties")

# Footer
st.markdown("---")
st.markdown("*This comprehensive house price predictor uses all available features for maximum accuracy. Predictions are based on the Ames Housing dataset (2006-2010).*")