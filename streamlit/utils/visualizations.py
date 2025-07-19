import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def create_missing_values_heatmap(df):
    """Create interactive heatmap of missing values."""
    missing_data = df.isnull()
    missing_counts = missing_data.sum()
    missing_features = missing_counts[missing_counts > 0].index.tolist()
    
    if not missing_features:
        return None
    
    # Limit to top 20 features with most missing values for better visualization
    top_missing_features = missing_counts.nlargest(20).index.tolist()
    
    # Sample data for better performance (every 10th row)
    sample_indices = range(0, len(df), 10)
    sample_df = df.iloc[sample_indices]
    
    # Create heatmap data
    heatmap_data = sample_df[top_missing_features].isnull().T
    
    # Convert boolean to int for better visualization
    heatmap_values = heatmap_data.astype(int).values
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_values,
        x=[f"Row {i}" for i in sample_indices],
        y=top_missing_features,
        colorscale=[[0, 'lightblue'], [1, 'red']],
        showscale=True,
        colorbar=dict(title="Missing Values", tickvals=[0, 1], ticktext=["Present", "Missing"]),
        hovertemplate="Feature: %{y}<br>Row: %{x}<br>Missing: %{z}<extra></extra>"
    ))
    
    fig.update_layout(
        title="Missing Values Pattern (Top 20 Features, Sample Data)",
        xaxis_title="Data Points (sampled)",
        yaxis_title="Features",
        height=600,
        width=1000
    )
    
    return fig

def create_missing_values_bar_chart(df):
    """Create bar chart of missing values per feature."""
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)
    
    if missing_counts.empty:
        return None
    
    # Calculate percentages
    missing_pct = (missing_counts / len(df)) * 100
    
    fig = go.Figure(data=[
        go.Bar(
            x=missing_counts.index,
            y=missing_counts.values,
            text=[f"{pct:.1f}%" for pct in missing_pct],
            textposition='auto',
            marker_color='coral'
        )
    ])
    
    fig.update_layout(
        title="Missing Values Count by Feature",
        xaxis_title="Features",
        yaxis_title="Missing Value Count",
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def create_target_distribution_plot(target_series):
    """Create distribution plot for target variable."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original Distribution', 'Box Plot', 'Log Distribution', 'Q-Q Plot'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Original distribution
    fig.add_trace(
        go.Histogram(x=target_series, nbinsx=50, name="Original", marker_color='skyblue'),
        row=1, col=1
    )
    
    # Box plot
    fig.add_trace(
        go.Box(y=target_series, name="Original", marker_color='lightblue'),
        row=1, col=2
    )
    
    # Log distribution
    log_target = np.log1p(target_series)
    fig.add_trace(
        go.Histogram(x=log_target, nbinsx=50, name="Log", marker_color='orange'),
        row=2, col=1
    )
    
    # Q-Q plot
    probplot = stats.probplot(log_target, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=probplot[0][0],
            y=probplot[0][1],
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='red', size=4)
        ),
        row=2, col=2
    )
    
    # Add theoretical line for Q-Q plot
    fig.add_trace(
        go.Scatter(
            x=probplot[0][0],
            y=probplot[1][1] + probplot[1][0] * probplot[0][0],
            mode='lines',
            name='Theoretical',
            line=dict(color='blue', dash='dash')
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Target Variable Distribution Analysis")
    
    return fig

def create_correlation_heatmap(df, target_col=None):
    """Create correlation heatmap with target variable highlighted."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numerical_cols:
        numerical_cols.remove('Id')
    
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 8},
        showscale=True
    ))
    
    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_tickangle=-45,
        height=600,
        width=800
    )
    
    return fig

def create_target_correlation_bar_chart(df, target_col):
    """Create bar chart of feature correlations with target."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numerical_cols:
        numerical_cols.remove('Id')
    
    correlations = df[numerical_cols].corrwith(df[target_col]).drop(target_col).sort_values(key=abs, ascending=False)
    
    colors = ['#1f77b4' if val > 0 else '#d62728' for val in correlations.values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            marker_color=colors,
            text=correlations.round(3).values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Feature Correlations with {target_col}",
        xaxis_title="Correlation",
        yaxis_title="Features",
        height=600
    )
    
    return fig

def create_categorical_boxplot(df, categorical_col, target_col):
    """Create boxplot for categorical feature vs target."""
    fig = px.box(
        df, 
        x=categorical_col, 
        y=target_col,
        title=f'{target_col} Distribution by {categorical_col}'
    )
    
    fig.update_xaxes(tickangle=-45)
    fig.update_layout(height=500)
    
    return fig

def create_scatter_plot(df, x_col, y_col, highlight_outliers=True):
    """Create scatter plot with optional outlier highlighting."""
    fig = go.Figure()
    
    if highlight_outliers:
        # Calculate IQR bounds for outlier detection
        Q1 = df[x_col].quantile(0.25)
        Q3 = df[x_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (df[x_col] < lower_bound) | (df[x_col] > upper_bound)
        normal_points = ~outliers
        
        # Plot normal points
        fig.add_trace(go.Scatter(
            x=df[x_col][normal_points],
            y=df[y_col][normal_points],
            mode='markers',
            name='Normal',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
        
        # Plot outliers
        if outliers.any():
            fig.add_trace(go.Scatter(
                x=df[x_col][outliers],
                y=df[y_col][outliers],
                mode='markers',
                name=f'Outliers ({outliers.sum()})',
                marker=dict(color='red', size=6, opacity=0.8)
            ))
    else:
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            mode='markers',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
    
    fig.update_layout(
        title=f'{x_col} vs {y_col}',
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500
    )
    
    return fig

def create_feature_importance_plot(feature_names, importance_scores, title="Feature Importance"):
    """Create horizontal bar chart for feature importance."""
    # Sort by importance
    sorted_idx = np.argsort(importance_scores)
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance_scores[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation='h',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=600
    )
    
    return fig

def create_model_comparison_plot(results_df):
    """Create model comparison visualization."""
    # Assuming results_df has columns: Model, RMSE, MAE, R2
    fig = go.Figure()
    
    # Add RMSE bars
    fig.add_trace(go.Bar(
        name='RMSE',
        x=results_df['Model'],
        y=results_df['RMSE'],
        marker_color='lightcoral'
    ))
    
    # Add MAE bars
    fig.add_trace(go.Bar(
        name='MAE',
        x=results_df['Model'],
        y=results_df['MAE'],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Error Metrics',
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def create_neighborhood_analysis_plot(df, price_col='SalePrice', neighborhood_col='Neighborhood'):
    """Create neighborhood price analysis visualization."""
    if neighborhood_col not in df.columns or price_col not in df.columns:
        return None
    
    # Calculate neighborhood statistics
    neighborhood_stats = df.groupby(neighborhood_col)[price_col].agg([
        'count', 'mean', 'median', 'std'
    ]).round(0)
    neighborhood_stats.columns = ['Count', 'Mean_Price', 'Median_Price', 'Std_Price']
    neighborhood_stats = neighborhood_stats.sort_values('Mean_Price', ascending=False)
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Price by Neighborhood', 'Price Distribution by Neighborhood (Top 8)', 
                       'Price Variability', 'Sales Volume'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Average price bar chart
    fig.add_trace(
        go.Bar(
            x=neighborhood_stats.index,
            y=neighborhood_stats['Mean_Price'],
            name='Average Price',
            marker_color='steelblue',
            text=neighborhood_stats['Mean_Price'].apply(lambda x: f'${x:,.0f}'),
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # 2. Box plot for top 8 neighborhoods
    top_neighborhoods = neighborhood_stats.head(8).index
    for i, neighborhood in enumerate(top_neighborhoods):
        prices = df[df[neighborhood_col] == neighborhood][price_col]
        fig.add_trace(
            go.Box(
                y=prices,
                name=neighborhood,
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Price variability (coefficient of variation)
    cv = (neighborhood_stats['Std_Price'] / neighborhood_stats['Mean_Price'] * 100).fillna(0)
    fig.add_trace(
        go.Bar(
            x=neighborhood_stats.index,
            y=cv,
            name='Price Variability (%)',
            marker_color='orange',
            text=cv.apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Sales volume
    fig.add_trace(
        go.Bar(
            x=neighborhood_stats.index,
            y=neighborhood_stats['Count'],
            name='Sales Volume',
            marker_color='lightgreen',
            text=neighborhood_stats['Count'],
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Neighborhood Analysis Dashboard",
        showlegend=False
    )
    
    # Update x-axis labels
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=2)
    
    return fig

def create_feature_correlation_network(df, target_col='SalePrice', threshold=0.3):
    """Create network visualization of feature correlations."""
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Id' in numerical_cols:
        numerical_cols.remove('Id')
    
    corr_matrix = df[numerical_cols].corr()
    
    # Get strong correlations with target
    target_correlations = corr_matrix[target_col].abs().sort_values(ascending=False)
    top_features = target_correlations[target_correlations >= threshold].index.tolist()
    
    if len(top_features) < 2:
        return None
    
    # Create network data
    subset_corr = corr_matrix.loc[top_features, top_features]
    
    # Create edges for strong correlations
    edges = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            corr_value = subset_corr.iloc[i, j]
            if abs(corr_value) >= threshold:
                edges.append({
                    'source': top_features[i],
                    'target': top_features[j],
                    'correlation': corr_value
                })
    
    # Create visualization
    fig = go.Figure()
    
    # Add correlation bars
    fig.add_trace(go.Bar(
        x=top_features,
        y=target_correlations[top_features],
        name=f'Correlation with {target_col}',
        marker_color=['red' if x < 0 else 'blue' for x in target_correlations[top_features]]
    ))
    
    fig.update_layout(
        title=f'Top Features Correlated with {target_col} (|r| >= {threshold})',
        xaxis_title='Features',
        yaxis_title='Correlation',
        xaxis_tickangle=-45,
        height=500
    )
    
    return fig

def create_time_series_analysis(df, year_col='YearBuilt', price_col='SalePrice'):
    """Create time series analysis of prices over years."""
    if year_col not in df.columns or price_col not in df.columns:
        return None
    
    # Group by year and calculate statistics
    yearly_stats = df.groupby(year_col)[price_col].agg([
        'count', 'mean', 'median', 'std'
    ]).reset_index()
    yearly_stats.columns = ['Year', 'Count', 'Mean_Price', 'Median_Price', 'Std_Price']
    
    # Filter years with sufficient data
    yearly_stats = yearly_stats[yearly_stats['Count'] >= 5]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average House Price by Year Built', 'Number of Houses Built by Year'),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    
    # Price trend
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['Mean_Price'],
            mode='lines+markers',
            name='Average Price',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ),
        row=1, col=1
    )
    
    # Add median price
    fig.add_trace(
        go.Scatter(
            x=yearly_stats['Year'],
            y=yearly_stats['Median_Price'],
            mode='lines+markers',
            name='Median Price',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Volume of construction
    fig.add_trace(
        go.Bar(
            x=yearly_stats['Year'],
            y=yearly_stats['Count'],
            name='Houses Built',
            marker_color='lightgreen',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        title_text="Housing Market Time Series Analysis"
    )
    
    fig.update_xaxes(title_text="Year Built", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Houses", row=2, col=1)
    
    return fig

def create_quality_vs_price_analysis(df, quality_col='OverallQual', price_col='SalePrice'):
    """Create quality vs price analysis with multiple dimensions."""
    if quality_col not in df.columns or price_col not in df.columns:
        return None
    
    # Create quality bins for better visualization
    quality_stats = df.groupby(quality_col)[price_col].agg([
        'count', 'mean', 'median', 'std'
    ]).reset_index()
    quality_stats.columns = ['Quality', 'Count', 'Mean_Price', 'Median_Price', 'Std_Price']
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price Distribution by Quality', 'Average Price by Quality', 
                       'Quality Distribution', 'Price Variability by Quality'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. Box plot of price by quality
    for quality in sorted(df[quality_col].unique()):
        prices = df[df[quality_col] == quality][price_col]
        fig.add_trace(
            go.Box(
                y=prices,
                name=f'Quality {quality}',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. Average price by quality
    fig.add_trace(
        go.Scatter(
            x=quality_stats['Quality'],
            y=quality_stats['Mean_Price'],
            mode='lines+markers',
            name='Average Price',
            line=dict(color='blue', width=3),
            marker=dict(size=8, color='blue')
        ),
        row=1, col=2
    )
    
    # 3. Quality distribution
    fig.add_trace(
        go.Bar(
            x=quality_stats['Quality'],
            y=quality_stats['Count'],
            name='Count',
            marker_color='lightcoral',
            text=quality_stats['Count'],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    # 4. Price variability (coefficient of variation)
    cv = (quality_stats['Std_Price'] / quality_stats['Mean_Price'] * 100).fillna(0)
    fig.add_trace(
        go.Bar(
            x=quality_stats['Quality'],
            y=cv,
            name='Price Variability (%)',
            marker_color='orange',
            text=cv.apply(lambda x: f'{x:.1f}%'),
            textposition='outside'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text="Quality vs Price Analysis Dashboard",
        showlegend=False
    )
    
    fig.update_xaxes(title_text="Overall Quality", row=1, col=2)
    fig.update_xaxes(title_text="Overall Quality", row=2, col=1)
    fig.update_xaxes(title_text="Overall Quality", row=2, col=2)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Average Price ($)", row=1, col=2)
    fig.update_yaxes(title_text="Number of Houses", row=2, col=1)
    fig.update_yaxes(title_text="Price Variability (%)", row=2, col=2)
    
    return fig

def create_before_after_comparison(before_df, after_df, metric_name="Data Quality"):
    """Create before/after comparison visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{metric_name} - Before', f'{metric_name} - After'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Calculate missing values for both datasets
    before_missing = before_df.isnull().sum()
    after_missing = after_df.isnull().sum()
    
    # Get features that had missing values in the original dataset
    missing_features = before_missing[before_missing > 0].index.tolist()
    
    if missing_features:
        # Before
        fig.add_trace(
            go.Bar(
                x=missing_features,
                y=before_missing[missing_features],
                name='Before',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # After - get actual missing values from processed dataset
        # Some features may still exist in after_df, others may not
        after_values = []
        for feature in missing_features:
            if feature in after_df.columns:
                after_values.append(after_missing[feature])
            else:
                after_values.append(0)  # Feature doesn't exist in processed data
        
        fig.add_trace(
            go.Bar(
                x=missing_features,
                y=after_values,
                name='After',
                marker_color='green',
                opacity=0.7
            ),
            row=1, col=2
        )
    
    fig.update_layout(
        height=400,
        title_text=f"{metric_name} Comparison",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Missing Values", row=1, col=1)
    fig.update_yaxes(title_text="Missing Values", row=1, col=2)
    
    return fig

def create_enhanced_outlier_plot(df, x_col, y_col, outlier_ids=None):
    """Create enhanced outlier visualization with detailed analysis."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Scatter Plot with Outliers', 'Box Plot - X Variable', 
                       'Box Plot - Y Variable', 'Outlier Impact Analysis'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Calculate outliers using IQR method
    Q1_x = df[x_col].quantile(0.25)
    Q3_x = df[x_col].quantile(0.75)
    IQR_x = Q3_x - Q1_x
    
    Q1_y = df[y_col].quantile(0.25)
    Q3_y = df[y_col].quantile(0.75)
    IQR_y = Q3_y - Q1_y
    
    # Define IQR outlier conditions
    iqr_outlier_condition = (
        (df[x_col] < Q1_x - 1.5 * IQR_x) | (df[x_col] > Q3_x + 1.5 * IQR_x) |
        (df[y_col] < Q1_y - 1.5 * IQR_y) | (df[y_col] > Q3_y + 1.5 * IQR_y)
    )
    
    # Define specific outlier conditions (data quality issues)
    specific_outlier_condition = pd.Series([False] * len(df), index=df.index)
    if outlier_ids:
        # Check if 'Id' column exists and use it, otherwise use index
        if 'Id' in df.columns:
            specific_outlier_condition = df['Id'].isin(outlier_ids)
        else:
            specific_outlier_condition = df.index.isin(outlier_ids)
    
    # Create different point categories
    normal_points = ~iqr_outlier_condition & ~specific_outlier_condition
    iqr_only_outliers = iqr_outlier_condition & ~specific_outlier_condition
    specific_outliers = specific_outlier_condition
    
    # 1. Scatter plot
    # Normal points
    fig.add_trace(
        go.Scatter(
            x=df[x_col][normal_points],
            y=df[y_col][normal_points],
            mode='markers',
            name='Normal Points',
            marker=dict(color='blue', size=4, opacity=0.6)
        ),
        row=1, col=1
    )
    
    # IQR outliers (statistical outliers)
    if iqr_only_outliers.any():
        fig.add_trace(
            go.Scatter(
                x=df[x_col][iqr_only_outliers],
                y=df[y_col][iqr_only_outliers],
                mode='markers',
                name='IQR Outliers',
                marker=dict(color='red', size=8, opacity=0.8)
            ),
            row=1, col=1
        )
    
    # Specific outliers (data quality issues - removed)
    if specific_outliers.any():
        fig.add_trace(
            go.Scatter(
                x=df[x_col][specific_outliers],
                y=df[y_col][specific_outliers],
                mode='markers',
                name='Removed Outliers',
                marker=dict(color='gold', size=12, opacity=0.9, symbol='diamond')
            ),
            row=1, col=1
        )
    
    # 2. Box plot - X variable
    fig.add_trace(
        go.Box(
            y=df[x_col],
            name=x_col,
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # 3. Box plot - Y variable
    fig.add_trace(
        go.Box(
            y=df[y_col],
            name=y_col,
            marker_color='lightcoral'
        ),
        row=2, col=1
    )
    
    # 4. Outlier impact analysis
    all_outliers = iqr_outlier_condition | specific_outlier_condition
    if all_outliers.any():
        impact_data = {
            'Metric': ['Mean (with outliers)', 'Mean (without outliers)', 
                      'Median (with outliers)', 'Median (without outliers)'],
            'X Variable': [df[x_col].mean(), df[x_col][~all_outliers].mean(),
                          df[x_col].median(), df[x_col][~all_outliers].median()],
            'Y Variable': [df[y_col].mean(), df[y_col][~all_outliers].mean(),
                          df[y_col].median(), df[y_col][~all_outliers].median()]
        }
        
        fig.add_trace(
            go.Bar(
                x=impact_data['Metric'],
                y=impact_data['Y Variable'],
                name='Impact Analysis',
                marker_color=['red', 'green', 'red', 'green']
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=700,
        title_text="Enhanced Outlier Analysis",
        showlegend=True
    )
    
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_yaxes(title_text=f"{x_col} Values", row=1, col=2)
    fig.update_yaxes(title_text=f"{y_col} Values", row=2, col=1)
    fig.update_yaxes(title_text=f"{y_col} Values", row=2, col=2)
    
    return fig

def create_skewness_comparison(original_data, transformed_data, feature_name):
    """Create before/after skewness transformation comparison."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(f'{feature_name} - Original Distribution', 
                       f'{feature_name} - Transformed Distribution',
                       'Q-Q Plot - Original', 'Q-Q Plot - Transformed'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Original distribution
    fig.add_trace(
        go.Histogram(
            x=original_data,
            nbinsx=30,
            name='Original',
            marker_color='red',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Transformed distribution
    fig.add_trace(
        go.Histogram(
            x=transformed_data,
            nbinsx=30,
            name='Transformed',
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Q-Q plots
    # Original
    probplot_orig = stats.probplot(original_data, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=probplot_orig[0][0],
            y=probplot_orig[0][1],
            mode='markers',
            name='Original Q-Q',
            marker=dict(color='red', size=4)
        ),
        row=2, col=1
    )
    
    # Transformed
    probplot_trans = stats.probplot(transformed_data, dist="norm")
    fig.add_trace(
        go.Scatter(
            x=probplot_trans[0][0],
            y=probplot_trans[0][1],
            mode='markers',
            name='Transformed Q-Q',
            marker=dict(color='blue', size=4)
        ),
        row=2, col=2
    )
    
    # Add theoretical lines
    fig.add_trace(
        go.Scatter(
            x=probplot_orig[0][0],
            y=probplot_orig[1][1] + probplot_orig[1][0] * probplot_orig[0][0],
            mode='lines',
            name='Theoretical',
            line=dict(color='black', dash='dash')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=probplot_trans[0][0],
            y=probplot_trans[1][1] + probplot_trans[1][0] * probplot_trans[0][0],
            mode='lines',
            name='Theoretical',
            line=dict(color='black', dash='dash')
        ),
        row=2, col=2
    )
    
    # Add skewness annotations
    orig_skew = stats.skew(original_data)
    trans_skew = stats.skew(transformed_data)
    
    fig.add_annotation(
        x=0.5, y=0.95,
        text=f"Skewness: {orig_skew:.3f}",
        showarrow=False,
        xref="x domain", yref="y domain",
        row=1, col=1
    )
    
    fig.add_annotation(
        x=0.5, y=0.95,
        text=f"Skewness: {trans_skew:.3f}",
        showarrow=False,
        xref="x domain", yref="y domain",
        row=1, col=2
    )
    
    fig.update_layout(
        height=600,
        title_text=f"Skewness Transformation Analysis - {feature_name}",
        showlegend=False
    )
    
    return fig