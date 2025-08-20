# Interactive Data Analysis Notebook
# Author: Data Science Team
# Contact: 23f3002227@ds.study.iitm.ac.in
# Description: Interactive analysis of housing market data demonstrating
# relationships between price, size, and location factors

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    # Cell 1: Import libraries and setup
    # This cell establishes the foundation for our analysis
    # Dependencies: None (root cell)
    # Outputs: Libraries available for downstream cells
    
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import make_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set style for better visualizations
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    mo.md("# 🏠 Housing Market Data Analysis")
    return LinearRegression, StandardScaler, make_regression, mo, np, pd, plt, r2_score, mean_squared_error, sns, warnings


@app.cell
def __(make_regression, np, pd):
    # Cell 2: Generate synthetic housing dataset
    # This cell creates our analysis dataset with realistic housing features
    # Dependencies: Cell 1 (libraries)
    # Outputs: housing_data (DataFrame) -> Used by cells 3, 4, 5
    
    # Generate synthetic housing data for demonstration
    np.random.seed(42)
    
    # Create base features using sklearn's make_regression
    X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)
    
    # Transform to realistic housing data ranges
    housing_data = pd.DataFrame({
        'square_feet': (X[:, 0] * 500 + 2000).astype(int),  # 1000-3500 sq ft
        'bedrooms': np.clip((X[:, 1] * 1.5 + 3).astype(int), 1, 6),  # 1-6 bedrooms
        'age_years': np.clip((abs(X[:, 2]) * 15 + 5).astype(int), 1, 50),  # 1-50 years
        'price': (y * 50000 + 300000).astype(int)  # $200k-$800k range
    })
    
    # Add location factor (categorical)
    locations = ['Downtown', 'Suburbs', 'Rural']
    location_multipliers = [1.3, 1.0, 0.8]
    housing_data['location'] = np.random.choice(locations, size=1000)
    
    # Apply location multiplier to price
    for loc, mult in zip(locations, location_multipliers):
        mask = housing_data['location'] == loc
        housing_data.loc[mask, 'price'] = (housing_data.loc[mask, 'price'] * mult).astype(int)
    
    print(f"Generated dataset with {len(housing_data)} housing records")
    print(f"Price range: ${housing_data['price'].min():,} - ${housing_data['price'].max():,}")
    
    housing_data
    return X, housing_data, location_multipliers, locations, y


@app.cell
def __(housing_data, mo):
    # Cell 3: Interactive parameter controls
    # This cell creates UI widgets for user interaction
    # Dependencies: Cell 2 (housing_data)
    # Outputs: Widget values -> Used by cells 4, 5, 6 for dynamic analysis
    
    # Interactive slider for minimum square footage filter
    min_sqft_slider = mo.ui.slider(
        start=housing_data['square_feet'].min(),
        stop=housing_data['square_feet'].max(),
        step=100,
        value=2000,
        label="Minimum Square Feet"
    )
    
    # Dropdown for location filter
    location_dropdown = mo.ui.dropdown(
        options=['All'] + housing_data['location'].unique().tolist(),
        value='All',
        label="Location Filter"
    )
    
    # Slider for maximum house age
    max_age_slider = mo.ui.slider(
        start=1,
        stop=housing_data['age_years'].max(),
        step=1,
        value=25,
        label="Maximum House Age (years)"
    )
    
    # Display widgets in a clean layout
    mo.md(f"""
    ## 🎛️ Analysis Controls
    
    Use these controls to filter and analyze the housing data:
    
    {min_sqft_slider}
    {location_dropdown}
    {max_age_slider}
    """)
    return location_dropdown, max_age_slider, min_sqft_slider


@app.cell
def __(housing_data, location_dropdown, max_age_slider, min_sqft_slider):
    # Cell 4: Data filtering based on widget inputs
    # This cell processes the dataset according to user selections
    # Dependencies: Cell 2 (housing_data), Cell 3 (all sliders/dropdowns)
    # Outputs: filtered_data -> Used by cells 5, 6 for visualization and modeling
    
    # Apply filters based on widget values
    filtered_data = housing_data.copy()
    
    # Filter by minimum square feet
    filtered_data = filtered_data[filtered_data['square_feet'] >= min_sqft_slider.value]
    
    # Filter by location (if not 'All')
    if location_dropdown.value != 'All':
        filtered_data = filtered_data[filtered_data['location'] == location_dropdown.value]
    
    # Filter by maximum age
    filtered_data = filtered_data[filtered_data['age_years'] <= max_age_slider.value]
    
    # Store filter statistics for display
    filter_stats = {
        'total_records': len(housing_data),
        'filtered_records': len(filtered_data),
        'filter_percentage': (len(filtered_data) / len(housing_data)) * 100,
        'avg_price': filtered_data['price'].mean() if len(filtered_data) > 0 else 0,
        'price_std': filtered_data['price'].std() if len(filtered_data) > 0 else 0
    }
    
    filtered_data
    return filter_stats, filtered_data


@app.cell
def __(filter_stats, filtered_data, location_dropdown, max_age_slider, min_sqft_slider, mo):
    # Cell 5: Dynamic markdown output based on widget state
    # This cell generates contextual analysis based on current filter settings
    # Dependencies: Cell 3 (widget values), Cell 4 (filtered_data, filter_stats)
    # Outputs: Dynamic markdown display -> Self-updating based on widget changes
    
    # Generate dynamic content based on current widget states and filtered data
    if len(filtered_data) == 0:
        dynamic_content = mo.md("""
        ## ⚠️ No Data Available
        
        The current filter settings have removed all records. 
        Please adjust the filters to see analysis results.
        """)
    else:
        # Calculate key statistics
        correlation_sqft_price = filtered_data['square_feet'].corr(filtered_data['price'])
        correlation_age_price = filtered_data['age_years'].corr(filtered_data['price'])
        
        # Create location-specific insights
        if location_dropdown.value == 'All':
            location_insight = "across all locations"
        else:
            location_insight = f"in {location_dropdown.value} areas"
        
        # Generate insights based on correlations
        sqft_insight = "positively" if correlation_sqft_price > 0.3 else "weakly" if correlation_sqft_price > 0 else "negatively"
        age_insight = "negatively" if correlation_age_price < -0.3 else "weakly" if correlation_age_price < 0 else "positively"
        
        dynamic_content = mo.md(f"""
        ## 📊 Dynamic Analysis Results
        
        **Current Filter Settings:**
        - Minimum Size: {min_sqft_slider.value:,} sq ft
        - Location: {location_dropdown.value}
        - Maximum Age: {max_age_slider.value} years
        
        **Dataset Summary:**
        - **Records Analyzed**: {filter_stats['filtered_records']:,} of {filter_stats['total_records']:,} ({filter_stats['filter_percentage']:.1f}%)
        - **Average Price**: ${filter_stats['avg_price']:,.0f} (± ${filter_stats['price_std']:,.0f})
        - **Price Range**: ${filtered_data['price'].min():,} - ${filtered_data['price'].max():,}
        
        **Key Insights:**
        - House size is **{sqft_insight}** correlated with price (r = {correlation_sqft_price:.3f}) {location_insight}
        - House age is **{age_insight}** correlated with price (r = {correlation_age_price:.3f})
        - Properties in this filtered dataset average {filtered_data['square_feet'].mean():.0f} sq ft
        
        **Market Recommendation:**
        {"🏡 Great value opportunities in this segment!" if filter_stats['avg_price'] < 400000 else "💰 Premium market segment with higher prices."}
        """)
    
    dynamic_content
    return age_insight, correlation_age_price, correlation_sqft_price, dynamic_content, location_insight, sqft_insight


@app.cell
def __(LinearRegression, filtered_data, mo, plt, r2_score, sns):
    # Cell 6: Advanced visualization and modeling
    # This cell creates interactive plots and performs predictive modeling
    # Dependencies: Cell 1 (libraries), Cell 4 (filtered_data)
    # Outputs: Visual analysis -> Updates automatically when filters change
    
    if len(filtered_data) > 10:  # Ensure sufficient data for meaningful analysis
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Housing Market Analysis Dashboard', fontsize=16, y=0.98)
        
        # Plot 1: Price vs Square Feet scatter
        axes[0, 0].scatter(filtered_data['square_feet'], filtered_data['price'], 
                          alpha=0.6, s=50, c='skyblue', edgecolors='navy', linewidth=0.5)
        axes[0, 0].set_xlabel('Square Feet')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].set_title('Price vs. Square Footage')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(filtered_data['square_feet'], filtered_data['price'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(filtered_data['square_feet'], p(filtered_data['square_feet']), 
                       "r--", alpha=0.8, linewidth=2)
        
        # Plot 2: Price distribution by location
        if 'location' in filtered_data.columns and len(filtered_data['location'].unique()) > 1:
            sns.boxplot(data=filtered_data, x='location', y='price', ax=axes[0, 1])
            axes[0, 1].set_title('Price Distribution by Location')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].hist(filtered_data['price'], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Price Distribution')
            axes[0, 1].set_xlabel('Price ($)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Age vs Price relationship
        axes[1, 0].scatter(filtered_data['age_years'], filtered_data['price'], 
                          alpha=0.6, s=50, c='coral', edgecolors='darkred', linewidth=0.5)
        axes[1, 0].set_xlabel('House Age (years)')
        axes[1, 0].set_ylabel('Price ($)')
        axes[1, 0].set_title('Price vs. House Age')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Predictive modeling visualization
        if len(filtered_data) > 20:  # Need sufficient data for modeling
            # Simple linear regression model
            X_model = filtered_data[['square_feet', 'bedrooms', 'age_years']]
            y_model = filtered_data['price']
            
            model = LinearRegression()
            model.fit(X_model, y_model)
            predictions = model.predict(X_model)
            r2 = r2_score(y_model, predictions)
            
            axes[1, 1].scatter(y_model, predictions, alpha=0.6, s=50, c='gold', edgecolors='orange')
            axes[1, 1].plot([y_model.min(), y_model.max()], [y_model.min(), y_model.max()], 
                           'r--', lw=2, alpha=0.8)
            axes[1, 1].set_xlabel('Actual Price ($)')
            axes[1, 1].set_ylabel('Predicted Price ($)')
            axes[1, 1].set_title(f'Model Performance (R² = {r2:.3f})')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data\nfor modeling', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=14, style='italic')
            axes[1, 1].set_title('Predictive Model')
        
        plt.tight_layout()
        visualization_output = mo.as_html(fig)
        plt.close()  # Prevent memory leaks
        
    else:
        # Fallback for insufficient data
        visualization_output = mo.md("""
        ### 📈 Visualization Unavailable
        
        Need at least 10 records to generate meaningful visualizations.
        Please adjust your filters to include more data.
        """)
    
    visualization_output
    return axes, fig, model, predictions, visualization_output


@app.cell
def __(filtered_data, mo):
    # Cell 7: Data summary table and export functionality
    # This cell provides detailed data inspection capabilities
    # Dependencies: Cell 4 (filtered_data)
    # Outputs: Interactive data table with summary statistics
    
    if len(filtered_data) > 0:
        # Generate summary statistics
        summary_stats = filtered_data.describe()
        
        # Create correlation matrix for numeric columns
        numeric_cols = ['square_feet', 'bedrooms', 'age_years', 'price']
        correlation_matrix = filtered_data[numeric_cols].corr()
        
        # Display summary information
        data_summary = mo.md(f"""
        ## 📋 Detailed Data Summary
        
        **Dataset Overview:**
        - **Total Properties**: {len(filtered_data)}
        - **Unique Locations**: {filtered_data['location'].nunique()}
        - **Average Bedrooms**: {filtered_data['bedrooms'].mean():.1f}
        
        **Key Statistics:**
        """)
        
        # Show sample data and summary stats
        sample_data = mo.ui.table(
            filtered_data.head(10),
            label="Sample Data (First 10 Records)"
        )
        
        summary_table = mo.ui.table(
            summary_stats.round(2),
            label="Summary Statistics"
        )
        
        correlation_table = mo.ui.table(
            correlation_matrix.round(3),
            label="Correlation Matrix"
        )
        
        combined_output = mo.vstack([
            data_summary,
            sample_data,
            mo.md("### Summary Statistics"),
            summary_table,
            mo.md("### Variable Correlations"),
            correlation_table
        ])
        
    else:
        combined_output = mo.md("No data available for summary.")
    
    combined_output
    return combined_output, correlation_matrix, correlation_table, data_summary, sample_data, summary_stats, summary_table


@app.cell
def __(mo):
    # Cell 8: Documentation and methodology
    # This cell provides comprehensive documentation of the analysis approach
    # Dependencies: None (documentation cell)
    # Outputs: Static documentation for reference
    
    methodology_docs = mo.md("""
    ## 📖 Analysis Methodology & Documentation
    
    ### Data Flow Architecture:
    
    ```
    Cell 1 (Libraries) 
    ↓
    Cell 2 (Data Generation) → housing_data
    ↓
    Cell 3 (Interactive Controls) → widget_values
    ↓
    Cell 4 (Data Filtering) → filtered_data ← depends on housing_data + widget_values
    ↓
    Cell 5 (Dynamic Analysis) ← depends on filtered_data + widget_values
    ↓
    Cell 6 (Visualizations) ← depends on filtered_data
    ↓
    Cell 7 (Data Tables) ← depends on filtered_data
    ```
    
    ### Variable Dependencies:
    
    1. **Root Variables**: Libraries, housing_data (no dependencies)
    2. **Interactive Variables**: Widget values from sliders/dropdowns
    3. **Derived Variables**: filtered_data (depends on housing_data + widgets)
    4. **Output Variables**: All visualizations and analyses (depend on filtered_data)
    
    ### Key Features Demonstrated:
    
    - **🔄 Reactive Programming**: All downstream cells automatically update when widgets change
    - **📊 Dynamic Visualizations**: Charts and plots update based on filter selections  
    - **📝 Contextual Documentation**: Markdown output changes based on current analysis state
    - **🎛️ Interactive Controls**: Sliders and dropdowns for user-driven exploration
    - **📈 Statistical Modeling**: Linear regression with performance metrics
    - **🏠 Domain-Specific Analysis**: Real estate market insights and recommendations
    
    ### Technical Implementation:
    
    - **Framework**: Marimo reactive notebook environment
    - **Data Generation**: Synthetic housing dataset with realistic distributions
    - **Interactivity**: Multi-widget filtering system with real-time updates
    - **Visualization**: Matplotlib/Seaborn for publication-quality plots
    - **Modeling**: Scikit-learn for predictive analytics
    
    ### Author Information:
    
    **Contact**: 23f3002227@ds.study.iitm.ac.in  
    **Purpose**: Demonstrate interactive data analysis capabilities  
    **Dataset**: Synthetic housing market data (1000 properties)  
    **Last Updated**: August 2024
    """)
    
    methodology_docs
    return (methodology_docs,)


if __name__ == "__main__":
    app.run()