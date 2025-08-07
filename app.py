import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Disease Outbreak Prediction Agent",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-medium {
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .alert-low {
        background: linear-gradient(135deg, #48dbfb 0%, #0abde3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the disease outbreak data"""
    try:
        # Try to load the full dataset
        data = pd.read_csv('final_data.csv')
    except FileNotFoundError:
        # If file not found, create sample data based on the provided snippet
        sample_data = {
            'week_of_outbreak': ['1st week', '2nd week', '3rd week', '3rd week', '3rd week', '3rd week', '5th week', '6th week', '6th week', '7th week', '7th week', '7th week', '8th week', '9th week', '9th week', '9th week', '9th week', '9th week', '9th week', '9th week', '10th week'],
            'state_ut': ['Meghalaya', 'Maharashtra', 'Tamil Nadu', 'Gujarat', 'Kerala', 'Tamil Nadu', 'Uttar Pradesh', 'Maharashtra', 'Odisha', 'Gujarat', 'West Bengal', 'Kerala', 'West Bengal', 'Kerala', 'West Bengal', 'West Bengal', 'Kerala', 'Karnataka', 'West Bengal', 'West Bengal', 'Gujarat'],
            'district': ['East Jaintia Hills', 'Gadchiroli', 'Pudukottai', 'Patan', 'Ernakulam', 'Pudukottai', 'Fatehpur', 'Gadchiroli', 'Koraput', 'Ahmedabad', 'Bardhaman', 'Trivandrum', 'South 24 parganas', 'Ernakulam', 'South 24 parganas', 'North 24 Parganas', 'Ernakulam', 'Tumakuru', 'South 24 Parganas', 'North 24 Parganas', 'Navsari'],
            'Disease': ['Acute Diarrhoeal Disease', 'Malaria', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Encephalitis Syndrome', 'Malaria', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease', 'Acute Diarrhoeal Disease'],
            'Cases': [160, 7, 8, 7, 14, 8, 1, 1, 15, 67, 89, 5, 21, 21, 21, 359, 21, 27, 21, 359, 57],
            'Deaths': [np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            'day': [2, 10, 18, 11, 24, 18, 24, 9, 11, 17, 17, 14, 23, 24, 23, 5, 24, 3, 23, 5, 8],
            'mon': [1, 1, 1, 1, 12, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3],
            'year': [2022, 2022, 2022, 2022, 2021, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022, 2022],
            'Latitude': [25.25157605, 19.75907035, 10.3826515, 23.77405735, 9.98408, 10.3826515, 25.8435395, 19.75907035, 18.7232023, 23.0216238, 23.41146995, 8.4882267, 22.5760669, 9.98408, 22.5760669, 22.642931, 9.98408, 13.3400771, 22.5760669, 22.642931, 20.952407],
            'Longitude': [92.48405007, 80.16228073, 78.8191259, 71.68373466, 76.2741457, 78.8191259, 80.91800397, 80.16228073, 82.60811828, 72.5797068, 87.84499445, 76.947551, 88.42006758, 76.2741457, 88.42006758, 88.8930579, 76.2741457, 77.1006208, 88.42006758, 88.8930579, 72.9323831],
            'preci': [0.02035381907, 0.007479298714, 0.1074129745, 0.06509449409, 0.04125608103, 0.1074129745, 0.09903011933, 8.36e-05, 0.0004905797777, 7.25e-05, 0.003711025774, 0.348763274, 0.1333235523, 0.05127556617, 0.1333235523, 0.0127255705, 0.05127556617, 6.49e-05, 0.1333235523, 0.0127255705, 0.0001374485643],
            'LAI': [34.5, 9.0, 12.0, 9.0, 33.0, 12.0, np.nan, np.nan, 12.0, np.nan, np.nan, 37.0, 10.0, 40.0, 10.0, np.nan, 40.0, 7.0, 10.0, np.nan, 4.0],
            'Temp': [291.5333334, 299.97, 300.7666666, 299.08, 303.028, 300.7666666, 290.18, 300.06333340000003, 300.55666659999997, 302.55333340000004, 296.68, 302.955, 299.22, 302.8933334, 299.22, 299.72, 302.8933334, 308.4633334, 299.22, 299.72, 307.9233334]
        }
        data = pd.DataFrame(sample_data)
        st.warning("Using sample data. Please upload 'final_data.csv' for full functionality.")
    
    # Data preprocessing
    data['Cases'] = pd.to_numeric(data['Cases'], errors='coerce')
    data = data.rename(columns={'mon': 'month'})
    data['Deaths'] = data['Deaths'].fillna(0)
    data['LAI'] = data['LAI'].fillna(data['LAI'].mean())
    data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
    data['Temp_Celsius'] = data['Temp'] - 273.15  # Convert Kelvin to Celsius
    data['month'] = data['date'].dt.month
    season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Summer', 4: 'Summer', 5: 'Summer',
    6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
    10: 'Post-Monsoon', 11: 'Post-Monsoon'
}
    data['season'] = data['month'].map(season_map)
    
    return data

def create_ml_model(data):
    """Create and train machine learning models for prediction and anomaly detection"""
    # Prepare features for ML
    le_disease = LabelEncoder()
    le_state = LabelEncoder()
    le_district = LabelEncoder()
    
    ml_data = data.copy()
    ml_data['Disease_encoded'] = le_disease.fit_transform(ml_data['Disease'])
    ml_data['state_encoded'] = le_state.fit_transform(ml_data['state_ut'])
    ml_data['district_encoded'] = le_district.fit_transform(ml_data['district'])
    
    # Features for prediction
    features = ['Disease_encoded', 'state_encoded', 'district_encoded', 'Latitude', 'Longitude', 
                'preci', 'LAI', 'Temp', 'month']
    
    X = ml_data[features].fillna(ml_data[features].mean())
    y = ml_data['Cases']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Random Forest for prediction
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Isolation Forest for anomaly detection
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(X_train)
    
    # Model evaluation
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return rf_model, isolation_forest, le_disease, le_state, le_district, mae, r2

def create_outbreak_map(data):
    """Create an interactive map showing disease outbreaks"""
    # Calculate center of India
    center_lat = data['Latitude'].mean()
    center_lon = data['Longitude'].mean()
    
    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    # Dynamic color mapping for diseases
    diseases = data['Disease'].unique()
    num_diseases = len(diseases)
    # Use a Plotly sequential color scale to generate distinct colors
    colors = px.colors.sample_colorscale("Turbo", [n/num_diseases for n in range(num_diseases)])
    color_map = dict(zip(diseases, colors))
    
    # Add markers for each outbreak
    for idx, row in data.iterrows():
        popup_text = f"""
        <b>Disease:</b> {row['Disease']}<br>
        <b>Location:</b> {row['district']}, {row['state_ut']}<br>
        <b>Cases:</b> {row['Cases']}<br>
        <b>Deaths:</b> {row['Deaths']}<br>
        <b>Date:</b> {row['date'].strftime('%Y-%m-%d')}<br>
        <b>Temperature:</b> {row['Temp_Celsius']:.1f}°C<br>
        <b>Precipitation:</b> {row['preci']:.4f}
        """
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=min(row['Cases']/10 + 5, 20),  # Size based on cases
            popup=popup_text,
            color=color_map.get(row['Disease'], 'gray'),
            fillColor=color_map.get(row['Disease'], 'gray'),
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    # Add legend
    legend_html = '<div style="position: fixed; top: 10px; right: 10px; width: 200px; height: auto; background-color: white; border:2px solid grey; z-index:9999; font-size:14px; padding: 10px"><p><b>Disease Types</b></p>'
    for disease, color in color_map.items():
        legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {disease}</p>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def calculate_risk_score(cases, deaths, population_density=1000): # Note: population_density is a placeholder
    """Calculate a risk score. TODO: Integrate real population density data."""
    epsilon = 1e-6  # Small value to avoid division by zero
    death_rate = deaths / (cases + epsilon)
    
    # Weighted impact of cases and deaths, adjusted by death rate and population density
    risk_score = (cases * 0.6 + deaths * 0.4) * (1 + death_rate) / (population_density + epsilon) * 10000
    return min(risk_score, 100)  # Cap at 100 for a consistent scale

def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 Disease Outbreak Prediction Agent</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    # Sidebar
    st.sidebar.title("🔧 Control Panel")
    
    # Filters
    st.sidebar.subheader("📊 Data Filters")
    selected_states = st.sidebar.multiselect("Select States", data['state_ut'].unique(), default=data['state_ut'].unique()[:5])
    selected_diseases = st.sidebar.multiselect("Select Diseases", data['Disease'].unique(), default=data['Disease'].unique())
    # Use the min and max dates from the full dataset for the date_input range
    min_date = data['date'].min().date()
    max_date = data['date'].max().date()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)
    
    # Filter data
    if selected_states:
        data = data[data['state_ut'].isin(selected_states)]
    if selected_diseases:
        data = data[data['Disease'].isin(selected_diseases)]
    if len(date_range) == 2:
        data = data[(data['date'] >= pd.to_datetime(date_range[0])) & (data['date'] <= pd.to_datetime(date_range[1]))]
    # ============== FIX: ADD THIS VALIDATION BLOCK ==============
    # Check if the dataframe is empty after filtering and stop execution if it is.
    if data.empty:
        st.error("No data available for the selected filters. Please adjust your selection or date range.")
        st.stop() # This halts the script and prevents errors.
    # ============================================================
    # Create ML models
    if len(data) > 10:  # Only create models if sufficient data
        rf_model, isolation_forest, le_disease, le_state, le_district, mae, r2 = create_ml_model(data)
        
        # Model performance
        st.sidebar.subheader("🤖 ML Model Performance")
        st.sidebar.metric("Mean Absolute Error", f"{mae:.2f}")
        st.sidebar.metric("R² Score", f"{r2:.3f}")
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = data['Cases'].sum()
        st.markdown(f'<div class="metric-card"><h3>{total_cases:,}</h3><p>Total Cases</p></div>', unsafe_allow_html=True)
    
    with col2:
        total_deaths = data['Deaths'].sum()
        st.markdown(f'<div class="metric-card"><h3>{total_deaths:,.0f}</h3><p>Total Deaths</p></div>', unsafe_allow_html=True)
    
    with col3:
        affected_states = data['state_ut'].nunique()
        st.markdown(f'<div class="metric-card"><h3>{affected_states}</h3><p>Affected States</p></div>', unsafe_allow_html=True)
    
    with col4:
        case_fatality_rate = (total_deaths / max(total_cases, 1)) * 100
        st.markdown(f'<div class="metric-card"><h3>{case_fatality_rate:.2f}%</h3><p>Case Fatality Rate</p></div>', unsafe_allow_html=True)
    
    # Risk Assessment
    st.subheader("🚨 Risk Assessment")
    risk_data = data.groupby(['state_ut', 'Disease']).agg({
        'Cases': 'sum',
        'Deaths': 'sum'
    }).reset_index()
    risk_data['Risk_Score'] = risk_data.apply(lambda x: calculate_risk_score(x['Cases'], x['Deaths']), axis=1)
    
    # Display top risks
    top_risks = risk_data.nlargest(3, 'Risk_Score')
    
    col1, col2, col3 = st.columns(3)
    risk_colors = ['alert-high', 'alert-medium', 'alert-low']
    
    for i, (_, risk) in enumerate(top_risks.iterrows()):
        with [col1, col2, col3][i]:
            st.markdown(f'''
            <div class="{risk_colors[i]}">
                <h4>Risk Level {i+1}</h4>
                <p><b>{risk['state_ut']}</b></p>
                <p>{risk['Disease']}</p>
                <p>Score: {risk['Risk_Score']:.1f}/100</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📍 Interactive Map", "📊 Analytics", "🔮 Predictions", "🌡️ Environmental Factors", "📈 Trends"])
    
    with tab1:
        st.subheader("📍 Disease Outbreak Map")
        outbreak_map = create_outbreak_map(data)
        st_folium(outbreak_map, width=1000, height=500)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        with col1:
            disease_counts = data.groupby('Disease')['Cases'].sum().sort_values(ascending=False)
            fig = px.pie(values=disease_counts.values, names=disease_counts.index, 
                        title="Disease Distribution by Cases")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            state_counts = data.groupby('state_ut')['Cases'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(x=state_counts.values, y=state_counts.index, orientation='h',
                        title="Top 10 States by Cases", labels={'x': 'Cases', 'y': 'State'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📊 Advanced Analytics")
        
        # Time series analysis
        daily_cases = data.groupby('date')['Cases'].sum().reset_index()
        fig = px.line(daily_cases, x='date', y='Cases', title="Cases Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        col1, col2 = st.columns(2)
        with col1:
            corr_data = data[['Cases', 'Deaths', 'Temp_Celsius', 'preci', 'LAI']].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect="auto", 
                           title="Environmental Factors Correlation")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Seasonal analysis
            seasonal_data = data.groupby('season')['Cases'].sum()
            fig = px.bar(x=seasonal_data.index, y=seasonal_data.values,
                        title="Cases by Season", labels={'x': 'Season', 'y': 'Total Cases'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔮 AI-Powered Predictions")
        
        if len(data) > 10:
            # Prediction interface
            col1, col2 = st.columns(2)
            
            with col1:
                pred_disease = st.selectbox("Select Disease", data['Disease'].unique())
                pred_state = st.selectbox("Select State", data['state_ut'].unique())
                pred_district = st.selectbox("Select District", 
                                           data[data['state_ut'] == pred_state]['district'].unique())
                pred_temp = st.slider("Temperature (°C)", 15.0, 45.0, 25.0)
                pred_preci = st.slider("Precipitation", 0.0, 1.0, 0.1)
                pred_lai = st.slider("Leaf Area Index", 0.0, 50.0, 20.0)
                pred_month = st.selectbox("Month", range(1, 13))
            
            with col2:
                if st.button("🔮 Predict Outbreak"):
                    
                    # ... inside the 'if st.button("🔮 Predict Outbreak"):' block
                    try:
                        # Get location data safely
                        location_data = data[(data['state_ut'] == pred_state) & (data['district'] == pred_district)]
                        if location_data.empty:
                            st.error(f"No historical data available for {pred_district}, {pred_state}. Cannot retrieve location coordinates.")
                            st.stop()
        
                        # Safely transform categorical features
                        disease_encoded = le_disease.transform([pred_disease])[0]
    
                        # Handle potentially unseen states/districts
                        try:
                            state_encoded = le_state.transform([pred_state])[0]
                            district_encoded = le_district.transform([pred_district])[0]
                        except ValueError as e:
                            st.error(f"The selected location '{pred_district}, {pred_state}' is new or not in the training data. Prediction is not possible. Error: {e}")
                            st.stop()

                        # Prepare prediction data
                        pred_data = np.array([[
                            disease_encoded,
                            state_encoded,
                            district_encoded,
                            location_data['Latitude'].mean(),  # Use mean to be safe
                            location_data['Longitude'].mean(), # Use mean to be safe
                            pred_preci,
                            pred_lai,
                            pred_temp + 273.15,  # Convert to Kelvin
                            pred_month
                            ]])
    
                        # Make prediction
                        predicted_cases = rf_model.predict(pred_data)[0]
                        # ... rest of the prediction logic
    
                    except Exception as e:
                        st.error(f"An error occurred during prediction: {e}")
                    
                    # Make prediction
                    predicted_cases = rf_model.predict(pred_data)[0]
                    anomaly_score = isolation_forest.decision_function(pred_data)[0]
                    is_anomaly = isolation_forest.predict(pred_data)[0] == -1
                    
                    # Display results
                    st.success(f"🎯 Predicted Cases: **{predicted_cases:.0f}**")
                    
                    if is_anomaly:
                        st.warning(f"⚠️ Anomaly Detected! Anomaly Score: {anomaly_score:.3f}")
                    else:
                        st.info(f"✅ Normal Pattern. Anomaly Score: {anomaly_score:.3f}")
                    
                    # Risk assessment
                    risk_level = "High" if predicted_cases > 100 else "Medium" if predicted_cases > 50 else "Low"
                    st.metric("Risk Level", risk_level)
            
            # Feature importance
            feature_names = ['Disease', 'State', 'District', 'Latitude', 'Longitude', 
                           'Precipitation', 'LAI', 'Temperature', 'Month']
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': rf_model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Feature Importance for Prediction")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Insufficient data for predictions. Please adjust filters or add more data.")
    
    with tab4:
        st.subheader("🌡️ Environmental Factors Analysis")
        
        # Temperature vs Cases
        fig = px.scatter(data, x='Temp_Celsius', y='Cases', color='Disease',
                        title="Temperature vs Cases", hover_data=['state_ut', 'district'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Precipitation vs Cases
        fig = px.scatter(data, x='preci', y='Cases', color='Disease',
                        title="Precipitation vs Cases", hover_data=['state_ut', 'district'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Environmental factors by disease
        col1, col2 = st.columns(2)
        
        with col1:
            env_by_disease = data.groupby('Disease')[['Temp_Celsius', 'preci', 'LAI']].mean()
            fig = px.bar(env_by_disease, title="Average Environmental Conditions by Disease")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot for temperature distribution by disease
            fig = px.box(data, x='Disease', y='Temp_Celsius', 
                        title="Temperature Distribution by Disease")
            fig.update_layout(xaxis=dict(tickangle=45))
            st.plotly_chart(fig, use_container_width=True)

    
    with tab5:
        st.subheader("📈 Outbreak Trends & Patterns")
        
        # Monthly trends
        monthly_data = data.groupby(['month', 'Disease'])['Cases'].sum().reset_index()
        fig = px.line(monthly_data, x='month', y='Cases', color='Disease',
                     title="Monthly Disease Trends")
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographical spread
        geo_spread = data.groupby(['state_ut', 'Disease']).size().reset_index(name='Outbreaks')
        fig = px.treemap(geo_spread, path=['state_ut', 'Disease'], values='Outbreaks',
                        title="Geographical Distribution of Outbreaks")
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly progression
        weekly_data = data.groupby(['week_of_outbreak', 'Disease'])['Cases'].sum().reset_index()
        # Create a numeric sort key from the 'week_of_outbreak' string
        weekly_data['week_num'] = weekly_data['week_of_outbreak'].str.extract('(\d+)').astype(int)
        weekly_data = weekly_data.sort_values('week_num') # Sort by the new numeric column

        fig = px.bar(weekly_data, x='week_of_outbreak', y='Cases', color='Disease',
             title="Weekly Outbreak Progression")
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🦠 Disease Outbreak Prediction Agent | Powered by AI & Machine Learning</p>
        <p>📊 Real-time monitoring • 🔮 Predictive analytics • 🗺️ Geographic mapping</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()