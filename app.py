import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from ml_models import AddictionPredictor
from visualizations import DashboardVisualizations
from community_insights import CommunityInsights
from real_time_ingestion import RealTimeDataIngestion, RealTimeVisualization

# Configure page
st.set_page_config(
    page_title="Addiction Recovery Analytics Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for white and green theme
st.markdown("""
<style>
    .main-header {
        color: #28a745;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #28a745;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #28a745;
        margin: 1rem 0;
    }
    .section-header {
        color: #28a745;
        border-bottom: 1px solid #28a745;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">Addiction Recovery Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Initialize components
    data_processor = DataProcessor()
    visualizations = DashboardVisualizations()
    community_insights = CommunityInsights()
    
    # Initialize real-time monitoring (session state)
    if 'rt_ingestion' not in st.session_state:
        st.session_state.rt_ingestion = RealTimeDataIngestion(data_processor)
    if 'rt_visualization' not in st.session_state:
        st.session_state.rt_visualization = RealTimeVisualization()
    
    # Sidebar for file upload and navigation
    st.sidebar.markdown('<h2 class="section-header">Navigation</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset (CSV)", 
        type=['csv'],
        help="Upload your addiction dataset CSV file"
    )
    
    # Use uploaded file or default file
    if uploaded_file is not None:
        df = data_processor.load_data(uploaded_file)
    else:
        # Try to load the default dataset
        try:
            df = data_processor.load_data("attached_assets/student_addiction_dataset_train_1758622709749.csv")
        except:
            st.error("Please upload a dataset to begin analysis")
            return
    
    if df is None:
        st.error("Failed to load dataset. Please check the file format.")
        return
    
    # Process data
    df_processed = data_processor.preprocess_data(df)
    
    # Navigation
    pages = [
        "Dashboard Overview",
        "Big Numbers & KPIs", 
        "Comparative Analysis",
        "Data Composition",
        "Time Series Analysis",
        "Correlation Analysis",
        "Data Explorer",
        "ML Predictions",
        "User Segmentation",
        "Community Insights",
        "Real-Time Monitoring"
    ]
    
    selected_page = st.sidebar.selectbox("Select Analysis", pages)
    
    # Main content based on selected page
    if selected_page == "Dashboard Overview":
        show_dashboard_overview(df_processed, visualizations)
    elif selected_page == "Big Numbers & KPIs":
        show_big_numbers(df_processed, visualizations)
    elif selected_page == "Comparative Analysis":
        show_comparative_analysis(df_processed, visualizations)
    elif selected_page == "Data Composition":
        show_data_composition(df_processed, visualizations)
    elif selected_page == "Time Series Analysis":
        show_time_series_analysis(df_processed, visualizations)
    elif selected_page == "Correlation Analysis":
        show_correlation_analysis(df_processed, visualizations)
    elif selected_page == "Data Explorer":
        show_data_explorer(df_processed)
    elif selected_page == "ML Predictions":
        show_ml_predictions(df_processed)
    elif selected_page == "User Segmentation":
        show_user_segmentation(df_processed, predictor)
    
    elif selected_page == "Community Insights":
        show_community_insights(df_processed, community_insights)
    elif selected_page == "Real-Time Monitoring":
        show_real_time_monitoring()

def show_dashboard_overview(df, visualizations):
    st.markdown('<h2 class="section-header">Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    total_participants = len(df)
    addiction_rate = (df['Addiction_Class'] == 'Yes').mean() * 100
    high_risk_factors = df[['Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues']].apply(lambda x: x == 'Yes').sum(axis=1).mean()
    recovery_indicators = df[['Denial_and_Resistance_to_Treatment']].apply(lambda x: x == 'No').sum(axis=1).mean()
    
    with col1:
        st.metric("Total Participants", f"{total_participants:,}")
    with col2:
        st.metric("Addiction Rate", f"{addiction_rate:.1f}%")
    with col3:
        st.metric("Avg Risk Factors", f"{high_risk_factors:.1f}")
    with col4:
        st.metric("Recovery Readiness", f"{recovery_indicators:.1f}")
    
    # Overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizations.create_addiction_distribution(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizations.create_risk_factors_overview(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Key Insights:**")
    st.markdown(f"- {addiction_rate:.1f}% of participants show signs of addiction")
    st.markdown(f"- Most common risk factor: {visualizations.get_top_risk_factor(df)}")
    st.markdown(f"- Recovery readiness varies significantly across the population")
    st.markdown('</div>', unsafe_allow_html=True)

def show_big_numbers(df, visualizations):
    st.markdown('<h2 class="section-header">Big Numbers & Key Performance Indicators</h2>', unsafe_allow_html=True)
    
    # Calculate key metrics
    metrics = visualizations.calculate_big_numbers(df)
    
    # Display big numbers in grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Cases", f"{metrics['total_cases']:,}")
        st.metric("Addiction Cases", f"{metrics['addiction_cases']:,}")
        st.metric("Non-Addiction Cases", f"{metrics['non_addiction_cases']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Addiction Rate", f"{metrics['addiction_rate']:.1f}%")
        st.metric("Academic Decline", f"{metrics['academic_decline_rate']:.1f}%")
        st.metric("Social Isolation", f"{metrics['social_isolation_rate']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Financial Issues", f"{metrics['financial_issues_rate']:.1f}%")
        st.metric("Health Problems", f"{metrics['health_problems_rate']:.1f}%")
        st.metric("Treatment Resistance", f"{metrics['treatment_resistance_rate']:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown("### Risk Factor Severity")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizations.create_severity_gauge(metrics['addiction_rate'], "Addiction Rate")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizations.create_severity_gauge(metrics['treatment_resistance_rate'], "Treatment Resistance")
        st.plotly_chart(fig, use_container_width=True)

def show_comparative_analysis(df, visualizations):
    st.markdown('<h2 class="section-header">Comparative Analysis</h2>', unsafe_allow_html=True)
    
    # Compare addicted vs non-addicted groups
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizations.create_group_comparison_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizations.create_risk_factor_comparison(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed comparison table
    st.markdown("### Detailed Group Comparison")
    comparison_df = visualizations.create_comparison_table(df)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Statistical significance testing
    st.markdown("### Statistical Analysis")
    stats_results = visualizations.perform_statistical_tests(df)
    for factor, result in stats_results.items():
        if result['p_value'] < 0.05:
            significance = "Statistically significant"
            color = "#28a745"
        else:
            significance = "Not statistically significant"
            color = "#dc3545"
        
        st.markdown(f"**{factor}**: {significance} (p={result['p_value']:.4f})")

def show_data_composition(df, visualizations):
    st.markdown('<h2 class="section-header">Data Composition Analysis</h2>', unsafe_allow_html=True)
    
    # Pie charts for composition
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizations.create_addiction_pie_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizations.create_risk_factors_pie_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Stacked bar charts
    st.markdown("### Risk Factor Distribution by Addiction Status")
    fig = visualizations.create_stacked_bar_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Composition breakdown
    st.markdown("### Detailed Composition Breakdown")
    composition_data = visualizations.get_composition_breakdown(df)
    
    for category, data in composition_data.items():
        st.markdown(f"**{category}:**")
        for item, percentage in data.items():
            st.progress(percentage/100, text=f"{item}: {percentage:.1f}%")

def show_time_series_analysis(df, visualizations):
    st.markdown('<h2 class="section-header">Time Series Analysis</h2>', unsafe_allow_html=True)
    
    st.info("Simulating addiction progression patterns over time based on risk factors")
    
    # Generate time series simulation
    time_series_data = visualizations.simulate_addiction_progression(df)
    
    # Time series charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = visualizations.create_progression_timeline(time_series_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = visualizations.create_recovery_timeline(time_series_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend analysis
    st.markdown("### Trend Analysis")
    fig = visualizations.create_trend_analysis(time_series_data)
    st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown("**Time Series Insights:**")
    st.markdown("- Early intervention shows better recovery outcomes")
    st.markdown("- Social isolation tends to accelerate addiction progression")
    st.markdown("- Financial stability is crucial for sustained recovery")
    st.markdown('</div>', unsafe_allow_html=True)

def show_correlation_analysis(df, visualizations):
    st.markdown('<h2 class="section-header">Correlation Analysis</h2>', unsafe_allow_html=True)
    
    # Correlation heatmap
    fig = visualizations.create_correlation_heatmap(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Relationship explanations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Strong Positive Correlations")
        correlations = visualizations.get_correlation_insights(df)
        for relationship, strength in correlations['positive'].items():
            st.markdown(f"- **{relationship}**: {strength:.3f}")
    
    with col2:
        st.markdown("### Strong Negative Correlations")
        for relationship, strength in correlations['negative'].items():
            st.markdown(f"- **{relationship}**: {strength:.3f}")
    
    # Network analysis
    st.markdown("### Risk Factor Network")
    fig = visualizations.create_network_analysis(df)
    st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    st.markdown('<h2 class="section-header">Data Explorer</h2>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        addiction_filter = st.selectbox(
            "Addiction Status",
            ['All', 'Yes', 'No'],
            key="addiction_filter"
        )
    
    with col2:
        academic_filter = st.selectbox(
            "Academic Performance",
            ['All', 'Yes', 'No'],
            key="academic_filter"
        )
    
    with col3:
        social_filter = st.selectbox(
            "Social Isolation",
            ['All', 'Yes', 'No'],
            key="social_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    if addiction_filter != 'All':
        filtered_df = filtered_df[filtered_df['Addiction_Class'] == addiction_filter]
    if academic_filter != 'All':
        filtered_df = filtered_df[filtered_df['Academic_Performance_Decline'] == academic_filter]
    if social_filter != 'All':
        filtered_df = filtered_df[filtered_df['Social_Isolation'] == social_filter]
    
    # Display filtered data
    st.markdown(f"### Filtered Data ({len(filtered_df)} records)")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Summary statistics
    if len(filtered_df) > 0:
        st.markdown("### Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(filtered_df))
        with col2:
            addiction_pct = (filtered_df['Addiction_Class'] == 'Yes').mean() * 100
            st.metric("Addiction Rate", f"{addiction_pct:.1f}%")
        with col3:
            risk_factors = filtered_df[['Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues']].apply(lambda x: x == 'Yes').sum(axis=1).mean()
            st.metric("Avg Risk Factors", f"{risk_factors:.1f}")
        with col4:
            health_issues = (filtered_df['Physical_Mental_Health_Problems'] == 'Yes').mean() * 100
            st.metric("Health Issues", f"{health_issues:.1f}%")

def show_ml_predictions(df):
    st.markdown('<h2 class="section-header">Machine Learning Predictions</h2>', unsafe_allow_html=True)
    
    # Initialize ML model
    predictor = AddictionPredictor()
    
    # Train model
    with st.spinner("Training machine learning model..."):
        accuracy, report, feature_importance = predictor.train_model(df)
    
    # Display model performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Performance")
        st.metric("Accuracy", f"{accuracy:.3f}")
        st.text_area("Classification Report", report, height=200)
    
    with col2:
        st.markdown("### Feature Importance")
        fig = predictor.plot_feature_importance(feature_importance)
        st.plotly_chart(fig, use_container_width=True)
    
    # Individual prediction
    st.markdown("### Individual Risk Assessment")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            experimentation = st.selectbox("Experimentation", ['No', 'Yes'])
            academic_decline = st.selectbox("Academic Performance Decline", ['No', 'Yes'])
            social_isolation = st.selectbox("Social Isolation", ['No', 'Yes'])
            financial_issues = st.selectbox("Financial Issues", ['No', 'Yes'])
            health_problems = st.selectbox("Physical/Mental Health Problems", ['No', 'Yes'])
        
        with col2:
            legal_consequences = st.selectbox("Legal Consequences", ['No', 'Yes'])
            relationship_strain = st.selectbox("Relationship Strain", ['No', 'Yes'])
            risk_taking = st.selectbox("Risk Taking Behavior", ['No', 'Yes'])
            withdrawal = st.selectbox("Withdrawal Symptoms", ['No', 'Yes'])
            treatment_resistance = st.selectbox("Treatment Resistance", ['No', 'Yes'])
        
        submitted = st.form_submit_button("Predict Risk")
        
        if submitted:
            # Create input vector
            input_data = {
                'Experimentation': experimentation,
                'Academic_Performance_Decline': academic_decline,
                'Social_Isolation': social_isolation,
                'Financial_Issues': financial_issues,
                'Physical_Mental_Health_Problems': health_problems,
                'Legal_Consequences': legal_consequences,
                'Relationship_Strain': relationship_strain,
                'Risk_Taking_Behavior': risk_taking,
                'Withdrawal_Symptoms': withdrawal,
                'Denial_and_Resistance_to_Treatment': treatment_resistance
            }
            
            # Make prediction
            try:
                result = predictor.predict_individual(input_data)
                if len(result) == 3:
                    prediction, probability, insights = result
                else:
                    prediction, probability = result[:2]
                    insights = {}
            except:
                prediction, probability = predictor.predict_individual(input_data)
                insights = {}
            
            # Display results
            if prediction == 'Yes':
                st.error(f"High addiction risk detected (Probability: {probability:.3f})")
                
                # Show enhanced insights if available
                if insights:
                    risk_level = insights.get('risk_level', 'High Risk')
                    user_segment = insights.get('user_segment', 'Unknown')
                    st.info(f"Risk Level: **{risk_level}** | User Segment: **{user_segment}**")
                    
                    # Show personalized recommendations
                    recommendations = insights.get('recommendations', [])
                    if recommendations:
                        st.markdown("### Personalized Recommendations:")
                        for rec in recommendations:
                            priority_color = {
                                'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'
                            }.get(rec.get('priority', 'Medium'), '🟡')
                            st.markdown(f"{priority_color} **{rec['title']}**")
                            st.markdown(f"   - {rec['description']}")
                            st.markdown(f"   - *Action: {rec['action']}*")
                    else:
                        st.markdown("### Recommended Actions:")
                        st.markdown("- Immediate intervention recommended")
                        st.markdown("- Consider professional counseling")
                        st.markdown("- Engage support network")
                else:
                    st.markdown("### Recommended Actions:")
                    st.markdown("- Immediate intervention recommended")
                    st.markdown("- Consider professional counseling")
                    st.markdown("- Engage support network")
            else:
                st.success(f"Low addiction risk (Probability: {probability:.3f})")
                
                if insights:
                    risk_level = insights.get('risk_level', 'Low Risk')
                    user_segment = insights.get('user_segment', 'Unknown')
                    st.info(f"Risk Level: **{risk_level}** | User Segment: **{user_segment}**")
                
                st.markdown("### Preventive Measures:")
                st.markdown("- Continue healthy habits")
                st.markdown("- Regular check-ins with support system")
                st.markdown("- Monitor risk factors")

def show_community_insights(df, community_insights):
    st.markdown('<h2 class="section-header">Community Website Insights & Recommendations</h2>', unsafe_allow_html=True)
    
    # Analyze user behavior patterns
    user_patterns = community_insights.analyze_user_patterns(df)
    
    # Display user segments
    st.markdown("### User Behavior Segments")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**High-Risk Users**")
        st.metric("Percentage", f"{user_patterns['high_risk_percentage']:.1f}%")
        st.markdown("- Need immediate support")
        st.markdown("- Require crisis intervention features")
        st.markdown("- Benefit from peer mentoring")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Moderate-Risk Users**")
        st.metric("Percentage", f"{user_patterns['moderate_risk_percentage']:.1f}%")
        st.markdown("- Benefit from educational content")
        st.markdown("- Need progress tracking tools")
        st.markdown("- Respond to community support")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Low-Risk Users**")
        st.metric("Percentage", f"{user_patterns['low_risk_percentage']:.1f}%")
        st.markdown("- Can serve as peer mentors")
        st.markdown("- Benefit from prevention content")
        st.markdown("- Good candidates for success stories")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Website feature recommendations
    st.markdown("### Recommended Website Features")
    
    recommendations = community_insights.generate_website_recommendations(df)
    
    for category, features in recommendations.items():
        st.markdown(f"#### {category}")
        for feature in features:
            st.markdown(f"- {feature}")
    
    # Community engagement strategies
    st.markdown("### Community Engagement Strategies")
    
    engagement_strategies = community_insights.get_engagement_strategies(user_patterns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**High Engagement Features:**")
        for strategy in engagement_strategies['high_engagement']:
            st.markdown(f"✓ {strategy}")
    
    with col2:
        st.markdown("**Retention Strategies:**")
        for strategy in engagement_strategies['retention']:
            st.markdown(f"✓ {strategy}")
    
    # Success metrics
    st.markdown("### Recommended Success Metrics")
    
    metrics = community_insights.get_success_metrics()
    
    for metric_category, metric_list in metrics.items():
        st.markdown(f"**{metric_category}:**")
        for metric in metric_list:
            st.markdown(f"- {metric}")

def show_real_time_monitoring():
    st.markdown('<h2 class="section-header">Real-Time Data Monitoring</h2>', unsafe_allow_html=True)
    
    rt_ingestion = st.session_state.rt_ingestion
    rt_visualization = st.session_state.rt_visualization
    
    # Monitoring controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Start Monitoring", key="start_monitoring"):
            if rt_ingestion.start_monitoring(interval_seconds=30):
                st.success("Real-time monitoring started!")
                st.rerun()
            else:
                st.warning("Monitoring is already active")
    
    with col2:
        if st.button("Stop Monitoring", key="stop_monitoring"):
            rt_ingestion.stop_monitoring()
            st.success("Real-time monitoring stopped!")
            st.rerun()
    
    with col3:
        if st.button("Clear Alerts", key="clear_alerts"):
            rt_ingestion.clear_alerts()
            # Also clear session state alerts
            if 'alerts' in st.session_state:
                st.session_state.alerts = []
            st.success("Alerts cleared!")
            st.rerun()
    
    # Get current monitoring statistics
    monitoring_stats = rt_ingestion.get_monitoring_stats()
    
    # Display real-time metrics
    st.markdown("### Real-Time Metrics")
    rt_visualization.create_real_time_metrics(monitoring_stats)
    
    # Display alerts (get fresh alerts from thread-safe queue)
    alerts = rt_ingestion.get_alerts()
    if alerts:
        # Store alerts in session state for persistence
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        st.session_state.alerts.extend(alerts)
        # Keep only recent alerts
        st.session_state.alerts = st.session_state.alerts[-50:]
    
    # Display all stored alerts
    stored_alerts = st.session_state.get('alerts', [])
    rt_visualization.display_alerts(stored_alerts)
    
    # Get and display recent data
    st.markdown("### Live Data Stream")
    recent_data = rt_ingestion.get_recent_data(minutes=60)
    
    if not recent_data.empty:
        # Live chart
        rt_visualization.create_live_chart(recent_data)
        
        # Recent data table
        with st.expander("Recent Data Details", expanded=False):
            st.markdown(f"Showing last {len(recent_data)} records from the past hour")
            st.dataframe(recent_data.sort_values('timestamp', ascending=False), use_container_width=True)
        
        # Export functionality
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Recent Data", key="export_recent"):
                try:
                    filename = rt_ingestion.export_monitoring_data()
                    if filename:
                        st.success(f"Data exported to {filename}")
                    else:
                        st.warning("No data to export")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
        
        with col2:
            # Download button for recent data
            csv_data = recent_data.to_csv(index=False)
            st.download_button(
                label="Download Recent Data",
                data=csv_data,
                file_name=f"recent_monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_recent"
            )
    else:
        st.info("No recent data available. Start monitoring to begin collecting real-time data.")
    
    # Alert configuration
    with st.expander("Alert Configuration", expanded=False):
        st.markdown("### Configure Monitoring Alerts")
        
        current_thresholds = rt_ingestion.alert_thresholds
        
        new_high_risk_threshold = st.slider(
            "High Risk Rate Alert (%)",
            min_value=10.0,
            max_value=50.0,
            value=current_thresholds['high_risk_rate'],
            step=5.0,
            help="Alert when percentage of high-risk cases exceeds this threshold"
        )
        
        new_crisis_threshold = st.slider(
            "Crisis Indicators Alert",
            min_value=1,
            max_value=20,
            value=current_thresholds['crisis_indicators'],
            step=1,
            help="Alert when number of crisis indicators exceeds this threshold"
        )
        
        if st.button("Update Alert Thresholds", key="update_thresholds"):
            rt_ingestion.configure_alerts({
                'high_risk_rate': new_high_risk_threshold,
                'crisis_indicators': new_crisis_threshold
            })
            st.success("Alert thresholds updated!")
    
    # Auto-refresh toggle and control
    col1, col2 = st.columns(2)
    with col1:
        auto_refresh = st.toggle("Auto-refresh Dashboard", value=False, key="auto_refresh_toggle")
    
    with col2:
        refresh_interval = st.selectbox(
            "Refresh Interval", 
            options=[10, 15, 30, 60], 
            index=2,
            format_func=lambda x: f"{x} seconds",
            key="refresh_interval"
        )
    
    # Auto-refresh mechanism
    if auto_refresh and monitoring_stats.get('monitoring_active', False):
        # Use a placeholder for live updates
        placeholder = st.empty()
        with placeholder.container():
            st.info(f"Auto-refreshing every {refresh_interval} seconds...")
        
        time.sleep(refresh_interval)
        st.rerun()

def show_user_segmentation(df, predictor):
    """Display comprehensive user segmentation analysis."""
    st.title("User Segmentation Analysis")
    st.markdown("*Advanced clustering analysis to identify distinct user groups and targeted intervention strategies*")
    
    # Check if ML model is trained
    if not hasattr(predictor, 'is_trained') or not predictor.is_trained:
        st.warning("Machine learning model needs to be trained first. Please go to the ML Predictions page to train the model.")
        
        if st.button("Train Model Now"):
            with st.spinner("Training machine learning model..."):
                accuracy, report, feature_importance = predictor.train_model(df)
                if accuracy > 0:
                    st.success(f"Model trained successfully! Accuracy: {accuracy:.3f}")
                    st.rerun()
                else:
                    st.error("Model training failed. Please check the data quality and try again.")
        return
    
    # Main segmentation analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Segmentation Overview")
        
        # Get segmentation data
        if hasattr(predictor, 'get_user_segmentation_analysis'):
            segmentation_data = predictor.get_user_segmentation_analysis(df)
        else:
            # Fallback implementation if method doesn't exist
            segmentation_data = _calculate_segmentation_fallback(df, predictor)
        
        if segmentation_data and 'segments' in segmentation_data:
            segments = segmentation_data['segments']
            total_users = segmentation_data.get('total_users', len(df))
            
            # Display segment statistics
            for segment_name, segment_info in segments.items():
                size = segment_info['size']
                percentage = segment_info['percentage']
                addiction_rate = segment_info['addiction_rate']
                risk_level = segment_info['risk_level']
                
                # Color coding based on risk level
                if 'Critical' in risk_level or 'Crisis' in segment_name:
                    color = "🔴"
                elif 'High' in risk_level:
                    color = "🟠"
                elif 'Moderate' in risk_level:
                    color = "🟡"
                else:
                    color = "🟢"
                
                st.markdown(f"{color} **{segment_name}**")
                st.markdown(f"   - Size: {size} users ({percentage:.1f}%)")
                st.markdown(f"   - Addiction Rate: {addiction_rate:.1f}%")
                st.markdown(f"   - Risk Level: {risk_level}")
                st.markdown("")
        
        else:
            st.error("Unable to generate segmentation analysis. Please ensure the model is properly trained.")
    
    with col2:
        st.markdown("### Segment Distribution")
        
        # Create segmentation visualization
        if hasattr(predictor, 'plot_user_segmentation'):
            fig = predictor.plot_user_segmentation(df)
            if fig.data:  # Check if figure has data
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Segmentation visualization not available")
        else:
            # Create fallback pie chart
            if segmentation_data and 'segments' in segmentation_data:
                _create_fallback_segmentation_chart(segmentation_data['segments'])
    
    # Detailed segment analysis
    st.markdown("---")
    st.markdown("### Detailed Segment Analysis")
    
    if segmentation_data and 'segments' in segmentation_data:
        # Create tabs for each segment
        segment_names = list(segmentation_data['segments'].keys())
        tabs = st.tabs(segment_names)
        
        for i, (tab, segment_name) in enumerate(zip(tabs, segment_names)):
            with tab:
                segment_info = segmentation_data['segments'][segment_name]
                _display_segment_details(segment_name, segment_info, df)
    
    # Segment-specific recommendations
    st.markdown("---")
    st.markdown("### Intervention Strategies by Segment")
    
    intervention_strategies = {
        "Low Risk - Stable": {
            "strategy": "Prevention and Maintenance",
            "interventions": [
                "Regular wellness check-ins",
                "Educational workshops on addiction prevention",
                "Peer support group participation",
                "Healthy lifestyle promotion programs"
            ],
            "frequency": "Quarterly",
            "priority": "Low"
        },
        "Moderate Risk - Academic Stress": {
            "strategy": "Academic Support and Stress Management",
            "interventions": [
                "Academic counseling and tutoring services",
                "Stress management workshops",
                "Time management skill building",
                "Mental health counseling"
            ],
            "frequency": "Monthly", 
            "priority": "Medium"
        },
        "High Risk - Multiple Factors": {
            "strategy": "Comprehensive Treatment Approach",
            "interventions": [
                "Intensive counseling and therapy",
                "Multi-disciplinary treatment team",
                "Family involvement and therapy",
                "Skills training and relapse prevention"
            ],
            "frequency": "Weekly",
            "priority": "High"
        },
        "Crisis - Immediate Intervention": {
            "strategy": "Emergency Intervention and Stabilization",
            "interventions": [
                "24/7 crisis intervention services",
                "Medical supervision and detoxification",
                "Inpatient treatment consideration",
                "Emergency family notification and support"
            ],
            "frequency": "Immediate/Daily",
            "priority": "Critical"
        }
    }
    
    for segment, strategy in intervention_strategies.items():
        priority_color = {
            'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'
        }.get(strategy['priority'], '🟡')
        
        st.markdown(f"{priority_color} **{segment}**")
        st.markdown(f"**Strategy:** {strategy['strategy']}")
        st.markdown(f"**Priority:** {strategy['priority']} | **Frequency:** {strategy['frequency']}")
        st.markdown("**Key Interventions:**")
        for intervention in strategy['interventions']:
            st.markdown(f"- {intervention}")
        st.markdown("")

def _calculate_segmentation_fallback(df, predictor):
    """Fallback method to calculate segmentation if the predictor method doesn't exist."""
    try:
        # Prepare features
        X, y = predictor.prepare_features(df)
        X_scaled = predictor.scaler.transform(X)
        X_selected = predictor.feature_selector.transform(X_scaled)
        
        # Get cluster assignments
        cluster_labels = predictor.clustering_model.predict(X_selected)
        
        # Analyze segments
        segment_analysis = {}
        for segment_id, segment_name in predictor.user_segments.items():
            segment_mask = cluster_labels == segment_id
            segment_size = int(segment_mask.sum())
            
            if segment_size > 0:
                segment_addiction_rate = float(y[segment_mask].mean() * 100)
                segment_analysis[segment_name] = {
                    'size': segment_size,
                    'percentage': float(segment_size / len(y) * 100),
                    'addiction_rate': segment_addiction_rate,
                    'risk_level': _categorize_risk_level_fallback(segment_addiction_rate / 100)
                }
        
        return {
            'segments': segment_analysis,
            'total_users': len(y)
        }
        
    except Exception as e:
        st.error(f"Error calculating segmentation: {str(e)}")
        return {}

def _categorize_risk_level_fallback(probability):
    """Fallback risk level categorization."""
    if probability >= 0.8:
        return "Critical Risk"
    elif probability >= 0.6:
        return "High Risk"
    elif probability >= 0.4:
        return "Moderate Risk"
    elif probability >= 0.2:
        return "Low Risk"
    else:
        return "Very Low Risk"

def _create_fallback_segmentation_chart(segments):
    """Create fallback segmentation chart."""
    import plotly.graph_objects as go
    
    segment_names = list(segments.keys())
    sizes = [segments[name]['size'] for name in segment_names]
    
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    
    fig = go.Figure(data=[
        go.Pie(
            labels=segment_names,
            values=sizes,
            marker=dict(colors=colors[:len(segment_names)]),
            textinfo='label+percent',
            hole=0.3
        )
    ])
    
    fig.update_layout(
        title="User Segment Distribution",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _display_segment_details(segment_name, segment_info, df):
    """Display detailed information for a specific segment."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Segment Size", f"{segment_info['size']:,}", f"{segment_info['percentage']:.1f}%")
    
    with col2:
        st.metric("Addiction Rate", f"{segment_info['addiction_rate']:.1f}%")
    
    with col3:
        st.metric("Risk Level", segment_info['risk_level'])
    
    with col4:
        if segment_info['addiction_rate'] > 50:
            delta_color = "inverse"
        else:
            delta_color = "normal"
        
        overall_rate = (df['Addiction_Class'] == 'Yes').mean() * 100
        delta = segment_info['addiction_rate'] - overall_rate
        st.metric("vs Overall Rate", f"{delta:+.1f}%", delta=f"{delta:.1f}%", delta_color=delta_color)
    
    # Segment characteristics
    st.markdown("#### Segment Characteristics")
    
    # Key risk factors for this segment
    segment_descriptions = {
        "Low Risk - Stable": "Users with minimal risk factors and stable lifestyle patterns. Good candidates for prevention programs.",
        "Moderate Risk - Academic Stress": "Users primarily struggling with academic performance and related stress. Academic support is key.",
        "High Risk - Multiple Factors": "Users with multiple concurrent risk factors requiring comprehensive intervention.",
        "Crisis - Immediate Intervention": "Users with critical risk factors requiring immediate professional attention."
    }
    
    if segment_name in segment_descriptions:
        st.info(segment_descriptions[segment_name])
    
    # Treatment outcomes and prognosis
    treatment_outcomes = {
        "Low Risk - Stable": {"success_rate": "95%", "prognosis": "Excellent", "timeline": "N/A - Preventive"},
        "Moderate Risk - Academic Stress": {"success_rate": "80%", "prognosis": "Good", "timeline": "3-6 months"},
        "High Risk - Multiple Factors": {"success_rate": "65%", "prognosis": "Moderate", "timeline": "6-12 months"},
        "Crisis - Immediate Intervention": {"success_rate": "45%", "prognosis": "Guarded", "timeline": "12+ months"}
    }
    
    if segment_name in treatment_outcomes:
        outcome = treatment_outcomes[segment_name]
        st.markdown("#### Treatment Outlook")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Success Rate", outcome["success_rate"])
        with col2:
            st.metric("Prognosis", outcome["prognosis"])  
        with col3:
            st.metric("Timeline", outcome["timeline"])

if __name__ == "__main__":
    main()
