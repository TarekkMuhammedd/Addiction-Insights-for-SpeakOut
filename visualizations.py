import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from typing import Dict, List, Tuple, Any
import streamlit as st

class DashboardVisualizations:
    """Handles all visualization components for the dashboard."""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#28a745',
            'secondary': '#6c757d',
            'success': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
    
    def calculate_big_numbers(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key metrics for big number displays."""
        total_cases = len(df)
        addiction_cases = (df['Addiction_Class'] == 'Yes').sum()
        non_addiction_cases = total_cases - addiction_cases
        
        addiction_rate = (addiction_cases / total_cases) * 100 if total_cases > 0 else 0
        
        # Calculate rates for various risk factors
        academic_decline_rate = (df['Academic_Performance_Decline'] == 'Yes').mean() * 100
        social_isolation_rate = (df['Social_Isolation'] == 'Yes').mean() * 100
        financial_issues_rate = (df['Financial_Issues'] == 'Yes').mean() * 100
        health_problems_rate = (df['Physical_Mental_Health_Problems'] == 'Yes').mean() * 100
        treatment_resistance_rate = (df['Denial_and_Resistance_to_Treatment'] == 'Yes').mean() * 100
        
        return {
            'total_cases': total_cases,
            'addiction_cases': addiction_cases,
            'non_addiction_cases': non_addiction_cases,
            'addiction_rate': addiction_rate,
            'academic_decline_rate': academic_decline_rate,
            'social_isolation_rate': social_isolation_rate,
            'financial_issues_rate': financial_issues_rate,
            'health_problems_rate': health_problems_rate,
            'treatment_resistance_rate': treatment_resistance_rate
        }
    
    def create_severity_gauge(self, value: float, title: str) -> go.Figure:
        """Create a gauge chart for severity metrics."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': self.color_palette['primary']},
                'steps': [
                    {'range': [0, 30], 'color': self.color_palette['light']},
                    {'range': [30, 70], 'color': '#ffffcc'},
                    {'range': [70, 100], 'color': '#ffcccc'}
                ],
                'threshold': {
                    'line': {'color': self.color_palette['danger'], 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        
        return fig
    
    def create_addiction_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create addiction class distribution chart."""
        addiction_counts = df['Addiction_Class'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=addiction_counts.index,
                y=addiction_counts.values,
                marker_color=[self.color_palette['danger'] if x == 'Yes' else self.color_palette['success'] 
                             for x in addiction_counts.index],
                text=addiction_counts.values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Addiction Class Distribution",
            xaxis_title="Addiction Status",
            yaxis_title="Number of Cases",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_risk_factors_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create overview of risk factors."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        factor_percentages = []
        factor_names = []
        
        for factor in available_factors:
            percentage = (df[factor] == 'Yes').mean() * 100
            factor_percentages.append(percentage)
            factor_names.append(factor.replace('_', ' '))
        
        fig = go.Figure(data=[
            go.Bar(
                x=factor_names,
                y=factor_percentages,
                marker_color=self.color_palette['primary'],
                text=[f'{p:.1f}%' for p in factor_percentages],
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title="Risk Factors Prevalence",
            xaxis_title="Risk Factors",
            yaxis_title="Percentage (%)",
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    def get_top_risk_factor(self, df: pd.DataFrame) -> str:
        """Get the most common risk factor."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        max_percentage = 0
        top_factor = "Unknown"
        
        for factor in available_factors:
            percentage = (df[factor] == 'Yes').mean() * 100
            if percentage > max_percentage:
                max_percentage = percentage
                top_factor = factor.replace('_', ' ')
        
        return f"{top_factor} ({max_percentage:.1f}%)"
    
    def create_group_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create comparison chart between addicted and non-addicted groups."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        addicted_group = df[df['Addiction_Class'] == 'Yes']
        non_addicted_group = df[df['Addiction_Class'] == 'No']
        
        addicted_rates = []
        non_addicted_rates = []
        factor_names = []
        
        for factor in available_factors:
            addicted_rate = (addicted_group[factor] == 'Yes').mean() * 100
            non_addicted_rate = (non_addicted_group[factor] == 'Yes').mean() * 100
            
            addicted_rates.append(addicted_rate)
            non_addicted_rates.append(non_addicted_rate)
            factor_names.append(factor.replace('_', ' '))
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Addicted',
            x=factor_names,
            y=addicted_rates,
            marker_color=self.color_palette['danger']
        ))
        
        fig.add_trace(go.Bar(
            name='Non-Addicted',
            x=factor_names,
            y=non_addicted_rates,
            marker_color=self.color_palette['success']
        ))
        
        fig.update_layout(
            title="Risk Factors by Addiction Status",
            xaxis_title="Risk Factors",
            yaxis_title="Percentage (%)",
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    def create_risk_factor_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create radar chart comparing risk factor profiles."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        addicted_group = df[df['Addiction_Class'] == 'Yes']
        non_addicted_group = df[df['Addiction_Class'] == 'No']
        
        addicted_rates = []
        non_addicted_rates = []
        factor_names = []
        
        for factor in available_factors:
            addicted_rate = (addicted_group[factor] == 'Yes').mean() * 100
            non_addicted_rate = (non_addicted_group[factor] == 'Yes').mean() * 100
            
            addicted_rates.append(addicted_rate)
            non_addicted_rates.append(non_addicted_rate)
            factor_names.append(factor.replace('_', ' '))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=addicted_rates,
            theta=factor_names,
            fill='toself',
            name='Addicted',
            line_color=self.color_palette['danger']
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=non_addicted_rates,
            theta=factor_names,
            fill='toself',
            name='Non-Addicted',
            line_color=self.color_palette['success']
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Risk Factor Profile Comparison",
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed comparison table."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        addicted_group = df[df['Addiction_Class'] == 'Yes']
        non_addicted_group = df[df['Addiction_Class'] == 'No']
        
        comparison_data = []
        
        for factor in available_factors:
            addicted_rate = (addicted_group[factor] == 'Yes').mean() * 100
            non_addicted_rate = (non_addicted_group[factor] == 'Yes').mean() * 100
            difference = addicted_rate - non_addicted_rate
            
            comparison_data.append({
                'Risk Factor': factor.replace('_', ' '),
                'Addicted Group (%)': f"{addicted_rate:.1f}",
                'Non-Addicted Group (%)': f"{non_addicted_rate:.1f}",
                'Difference (%)': f"{difference:.1f}"
            })
        
        return pd.DataFrame(comparison_data)
    
    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Perform statistical significance tests."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        results = {}
        
        for factor in available_factors:
            # Create contingency table
            contingency = pd.crosstab(df[factor], df['Addiction_Class'])
            
            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            
            results[factor.replace('_', ' ')] = {
                'chi2': chi2,
                'p_value': p_value,
                'degrees_of_freedom': dof
            }
        
        return results
    
    def create_addiction_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart for addiction distribution."""
        addiction_counts = df['Addiction_Class'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=addiction_counts.index,
            values=addiction_counts.values,
            hole=0.3,
            marker_colors=[self.color_palette['danger'] if x == 'Yes' else self.color_palette['success'] 
                          for x in addiction_counts.index]
        )])
        
        fig.update_layout(
            title="Addiction Status Distribution",
            annotations=[dict(text='Cases', x=0.5, y=0.5, font_size=20, showarrow=False)],
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_risk_factors_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart for risk level distribution."""
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts()
            colors = {
                'Low': self.color_palette['success'],
                'Moderate': self.color_palette['warning'],
                'High': self.color_palette['danger']
            }
        else:
            # Create risk levels based on total risk factors
            risk_factors = df[['Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues']].apply(lambda x: x == 'Yes').sum(axis=1)
            risk_levels = risk_factors.apply(lambda x: 'High' if x >= 2 else 'Moderate' if x == 1 else 'Low')
            risk_counts = risk_levels.value_counts()
            colors = {
                'Low': self.color_palette['success'],
                'Moderate': self.color_palette['warning'],
                'High': self.color_palette['danger']
            }
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.3,
            marker_colors=[colors.get(str(x), self.color_palette['secondary']) for x in risk_counts.index]
        )])
        
        fig.update_layout(
            title="Risk Level Distribution",
            annotations=[dict(text='Risk', x=0.5, y=0.5, font_size=20, showarrow=False)],
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_stacked_bar_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create stacked bar chart for risk factors by addiction status."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        fig = go.Figure()
        
        for addiction_status in ['Yes', 'No']:
            group_data = df[df['Addiction_Class'] == addiction_status]
            percentages = []
            
            for factor in available_factors:
                percentage = (group_data[factor] == 'Yes').mean() * 100
                percentages.append(percentage)
            
            fig.add_trace(go.Bar(
                name=f'Addicted: {addiction_status}',
                x=[factor.replace('_', ' ') for factor in available_factors],
                y=percentages,
                marker_color=self.color_palette['danger'] if addiction_status == 'Yes' else self.color_palette['success']
            ))
        
        fig.update_layout(
            title="Risk Factor Prevalence by Addiction Status",
            xaxis_title="Risk Factors",
            yaxis_title="Percentage (%)",
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis={'tickangle': 45}
        )
        
        return fig
    
    def get_composition_breakdown(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get detailed composition breakdown."""
        composition = {}
        
        # Addiction status composition
        addiction_counts = df['Addiction_Class'].value_counts()
        composition['Addiction Status'] = {
            status: (count / len(df)) * 100 
            for status, count in addiction_counts.items()
        }
        
        # Risk level composition
        if 'Risk_Level' in df.columns:
            risk_counts = df['Risk_Level'].value_counts()
            composition['Risk Level'] = {
                level: (count / len(df)) * 100 
                for level, count in risk_counts.items()
            }
        
        # Key risk factors composition
        key_factors = ['Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues']
        available_key_factors = [factor for factor in key_factors if factor in df.columns]
        
        for factor in available_key_factors:
            factor_counts = df[factor].value_counts()
            composition[factor.replace('_', ' ')] = {
                status: (count / len(df)) * 100 
                for status, count in factor_counts.items()
            }
        
        return composition
    
    def simulate_addiction_progression(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate addiction progression over time."""
        # Create time series data based on risk factors
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='M')
        
        # Base rates from the data
        base_addiction_rate = (df['Addiction_Class'] == 'Yes').mean()
        base_risk_factors = df[['Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues']].apply(lambda x: x == 'Yes').sum(axis=1).mean()
        
        # Simulate progression with some trends
        progression_data = []
        
        for i, date in enumerate(dates):
            # Add some seasonal and trend effects
            trend_factor = 1 + (i * 0.002)  # Slight upward trend
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Seasonal variation
            
            addiction_rate = min(1.0, base_addiction_rate * trend_factor * seasonal_factor)
            risk_factor_avg = min(8.0, base_risk_factors * trend_factor * seasonal_factor)
            
            # Add some noise
            noise = np.random.normal(0, 0.05)
            addiction_rate = max(0, min(1, addiction_rate + noise))
            
            progression_data.append({
                'Date': date,
                'Addiction_Rate': addiction_rate * 100,
                'Average_Risk_Factors': risk_factor_avg,
                'Recovery_Rate': max(0, 30 - addiction_rate * 20),  # Inverse relationship
                'New_Cases': max(0, int(100 * addiction_rate * (1 + noise)))
            })
        
        return pd.DataFrame(progression_data)
    
    def create_progression_timeline(self, time_series_data: pd.DataFrame) -> go.Figure:
        """Create addiction progression timeline."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_series_data['Date'],
            y=time_series_data['Addiction_Rate'],
            mode='lines+markers',
            name='Addiction Rate (%)',
            line=dict(color=self.color_palette['danger'], width=3),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Addiction Rate Progression Over Time",
            xaxis_title="Date",
            yaxis_title="Addiction Rate (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig
    
    def create_recovery_timeline(self, time_series_data: pd.DataFrame) -> go.Figure:
        """Create recovery rate timeline."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time_series_data['Date'],
            y=time_series_data['Recovery_Rate'],
            mode='lines+markers',
            name='Recovery Rate (%)',
            line=dict(color=self.color_palette['success'], width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(40, 167, 69, 0.2)'
        ))
        
        fig.update_layout(
            title="Recovery Rate Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Recovery Rate (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True
        )
        
        return fig
    
    def create_trend_analysis(self, time_series_data: pd.DataFrame) -> go.Figure:
        """Create comprehensive trend analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Addiction Rate', 'Average Risk Factors', 'Recovery Rate', 'New Cases'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Addiction Rate
        fig.add_trace(
            go.Scatter(x=time_series_data['Date'], y=time_series_data['Addiction_Rate'],
                      mode='lines', name='Addiction Rate', line=dict(color=self.color_palette['danger'])),
            row=1, col=1
        )
        
        # Average Risk Factors
        fig.add_trace(
            go.Scatter(x=time_series_data['Date'], y=time_series_data['Average_Risk_Factors'],
                      mode='lines', name='Avg Risk Factors', line=dict(color=self.color_palette['warning'])),
            row=1, col=2
        )
        
        # Recovery Rate
        fig.add_trace(
            go.Scatter(x=time_series_data['Date'], y=time_series_data['Recovery_Rate'],
                      mode='lines', name='Recovery Rate', line=dict(color=self.color_palette['success'])),
            row=2, col=1
        )
        
        # New Cases
        fig.add_trace(
            go.Bar(x=time_series_data['Date'], y=time_series_data['New_Cases'],
                  name='New Cases', marker_color=self.color_palette['info']),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Comprehensive Trend Analysis",
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap."""
        # Select numeric columns or convert categorical to numeric
        numeric_columns = []
        correlation_df = df.copy()
        
        for col in df.columns:
            if df[col].dtype == 'object':
                correlation_df[col] = (correlation_df[col] == 'Yes').astype(int)
            numeric_columns.append(col)
        
        # Calculate correlation matrix
        corr_matrix = correlation_df[numeric_columns].corr(method='pearson')
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[col.replace('_', ' ') for col in corr_matrix.columns],
            y=[col.replace('_', ' ') for col in corr_matrix.index],
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Risk Factor Correlation Matrix",
            width=800,
            height=600,
            xaxis={'tickangle': 45},
            paper_bgcolor='white'
        )
        
        return fig
    
    def get_correlation_insights(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Get correlation insights."""
        # Convert categorical to numeric
        correlation_df = df.copy()
        for col in df.columns:
            if df[col].dtype == 'object':
                correlation_df[col] = (correlation_df[col] == 'Yes').astype(int)
        
        corr_matrix = correlation_df.corr()
        
        # Find strong correlations
        strong_positive = {}
        strong_negative = {}
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                var1 = corr_matrix.columns[i].replace('_', ' ')
                var2 = corr_matrix.columns[j].replace('_', ' ')
                
                if corr_value > 0.3:
                    strong_positive[f"{var1} ↔ {var2}"] = corr_value
                elif corr_value < -0.3:
                    strong_negative[f"{var1} ↔ {var2}"] = corr_value
        
        # Sort by absolute correlation strength
        strong_positive = dict(sorted(strong_positive.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        strong_negative = dict(sorted(strong_negative.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
        
        return {
            'positive': strong_positive,
            'negative': strong_negative
        }
    
    def create_network_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create network analysis of risk factors."""
        # This is a simplified network visualization
        # In a real implementation, you might use networkx for more complex network analysis
        
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        # Calculate co-occurrence matrix
        co_occurrence = {}
        for i, factor1 in enumerate(available_factors):
            for j, factor2 in enumerate(available_factors):
                if i != j:
                    # Count cases where both factors are 'Yes'
                    both_yes = ((df[factor1] == 'Yes') & (df[factor2] == 'Yes')).sum()
                    total_cases = len(df)
                    co_occurrence[f"{factor1}-{factor2}"] = both_yes / total_cases
        
        # Create a simple circular layout
        n_factors = len(available_factors)
        angles = np.linspace(0, 2 * np.pi, n_factors, endpoint=False)
        
        x_positions = np.cos(angles)
        y_positions = np.sin(angles)
        
        fig = go.Figure()
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers+text',
            marker=dict(size=30, color=self.color_palette['primary']),
            text=[factor.replace('_', ' ') for factor in available_factors],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            name='Risk Factors'
        ))
        
        # Add edges for strong connections
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                factor1 = available_factors[i]
                factor2 = available_factors[j]
                connection_strength = co_occurrence.get(f"{factor1}-{factor2}", 0)
                
                if connection_strength > 0.1:  # Threshold for showing connections
                    fig.add_trace(go.Scatter(
                        x=[x_positions[i], x_positions[j]],
                        y=[y_positions[i], y_positions[j]],
                        mode='lines',
                        line=dict(width=connection_strength*20, color='rgba(40, 167, 69, 0.5)'),
                        showlegend=False,
                        hoverinfo='none'
                    ))
        
        fig.update_layout(
            title="Risk Factor Network Analysis",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=500
        )
        
        return fig
