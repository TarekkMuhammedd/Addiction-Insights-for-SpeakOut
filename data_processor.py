import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, List

class DataProcessor:
    """Handles data loading, cleaning, and preprocessing for addiction analysis."""
    
    def __init__(self):
        self.categorical_columns = [
            'Experimentation', 'Academic_Performance_Decline', 'Social_Isolation',
            'Financial_Issues', 'Physical_Mental_Health_Problems', 'Legal_Consequences',
            'Relationship_Strain', 'Risk_Taking_Behavior', 'Withdrawal_Symptoms',
            'Denial_and_Resistance_to_Treatment', 'Addiction_Class'
        ]
    
    def load_data(self, file_path) -> Optional[pd.DataFrame]:
        """Load data from CSV file or uploaded file object."""
        try:
            if isinstance(file_path, str):
                # File path string
                df = pd.read_csv(file_path)
            else:
                # Uploaded file object
                df = pd.read_csv(file_path)
            
            st.success(f"Successfully loaded {len(df)} records")
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the dataset."""
        df_processed = df.copy()
        
        # Display data quality info
        self._display_data_quality(df_processed)
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Standardize categorical values
        df_processed = self._standardize_categories(df_processed)
        
        # Create derived features
        df_processed = self._create_derived_features(df_processed)
        
        return df_processed
    
    def _display_data_quality(self, df: pd.DataFrame):
        """Display data quality information."""
        st.sidebar.markdown("### Data Quality")
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        st.sidebar.metric("Data Completeness", f"{completeness:.1f}%")
        st.sidebar.metric("Missing Values", missing_cells)
        
        # Show missing values by column
        missing_by_column = df.isnull().sum()
        if missing_by_column.sum() > 0:
            st.sidebar.markdown("**Missing by Column:**")
            for col, missing in missing_by_column.items():
                if missing > 0:
                    pct = (missing / len(df)) * 100
                    st.sidebar.text(f"{col}: {missing} ({pct:.1f}%)")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        df_processed = df.copy()
        
        # For categorical columns, fill missing with mode
        for col in self.categorical_columns:
            if col in df_processed.columns:
                mode_value = df_processed[col].mode()
                if len(mode_value) > 0:
                    df_processed[col] = df_processed[col].fillna(mode_value.iloc[0])
                else:
                    # If no mode (all values missing), fill with 'No'
                    df_processed[col] = df_processed[col].fillna('No')
        
        return df_processed
    
    def _standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values to Yes/No format."""
        df_processed = df.copy()
        
        # Define mapping for different variations
        yes_values = ['yes', 'Yes', 'YES', 'y', 'Y', '1', 1, True, 'true', 'True']
        no_values = ['no', 'No', 'NO', 'n', 'N', '0', 0, False, 'false', 'False']
        
        for col in self.categorical_columns:
            if col in df_processed.columns:
                # Convert to string first
                df_processed[col] = df_processed[col].astype(str)
                
                # Map values
                df_processed[col] = df_processed[col].apply(lambda x: 
                    'Yes' if x in [str(v) for v in yes_values] else 
                    'No' if x in [str(v) for v in no_values] else 
                    'No'  # Default to 'No' for unclear values
                )
        
        return df_processed
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for analysis."""
        df_processed = df.copy()
        
        # Risk factor count
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_risk_factors = [col for col in risk_factors if col in df_processed.columns]
        
        df_processed['Total_Risk_Factors'] = df_processed[available_risk_factors].apply(
            lambda row: sum(1 for val in row if val == 'Yes'), axis=1
        )
        
        # Risk level categorization
        def categorize_risk(risk_count):
            if risk_count >= 6:
                return 'High'
            elif risk_count >= 3:
                return 'Moderate'
            else:
                return 'Low'
        
        df_processed['Risk_Level'] = df_processed['Total_Risk_Factors'].apply(categorize_risk)
        
        # Recovery readiness score (inverse of resistance indicators)
        resistance_factors = ['Denial_and_Resistance_to_Treatment']
        available_resistance = [col for col in resistance_factors if col in df_processed.columns]
        
        df_processed['Recovery_Readiness'] = df_processed[available_resistance].apply(
            lambda row: sum(1 for val in row if val == 'No'), axis=1
        )
        
        return df_processed
    
    def get_feature_descriptions(self) -> Dict[str, str]:
        """Get descriptions for all features."""
        return {
            'Experimentation': 'Has experimented with addictive substances',
            'Academic_Performance_Decline': 'Shows decline in academic performance',
            'Social_Isolation': 'Experiences social isolation',
            'Financial_Issues': 'Has financial problems',
            'Physical_Mental_Health_Problems': 'Has physical or mental health issues',
            'Legal_Consequences': 'Has faced legal consequences',
            'Relationship_Strain': 'Experiences relationship problems',
            'Risk_Taking_Behavior': 'Engages in risky behaviors',
            'Withdrawal_Symptoms': 'Experiences withdrawal symptoms',
            'Denial_and_Resistance_to_Treatment': 'Shows denial or treatment resistance',
            'Addiction_Class': 'Addiction classification (target variable)',
            'Total_Risk_Factors': 'Total number of risk factors present',
            'Risk_Level': 'Categorized risk level (Low/Moderate/High)',
            'Recovery_Readiness': 'Readiness for recovery treatment'
        }
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Get comprehensive data summary."""
        return {
            'total_records': len(df),
            'total_features': len(df.columns),
            'addiction_rate': (df['Addiction_Class'] == 'Yes').mean() if 'Addiction_Class' in df.columns else 0,
            'missing_data_percentage': float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100),
            'risk_level_distribution': df['Risk_Level'].value_counts().to_dict() if 'Risk_Level' in df.columns else {},
            'most_common_risk_factors': self._get_most_common_risk_factors(df)
        }
    
    def _get_most_common_risk_factors(self, df: pd.DataFrame) -> List[str]:
        """Get the most common risk factors."""
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [col for col in risk_factors if col in df.columns]
        
        factor_counts = {}
        for factor in available_factors:
            factor_counts[factor] = (df[factor] == 'Yes').sum()
        
        # Sort by count and return top factors
        sorted_factors = sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)
        return [factor for factor, count in sorted_factors[:5]]
