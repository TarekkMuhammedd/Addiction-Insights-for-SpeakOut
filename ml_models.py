import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
import streamlit as st
from typing import Tuple, Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

class AddictionPredictor:
    """Machine learning models for addiction prediction and analysis."""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(probability=True, random_state=42),
            'neural_network': MLPClassifier(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.best_model_name = None
        self.ensemble_model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_classif, k=15)
        self.feature_names = []
        self.selected_feature_names = []
        self.is_trained = False
        self.clustering_model = KMeans(n_clusters=4, random_state=42)
        self.user_segments = {
            0: "Low Risk - Stable",
            1: "Moderate Risk - Academic Stress", 
            2: "High Risk - Multiple Factors",
            3: "Crisis - Immediate Intervention"
        }
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning."""
        # Select feature columns (exclude target)
        feature_columns = [col for col in df.columns if col != 'Addiction_Class']
        
        # Remove derived features that might cause data leakage
        exclude_features = ['Total_Risk_Factors', 'Risk_Level', 'Recovery_Readiness']
        feature_columns = [col for col in feature_columns if col not in exclude_features]
        
        X = df[feature_columns].copy()
        y = df['Addiction_Class'].copy()
        
        # Encode categorical variables
        for column in X.columns:
            if X[column].dtype == 'object':
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column].astype(str))
                self.label_encoders[column] = le
        
        # Encode target variable
        if 'target' not in self.label_encoders:
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y.astype(str))
            self.label_encoders['target'] = le_target
        else:
            y_encoded = self.label_encoders['target'].transform(y.astype(str))
        
        self.feature_names = list(X.columns)
        return X, pd.Series(y_encoded, dtype=int)
    
    def train_model(self, df: pd.DataFrame) -> Tuple[float, str, Dict[str, float]]:
        """Train and compare multiple machine learning models with advanced techniques."""
        try:
            # Prepare data
            X, y = self.prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Feature selection - adjust k to not exceed number of features
            k = min(15, X_train_scaled.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Get selected feature names
            selected_features = self.feature_selector.get_support()
            self.selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selected_features[i]]
            
            # Train multiple models and compare
            model_results = {}
            best_score = -1  # Initialize to -1 so any valid score will be better
            trained_models = []
            
            for name, model in self.models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5)
                    
                    # Train on full training set
                    model.fit(X_train_selected, y_train)
                    
                    # Predictions
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = model.predict_proba(X_test_selected)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
                    
                    model_results[name] = {
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'auc_score': auc_score
                    }
                    
                    # Track best model
                    if accuracy > best_score:
                        best_score = accuracy
                        self.best_model = model
                        self.best_model_name = name
                    
                    trained_models.append((name, model))
                    
                except Exception as e:
                    model_results[name] = {'error': str(e)}
            
            # Create ensemble model from top performers
            top_models = sorted(
                [(name, model) for name, model in trained_models], 
                key=lambda x: model_results[x[0]].get('accuracy', 0), reverse=True
            )[:3]
            
            if len(top_models) >= 2:
                ensemble_estimators = [(name, model) for name, model in top_models]
                self.ensemble_model = VotingClassifier(
                    estimators=ensemble_estimators, 
                    voting='soft'
                )
                self.ensemble_model.fit(X_train_selected, y_train)
                
                # Test ensemble performance
                ensemble_pred = self.ensemble_model.predict(X_test_selected)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                
                # Use ensemble if it performs better
                if ensemble_accuracy > best_score:
                    self.best_model = self.ensemble_model
                    self.best_model_name = 'ensemble'
                    best_score = ensemble_accuracy
            
            # Train clustering model for user segmentation
            self.clustering_model.fit(X_train_selected)
            
            # Feature importance (use Random Forest if available)
            feature_importance = {}
            if 'random_forest' in model_results and 'error' not in model_results['random_forest']:
                rf_model = next((model for name, model in trained_models if name == 'random_forest'), None)
                if rf_model:
                    feature_importance = dict(zip(self.selected_feature_names, rf_model.feature_importances_))
            elif hasattr(self.best_model, 'feature_importances_'):
                feature_importance = dict(zip(self.selected_feature_names, self.best_model.feature_importances_))
            
            self.is_trained = True
            
            # Ensure we have at least one trained model
            if not trained_models:
                return 0.0, "All models failed to train", {}
            
            # Create comprehensive report
            report_lines = []
            if self.best_model_name:
                report_lines.append(f"Best Model: {self.best_model_name.title()}")
                report_lines.append(f"Best Accuracy: {best_score:.3f}")
            else:
                report_lines.append("No best model selected")
            report_lines.append("\\nModel Comparison:")
            for name, results in model_results.items():
                if 'error' not in results:
                    report_lines.append(f"{name}: {results['accuracy']:.3f} (CV: {results['cv_mean']:.3f}±{results['cv_std']:.3f})")
            
            comprehensive_report = "\\n".join(report_lines)
            
            return float(best_score), comprehensive_report, feature_importance
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return 0.0, "Training failed", {}
    
    def predict_individual(self, input_data: Dict[str, str]) -> Tuple[str, float, Dict[str, Any]]:
        """Make enhanced prediction for individual case with personalized insights."""
        if not self.is_trained or self.best_model is None:
            return "Unknown", 0.0, {}
        
        try:
            # Create DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables with proper unseen category handling
            for column in input_df.columns:
                if column in self.label_encoders:
                    try:
                        input_df[column] = self.label_encoders[column].transform(input_df[column].astype(str))
                    except ValueError:
                        # Handle unseen categories by setting to most frequent class (0)
                        input_df[column] = 0
                else:
                    # Handle columns not seen during training
                    input_df[column] = 0
            
            # Make sure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_names]
            
            # Apply preprocessing pipeline
            input_scaled = self.scaler.transform(input_df)
            input_selected = self.feature_selector.transform(input_scaled)
            
            # Make prediction with best model
            prediction_class = self.best_model.predict(input_selected)[0]
            
            if hasattr(self.best_model, 'predict_proba'):
                prediction_proba = self.best_model.predict_proba(input_selected)[0]
                risk_probability = float(prediction_proba[1])
            else:
                # For models without probability prediction
                decision_score = self.best_model.decision_function(input_selected)[0]
                risk_probability = 1 / (1 + np.exp(-decision_score))
            
            # Convert back to original labels
            prediction_label = self.label_encoders['target'].inverse_transform([prediction_class])[0]
            
            # Get user segment
            user_segment_id = self.clustering_model.predict(input_selected)[0]
            user_segment = self.user_segments.get(user_segment_id, "Unknown Segment")
            
            # Determine risk level
            risk_level = self._categorize_risk_level(risk_probability)
            
            # Generate personalized recommendations
            recommendations = self._generate_personalized_recommendations(
                input_data, user_segment, risk_probability
            )
            
            # Calculate confidence score
            if hasattr(self.best_model, 'predict_proba'):
                confidence_score = float(np.max(prediction_proba))
            else:
                confidence_score = min(0.8, abs(decision_score) / 3.0)  # Normalize decision score
            
            # Create insights dictionary
            insights = {
                'risk_level': risk_level,
                'user_segment': user_segment,
                'recommendations': recommendations,
                'model_used': self.best_model_name,
                'confidence_score': confidence_score
            }
            
            return prediction_label, risk_probability, insights
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return "Unknown", 0.0, {}
    
    def plot_feature_importance(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance plot."""
        if not feature_importance:
            return go.Figure()
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features)
        
        # Create bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker=dict(color='#28a745'),
            text=[f'{imp:.3f}' for imp in importances],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Importance for Addiction Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_confusion_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create confusion matrix visualization."""
        if not self.is_trained or not self.best_model:
            return go.Figure()
        
        try:
            X, y = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Apply same preprocessing pipeline
            X_test_scaled = self.scaler.transform(X_test)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            y_pred = self.best_model.predict(X_test_selected)
            cm = confusion_matrix(y_test, y_pred)
            
            # Convert back to original labels
            labels = self.label_encoders['target'].inverse_transform([0, 1])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale='Greens',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16},
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                width=400,
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating confusion matrix: {str(e)}")
            return go.Figure()
    
    def get_risk_probability_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Get probability distribution for risk assessment."""
        if not self.is_trained or not self.best_model:
            return go.Figure()
        
        try:
            X, y = self.prepare_features(df)
            
            # Apply preprocessing pipeline
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Get probability predictions
            if hasattr(self.best_model, 'predict_proba'):
                probabilities = self.best_model.predict_proba(X_selected)[:, 1]  # Probability of addiction
            else:
                decision_scores = self.best_model.decision_function(X_selected)
                probabilities = 1 / (1 + np.exp(-decision_scores))
            
            # Create histogram
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=probabilities,
                nbinsx=30,
                marker=dict(color='#28a745', opacity=0.7),
                name='Risk Probability Distribution'
            ))
            
            fig.update_layout(
                title="Addiction Risk Probability Distribution",
                xaxis_title="Probability of Addiction",
                yaxis_title="Number of Cases",
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating probability distribution: {str(e)}")
            return go.Figure()
    
    def analyze_model_performance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze detailed model performance."""
        if not self.is_trained or not self.best_model:
            return {}
        
        try:
            X, y = self.prepare_features(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Apply preprocessing pipeline
            X_test_scaled = self.scaler.transform(X_test)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            # Predictions
            y_pred = self.best_model.predict(X_test_selected)
            
            if hasattr(self.best_model, 'predict_proba'):
                y_pred_proba = self.best_model.predict_proba(X_test_selected)
            else:
                y_pred_proba = None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Calculate additional metrics
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = sensitivity
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': cm,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            }
            
        except Exception as e:
            st.error(f"Error analyzing model performance: {str(e)}")
            return {}
    
    def _categorize_risk_level(self, probability: float) -> str:
        """Categorize risk level based on probability."""
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
    
    def _generate_personalized_recommendations(self, user_data: Dict[str, str], 
                                               user_segment: str, risk_probability: float) -> List[Dict[str, str]]:
        """Generate personalized recommendations based on user profile."""
        recommendations = []
        
        # Critical risk recommendations
        if risk_probability > 0.8:
            recommendations.append({
                'priority': 'Critical',
                'type': 'crisis_intervention',
                'title': 'Immediate Professional Help',
                'description': 'Seek immediate professional addiction counseling',
                'action': 'Contact addiction specialist or crisis hotline immediately'
            })
        
        # Feature-specific recommendations
        high_risk_features = {
            'Academic_Performance_Decline': {
                'title': 'Academic Support Services',
                'description': 'Address academic stress and performance issues',
                'action': 'Contact academic advisor and tutoring services'
            },
            'Social_Isolation': {
                'title': 'Social Connection Programs',
                'description': 'Join support groups and social activities',
                'action': 'Attend peer support group meetings'
            },
            'Financial_Issues': {
                'title': 'Financial Counseling',
                'description': 'Get help managing financial stress',
                'action': 'Schedule financial counseling session'
            },
            'Physical_Mental_Health_Problems': {
                'title': 'Integrated Health Care',
                'description': 'Address underlying mental and physical health issues',
                'action': 'Consult healthcare provider for comprehensive evaluation'
            },
            'Withdrawal_Symptoms': {
                'title': 'Medical Supervision',
                'description': 'Medical supervision for withdrawal management',
                'action': 'Seek immediate medical attention for withdrawal symptoms'
            }
        }
        
        for feature, value in user_data.items():
            if value == 'Yes' and feature in high_risk_features:
                rec_info = high_risk_features[feature]
                priority = 'High' if risk_probability > 0.5 else 'Medium'
                recommendations.append({
                    'priority': priority,
                    'type': feature.lower(),
                    'title': rec_info['title'],
                    'description': rec_info['description'],
                    'action': rec_info['action']
                })
        
        # Segment-based recommendations
        segment_recommendations = {
            "Low Risk - Stable": {
                'title': 'Prevention and Wellness',
                'description': 'Continue healthy habits and regular monitoring',
                'action': 'Schedule quarterly wellness check-ups'
            },
            "Moderate Risk - Academic Stress": {
                'title': 'Academic Stress Management',
                'description': 'Focus on academic support and stress reduction',
                'action': 'Enroll in stress management workshop'
            },
            "High Risk - Multiple Factors": {
                'title': 'Comprehensive Treatment Plan',
                'description': 'Multi-faceted approach addressing multiple factors',
                'action': 'Develop comprehensive treatment plan with professionals'
            },
            "Crisis - Immediate Intervention": {
                'title': 'Emergency Intervention',
                'description': 'Immediate professional intervention required',
                'action': 'Contact emergency services or crisis hotline'
            }
        }
        
        if user_segment in segment_recommendations and risk_probability < 0.8:
            rec_info = segment_recommendations[user_segment]
            priority = 'High' if 'Crisis' in user_segment else 'Medium'
            recommendations.append({
                'priority': priority,
                'type': 'segment_based',
                'title': rec_info['title'],
                'description': rec_info['description'],
                'action': rec_info['action']
            })
        
        # Remove duplicates and sort by priority
        unique_recommendations = []
        seen_types = set()
        for rec in recommendations:
            if rec['type'] not in seen_types:
                unique_recommendations.append(rec)
                seen_types.add(rec['type'])
        
        priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
        unique_recommendations.sort(key=lambda x: priority_order.get(x['priority'], 4))
        
        return unique_recommendations[:5]  # Return top 5 recommendations
    
    def get_user_segmentation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user segmentation patterns."""
        if not self.is_trained:
            return {}
        
        try:
            X, y = self.prepare_features(df)
            X_scaled = self.scaler.transform(X)
            X_selected = self.feature_selector.transform(X_scaled)
            
            # Get cluster assignments
            cluster_labels = self.clustering_model.predict(X_selected)
            
            # Analyze segments
            segment_analysis = {}
            for segment_id, segment_name in self.user_segments.items():
                segment_mask = cluster_labels == segment_id
                segment_size = np.sum(segment_mask)
                
                if segment_size > 0:
                    segment_addiction_rate = np.mean(y[segment_mask])
                    segment_analysis[segment_name] = {
                        'size': int(segment_size),
                        'percentage': float(segment_size / len(y) * 100),
                        'addiction_rate': float(segment_addiction_rate * 100),
                        'risk_level': self._categorize_risk_level(segment_addiction_rate)
                    }
            
            return {
                'segments': segment_analysis,
                'total_users': len(y),
                'clustering_model': 'K-Means',
                'num_clusters': len(self.user_segments)
            }
            
        except Exception as e:
            st.error(f"Error analyzing user segmentation: {str(e)}")
            return {}
    
    def plot_user_segmentation(self, df: pd.DataFrame) -> go.Figure:
        """Create user segmentation visualization."""
        if not self.is_trained:
            return go.Figure()
        
        try:
            segmentation_data = self.get_user_segmentation_analysis(df)
            
            if not segmentation_data or 'segments' not in segmentation_data:
                return go.Figure()
            
            segments = segmentation_data['segments']
            
            # Prepare data for plotting
            segment_names = list(segments.keys())
            sizes = [segments[name]['size'] for name in segment_names]
            addiction_rates = [segments[name]['addiction_rate'] for name in segment_names]
            
            # Create subplot with pie chart and bar chart
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'domain'}, {'type': 'bar'}]],
                subplot_titles=('User Segment Distribution', 'Addiction Rate by Segment')
            )
            
            # Pie chart for segment sizes
            colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            fig.add_trace(
                go.Pie(
                    labels=segment_names,
                    values=sizes,
                    marker=dict(colors=colors[:len(segment_names)]),
                    textinfo='label+percent',
                    hole=0.3
                ),
                row=1, col=1
            )
            
            # Bar chart for addiction rates
            fig.add_trace(
                go.Bar(
                    x=segment_names,
                    y=addiction_rates,
                    marker=dict(color=colors[:len(segment_names)]),
                    text=[f'{rate:.1f}%' for rate in addiction_rates],
                    textposition='outside'
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                title="User Segmentation Analysis",
                height=500,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating segmentation plot: {str(e)}")
            return go.Figure()
