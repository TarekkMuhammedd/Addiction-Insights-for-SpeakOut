import pandas as pd
import numpy as np
from typing import Dict, List, Any
import streamlit as st

class CommunityInsights:
    """Generates insights and recommendations for community website development."""
    
    def __init__(self):
        self.user_segments = {
            'high_risk': {
                'threshold': 5,
                'characteristics': ['Multiple risk factors', 'High addiction probability', 'Immediate intervention needed'],
                'needs': ['Crisis support', 'Professional help', 'Immediate resources']
            },
            'moderate_risk': {
                'threshold': 3,
                'characteristics': ['Some risk factors', 'Moderate addiction probability', 'Prevention focus'],
                'needs': ['Educational content', 'Peer support', 'Progress tracking']
            },
            'low_risk': {
                'threshold': 1,
                'characteristics': ['Few risk factors', 'Low addiction probability', 'Good candidates for mentoring'],
                'needs': ['Prevention information', 'Community engagement', 'Success stories']
            }
        }
    
    def analyze_user_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns from the dataset."""
        
        # Calculate risk scores for each user
        risk_factors = [
            'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
            'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
            'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
        ]
        
        available_factors = [factor for factor in risk_factors if factor in df.columns]
        
        # Calculate total risk factors for each user
        if 'Total_Risk_Factors' not in df.columns:
            df['Total_Risk_Factors'] = df[available_factors].apply(
                lambda row: sum(1 for val in row if val == 'Yes'), axis=1
            )
        
        # Categorize users into risk segments
        high_risk_users = df[df['Total_Risk_Factors'] >= self.user_segments['high_risk']['threshold']]
        moderate_risk_users = df[
            (df['Total_Risk_Factors'] >= self.user_segments['moderate_risk']['threshold']) & 
            (df['Total_Risk_Factors'] < self.user_segments['high_risk']['threshold'])
        ]
        low_risk_users = df[df['Total_Risk_Factors'] < self.user_segments['moderate_risk']['threshold']]
        
        total_users = len(df)
        
        # Calculate percentages
        high_risk_percentage = (len(high_risk_users) / total_users) * 100
        moderate_risk_percentage = (len(moderate_risk_users) / total_users) * 100
        low_risk_percentage = (len(low_risk_users) / total_users) * 100
        
        # Analyze behavioral patterns
        behavioral_patterns = self._analyze_behavioral_patterns(df, available_factors)
        
        # Identify intervention points
        intervention_points = self._identify_intervention_points(df, available_factors)
        
        return {
            'high_risk_percentage': high_risk_percentage,
            'moderate_risk_percentage': moderate_risk_percentage,
            'low_risk_percentage': low_risk_percentage,
            'high_risk_users': len(high_risk_users),
            'moderate_risk_users': len(moderate_risk_users),
            'low_risk_users': len(low_risk_users),
            'behavioral_patterns': behavioral_patterns,
            'intervention_points': intervention_points,
            'user_segments': {
                'high_risk': self._get_segment_characteristics(high_risk_users, available_factors),
                'moderate_risk': self._get_segment_characteristics(moderate_risk_users, available_factors),
                'low_risk': self._get_segment_characteristics(low_risk_users, available_factors)
            }
        }
    
    def _analyze_behavioral_patterns(self, df: pd.DataFrame, risk_factors: List[str]) -> Dict[str, Any]:
        """Analyze behavioral patterns in the data."""
        patterns = {}
        
        # Most common risk factor combinations
        risk_combinations = []
        for _, row in df.iterrows():
            active_risks = [factor for factor in risk_factors if row[factor] == 'Yes']
            if len(active_risks) > 1:
                risk_combinations.append(tuple(sorted(active_risks)))
        
        from collections import Counter
        common_combinations = Counter(risk_combinations).most_common(5)
        
        patterns['common_risk_combinations'] = [
            {
                'combination': list(combo),
                'frequency': freq,
                'percentage': (freq / len(df)) * 100
            }
            for combo, freq in common_combinations
        ]
        
        # Progression indicators
        patterns['progression_indicators'] = self._identify_progression_indicators(df, risk_factors)
        
        # Recovery readiness indicators
        patterns['recovery_readiness'] = self._analyze_recovery_readiness(df)
        
        return patterns
    
    def _identify_intervention_points(self, df: pd.DataFrame, risk_factors: List[str]) -> Dict[str, Any]:
        """Identify key intervention points."""
        intervention_points = {}
        
        # Early warning signs (factors that often appear before addiction)
        early_warning = []
        if 'Addiction_Class' in df.columns:
            addicted_users = df[df['Addiction_Class'] == 'Yes']
            for factor in risk_factors:
                prevalence_in_addicted = (addicted_users[factor] == 'Yes').mean()
                if prevalence_in_addicted > 0.7:  # Appears in >70% of addicted users
                    early_warning.append({
                        'factor': factor.replace('_', ' '),
                        'prevalence': prevalence_in_addicted * 100
                    })
        
        intervention_points['early_warning_signs'] = sorted(early_warning, key=lambda x: x['prevalence'], reverse=True)
        
        # Critical intervention thresholds
        intervention_points['critical_thresholds'] = {
            'risk_factor_count': 3,  # Users with 3+ risk factors need immediate attention
            'high_risk_factors': ['Withdrawal_Symptoms', 'Legal_Consequences'],  # These indicate advanced stages
            'recovery_indicators': ['Denial_and_Resistance_to_Treatment']  # Key for treatment success
        }
        
        return intervention_points
    
    def _get_segment_characteristics(self, segment_df: pd.DataFrame, risk_factors: List[str]) -> Dict[str, Any]:
        """Get characteristics of a user segment."""
        if len(segment_df) == 0:
            return {'size': 0, 'characteristics': {}}
        
        characteristics = {}
        
        # Most common risk factors in this segment
        for factor in risk_factors:
            prevalence = (segment_df[factor] == 'Yes').mean() * 100
            characteristics[factor.replace('_', ' ')] = prevalence
        
        # Average risk factor count
        if 'Total_Risk_Factors' in segment_df.columns:
            avg_risk_factors = segment_df['Total_Risk_Factors'].mean()
        else:
            avg_risk_factors = segment_df[risk_factors].apply(lambda x: x == 'Yes').sum(axis=1).mean()
        
        # Addiction rate in this segment
        addiction_rate = 0
        if 'Addiction_Class' in segment_df.columns:
            addiction_rate = (segment_df['Addiction_Class'] == 'Yes').mean() * 100
        
        return {
            'size': len(segment_df),
            'avg_risk_factors': avg_risk_factors,
            'addiction_rate': addiction_rate,
            'top_risk_factors': dict(sorted(characteristics.items(), key=lambda x: x[1], reverse=True)[:3])
        }
    
    def _identify_progression_indicators(self, df: pd.DataFrame, risk_factors: List[str]) -> Dict[str, Any]:
        """Identify addiction progression indicators."""
        progression = {}
        
        # Early stage indicators (lower severity)
        early_stage = ['Experimentation', 'Academic_Performance_Decline']
        middle_stage = ['Social_Isolation', 'Financial_Issues', 'Relationship_Strain']
        late_stage = ['Withdrawal_Symptoms', 'Legal_Consequences', 'Physical_Mental_Health_Problems']
        
        for stage, factors in [('early', early_stage), ('middle', middle_stage), ('late', late_stage)]:
            available_factors = [f for f in factors if f in df.columns]
            if available_factors:
                stage_prevalence = df[available_factors].apply(lambda x: x == 'Yes').mean().mean() * 100
                progression[f'{stage}_stage_prevalence'] = stage_prevalence
        
        return progression
    
    def _analyze_recovery_readiness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze recovery readiness indicators."""
        readiness = {}
        
        # Treatment resistance as negative indicator
        if 'Denial_and_Resistance_to_Treatment' in df.columns:
            resistance_rate = (df['Denial_and_Resistance_to_Treatment'] == 'Yes').mean() * 100
            readiness['treatment_resistance_rate'] = resistance_rate
            readiness['recovery_readiness_rate'] = 100 - resistance_rate
        
        # Social support indicators (inverse of social isolation)
        if 'Social_Isolation' in df.columns:
            social_support_rate = (df['Social_Isolation'] == 'No').mean() * 100
            readiness['social_support_availability'] = social_support_rate
        
        return readiness
    
    def generate_website_recommendations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Generate website feature recommendations based on user analysis."""
        user_patterns = self.analyze_user_patterns(df)
        
        recommendations = {
            'Crisis Intervention Features': [
                'Emergency hotline integration with one-click calling',
                'Crisis chat support with trained counselors',
                'Emergency contact notification system',
                'Immediate resource locator (nearby treatment centers)',
                'Crisis assessment questionnaire with risk scoring',
                'Safety planning tools and templates'
            ],
            
            'Educational Content': [
                'Interactive addiction education modules',
                'Risk factor awareness quizzes',
                'Progression timeline educational content',
                'Family education resources',
                'Myth-busting content about addiction',
                'Video testimonials from recovery success stories'
            ],
            
            'Community Support': [
                'Peer support group matching system',
                'Anonymous forum with moderation',
                'Mentorship program connecting low-risk with high-risk users',
                'Virtual support group meeting scheduler',
                'Success story sharing platform',
                'Community challenges and goals'
            ],
            
            'Progress Tracking': [
                'Daily check-in and mood tracking',
                'Risk factor monitoring dashboard',
                'Recovery milestone tracking',
                'Goal setting and achievement system',
                'Progress visualization charts',
                'Personal recovery journal'
            ],
            
            'Personalization Features': [
                'Risk-based content personalization',
                'Customizable dashboard based on user segment',
                'Personalized resource recommendations',
                'Adaptive intervention triggers',
                'Custom notification preferences',
                'Tailored recovery path suggestions'
            ],
            
            'Professional Integration': [
                'Therapist directory with specialization filtering',
                'Appointment scheduling integration',
                'Progress report generation for healthcare providers',
                'Treatment plan tracking',
                'Medication reminder system',
                'Professional consultation booking'
            ]
        }
        
        # Prioritize recommendations based on user distribution
        high_risk_percentage = user_patterns['high_risk_percentage']
        if high_risk_percentage > 30:
            recommendations['Priority Features'] = [
                'Crisis intervention must be primary focus',
                'Immediate professional help connection',
                'Advanced risk monitoring systems'
            ]
        elif high_risk_percentage > 15:
            recommendations['Priority Features'] = [
                'Balanced approach between crisis and prevention',
                'Strong peer support systems',
                'Educational content with intervention triggers'
            ]
        else:
            recommendations['Priority Features'] = [
                'Prevention-focused content',
                'Community building features',
                'Educational and awareness tools'
            ]
        
        return recommendations
    
    def get_engagement_strategies(self, user_patterns: Dict[str, Any]) -> Dict[str, List[str]]:
        """Get user engagement strategies based on patterns."""
        
        strategies = {
            'high_engagement': [
                'Gamification with meaningful rewards (not trivializing addiction)',
                'Personal progress visualization',
                'Peer recognition and support systems',
                'Regular check-ins with personalized feedback',
                'Success milestone celebrations',
                'Interactive educational content'
            ],
            
            'retention': [
                'Personalized content delivery based on risk level',
                'Regular but non-intrusive notification system',
                'Community building through shared experiences',
                'Professional support integration',
                'Family involvement features',
                'Long-term goal tracking and achievement'
            ],
            
            'user_onboarding': [
                'Risk assessment to personalize experience',
                'Clear explanation of platform benefits',
                'Immediate value demonstration',
                'Safe space assurance and privacy explanation',
                'Easy first steps with quick wins',
                'Connection to relevant resources immediately'
            ],
            
            'crisis_engagement': [
                'Immediate response protocols',
                'Always-available crisis resources',
                'Simplified interface during crisis moments',
                'Quick access to human support',
                'Emergency contact integration',
                'Follow-up care coordination'
            ]
        }
        
        # Customize strategies based on user distribution
        high_risk_percentage = user_patterns['high_risk_percentage']
        
        if high_risk_percentage > 25:
            strategies['primary_focus'] = [
                'Crisis intervention and immediate support',
                'Professional help connection',
                'Safety and harm reduction'
            ]
        else:
            strategies['primary_focus'] = [
                'Prevention and education',
                'Community building',
                'Early intervention'
            ]
        
        return strategies
    
    def get_success_metrics(self) -> Dict[str, List[str]]:
        """Define success metrics for the community website."""
        return {
            'User Engagement Metrics': [
                'Daily active users (DAU) and monthly active users (MAU)',
                'Session duration and frequency',
                'Feature utilization rates',
                'Community participation levels',
                'Content consumption patterns',
                'User retention rates'
            ],
            
            'Recovery Outcome Metrics': [
                'Self-reported recovery progress',
                'Risk factor reduction over time',
                'Treatment engagement rates',
                'Crisis intervention success rates',
                'Long-term abstinence rates',
                'Quality of life improvement scores'
            ],
            
            'Community Health Metrics': [
                'Peer support interaction frequency',
                'Support group participation',
                'Mentor-mentee relationship success',
                'Community-generated content quality',
                'User safety and report rates',
                'Professional referral success rates'
            ],
            
            'Platform Effectiveness Metrics': [
                'Crisis response time',
                'Resource accessibility and usage',
                'Educational content completion rates',
                'Goal achievement rates',
                'User satisfaction scores',
                'Platform reliability and uptime'
            ],
            
            'Clinical Integration Metrics': [
                'Healthcare provider adoption',
                'Treatment plan adherence',
                'Professional consultation rates',
                'Data sharing with healthcare providers',
                'Clinical outcome improvements',
                'Integrated care coordination success'
            ]
        }
    
    def generate_user_personas(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate detailed user personas based on data analysis."""
        user_patterns = self.analyze_user_patterns(df)
        
        personas = {
            'High-Risk Individual': {
                'description': 'Individual with multiple risk factors requiring immediate intervention',
                'characteristics': [
                    f'Has {user_patterns["user_segments"]["high_risk"]["avg_risk_factors"]:.1f} risk factors on average',
                    f'{user_patterns["user_segments"]["high_risk"]["addiction_rate"]:.1f}% addiction rate',
                    'Often experiences withdrawal symptoms and legal consequences',
                    'May resist treatment initially'
                ],
                'needs': [
                    'Immediate crisis support',
                    'Professional intervention',
                    'Safety planning',
                    'Harm reduction resources',
                    'Family support coordination'
                ],
                'website_features': [
                    'Emergency contact integration',
                    'Crisis chat support',
                    'Immediate resource locator',
                    'Professional referral system',
                    'Safety planning tools'
                ]
            },
            
            'Moderate-Risk Individual': {
                'description': 'Individual with some risk factors, good candidate for prevention programs',
                'characteristics': [
                    f'Has {user_patterns["user_segments"]["moderate_risk"]["avg_risk_factors"]:.1f} risk factors on average',
                    f'{user_patterns["user_segments"]["moderate_risk"]["addiction_rate"]:.1f}% addiction rate',
                    'Often experiences academic decline and social isolation',
                    'Responsive to educational interventions'
                ],
                'needs': [
                    'Educational content',
                    'Peer support',
                    'Progress tracking',
                    'Skill building resources',
                    'Early intervention programs'
                ],
                'website_features': [
                    'Educational modules',
                    'Peer support groups',
                    'Progress tracking dashboard',
                    'Goal setting tools',
                    'Resource library'
                ]
            },
            
            'Low-Risk Individual': {
                'description': 'Individual with few risk factors, potential peer mentor',
                'characteristics': [
                    f'Has {user_patterns["user_segments"]["low_risk"]["avg_risk_factors"]:.1f} risk factors on average',
                    f'{user_patterns["user_segments"]["low_risk"]["addiction_rate"]:.1f}% addiction rate',
                    'Generally stable social and academic functioning',
                    'Good candidates for peer mentoring roles'
                ],
                'needs': [
                    'Prevention information',
                    'Community engagement opportunities',
                    'Mentoring training',
                    'Continued support',
                    'Success story sharing'
                ],
                'website_features': [
                    'Prevention content',
                    'Mentor training modules',
                    'Community forums',
                    'Success story platform',
                    'Volunteer opportunities'
                ]
            }
        }
        
        return personas
    
    def get_implementation_roadmap(self, user_patterns: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate implementation roadmap for the community website."""
        high_risk_percentage = user_patterns['high_risk_percentage']
        
        roadmap = {
            'Phase 1: Foundation (Months 1-3)': [
                'User registration and authentication system',
                'Basic user profiling and risk assessment',
                'Essential crisis intervention features',
                'Emergency contact integration',
                'Basic educational content library',
                'Simple progress tracking'
            ],
            
            'Phase 2: Community Building (Months 4-6)': [
                'Peer support group functionality',
                'Forum and discussion boards',
                'Mentorship program framework',
                'Advanced educational modules',
                'Professional directory integration',
                'Enhanced progress tracking and analytics'
            ],
            
            'Phase 3: Advanced Features (Months 7-9)': [
                'Personalization engine',
                'Advanced crisis intervention tools',
                'Family involvement features',
                'Healthcare provider integration',
                'Mobile application development',
                'Advanced analytics and reporting'
            ],
            
            'Phase 4: Optimization (Months 10-12)': [
                'AI-powered personalization',
                'Predictive risk modeling',
                'Advanced community features',
                'Integration with external services',
                'Comprehensive outcome tracking',
                'Platform scaling and optimization'
            ]
        }
        
        # Adjust roadmap based on user distribution
        if high_risk_percentage > 30:
            roadmap['Priority Adjustments'] = [
                'Accelerate crisis intervention feature development',
                'Prioritize professional integration',
                'Enhanced safety features in Phase 1',
                'Immediate access to human support'
            ]
        
        return roadmap
