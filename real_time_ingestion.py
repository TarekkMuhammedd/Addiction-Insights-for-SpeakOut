import pandas as pd
import numpy as np
import streamlit as st
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import json
import os
from pathlib import Path
from queue import Queue
import threading

class RealTimeDataIngestion:
    """Handles real-time data ingestion for continuous monitoring."""
    
    def __init__(self, data_processor=None):
        self.data_processor = data_processor
        self.monitoring_active = False
        self.monitoring_thread = None
        self.data_queue = []
        self.alert_queue = Queue()
        self.data_lock = threading.Lock()
        self.alert_thresholds = {
            'high_risk_rate': 30.0,  # Alert if >30% high risk
            'new_cases_spike': 50,   # Alert if >50 new cases per hour
            'crisis_indicators': 5   # Alert if >5 crisis indicators
        }
        self.callbacks = []
        self.interval_seconds = 30
        
    def add_monitoring_callback(self, callback: Callable):
        """Add callback function to be called when new data arrives."""
        self.callbacks.append(callback)
    
    def simulate_real_time_data(self) -> Dict:
        """Simulate incoming real-time addiction data."""
        # Generate realistic addiction data similar to training set
        risk_factors = [
            'Experimentation', 'Academic_Performance_Decline', 'Social_Isolation',
            'Financial_Issues', 'Physical_Mental_Health_Problems', 'Legal_Consequences',
            'Relationship_Strain', 'Risk_Taking_Behavior', 'Withdrawal_Symptoms',
            'Denial_and_Resistance_to_Treatment'
        ]
        
        # Create realistic probabilities based on existing data patterns
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'user_id': f"user_{np.random.randint(1000, 9999)}",
        }
        
        # Generate correlated risk factors (some combinations are more likely)
        high_risk_scenario = np.random.random() < 0.25  # 25% chance of high-risk profile
        
        for factor in risk_factors:
            if high_risk_scenario:
                # High-risk individuals more likely to have multiple factors
                probability = 0.7 if factor in ['Social_Isolation', 'Academic_Performance_Decline', 
                                               'Withdrawal_Symptoms'] else 0.5
            else:
                # Low-risk individuals less likely to have factors
                probability = 0.3 if factor in ['Experimentation', 'Academic_Performance_Decline'] else 0.15
            
            new_record[factor] = 'Yes' if np.random.random() < probability else 'No'
        
        # Determine addiction class based on risk factors
        risk_count = sum(1 for key, value in new_record.items() 
                        if key in risk_factors and value == 'Yes')
        new_record['Addiction_Class'] = 'Yes' if risk_count >= 4 else 'No'
        
        return new_record
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start real-time data monitoring."""
        if self.monitoring_active:
            return False
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        return True
    
    def stop_monitoring(self):
        """Stop real-time data monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop that runs in background thread."""
        self.interval_seconds = interval_seconds
        while self.monitoring_active:
            try:
                # Generate new data
                new_data = self.simulate_real_time_data()
                
                # Thread-safe data queue update
                with self.data_lock:
                    self.data_queue.append(new_data)
                    
                    # Keep only recent data (last 1000 records)
                    if len(self.data_queue) > 1000:
                        self.data_queue = self.data_queue[-1000:]
                
                # Check for alerts (thread-safe)
                self._check_alerts_thread_safe()
                
                # Notify callbacks
                for callback in self.callbacks:
                    try:
                        callback(new_data)
                    except Exception as e:
                        # Don't use st.error in background thread
                        print(f"Callback error: {str(e)}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                # Don't use st.error in background thread
                print(f"Monitoring error: {str(e)}")
                time.sleep(interval_seconds)
    
    def _check_alerts_thread_safe(self):
        """Thread-safe alert checking."""
        with self.data_lock:
            if len(self.data_queue) < 10:  # Need minimum data for alerts
                return
            
            # Calculate time-based window (last hour)
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_data = []
            
            for record in self.data_queue:
                try:
                    record_time = datetime.fromisoformat(record['timestamp'])
                    if record_time >= cutoff_time:
                        recent_data.append(record)
                except:
                    # If timestamp parsing fails, include recent records by count
                    continue
            
            # Fallback to count-based if no timestamp filtering worked
            if not recent_data and len(self.data_queue) >= 10:
                # Calculate based on interval - last hour worth of records
                records_per_hour = max(10, int(3600 / self.interval_seconds))
                recent_data = self.data_queue[-records_per_hour:]
            
            if not recent_data:
                return
            
            df = pd.DataFrame(recent_data)
            alerts = []
            
            # Check high-risk rate
            if not df.empty:
                risk_factors = [
                    'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
                    'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
                    'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
                ]
                
                available_factors = [f for f in risk_factors if f in df.columns]
                if available_factors:
                    df['risk_count'] = df[available_factors].apply(
                        lambda row: sum(1 for val in row if val == 'Yes'), axis=1
                    )
                    high_risk_rate = (df['risk_count'] >= 5).mean() * 100
                    
                    if high_risk_rate > self.alert_thresholds['high_risk_rate']:
                        alerts.append({
                            'type': 'high_risk_rate',
                            'message': f"High risk rate alert: {high_risk_rate:.1f}% of recent cases are high-risk",
                            'severity': 'warning',
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Check for crisis indicators
                crisis_factors = ['Withdrawal_Symptoms', 'Legal_Consequences', 'Physical_Mental_Health_Problems']
                available_crisis = [f for f in crisis_factors if f in df.columns]
                
                if available_crisis:
                    crisis_count = df[available_crisis].apply(
                        lambda row: sum(1 for val in row if val == 'Yes'), axis=1
                    ).sum()
                    
                    if crisis_count > self.alert_thresholds['crisis_indicators']:
                        alerts.append({
                            'type': 'crisis_indicators',
                            'message': f"Crisis indicators alert: {crisis_count} crisis indicators detected in recent data",
                            'severity': 'critical',
                            'timestamp': datetime.now().isoformat()
                        })
                
                # Check new cases spike
                new_cases_count = len(recent_data)
                if new_cases_count > self.alert_thresholds['new_cases_spike']:
                    alerts.append({
                        'type': 'new_cases_spike',
                        'message': f"New cases spike alert: {new_cases_count} new cases in the last hour",
                        'severity': 'warning', 
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Add alerts to thread-safe queue
                for alert in alerts:
                    self.alert_queue.put(alert)
    
    def get_recent_data(self, minutes: int = 60) -> pd.DataFrame:
        """Get recent data from the specified time window."""
        with self.data_lock:
            if not self.data_queue:
                return pd.DataFrame()
            
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            recent_records = []
            
            for record in self.data_queue:
                try:
                    record_time = datetime.fromisoformat(record['timestamp'])
                    if record_time >= cutoff_time:
                        recent_records.append(record)
                except:
                    # If timestamp parsing fails, include the record
                    recent_records.append(record)
            
            return pd.DataFrame(recent_records)
    
    def get_monitoring_stats(self) -> Dict:
        """Get current monitoring statistics."""
        with self.data_lock:
            if not self.data_queue:
                return {
                    'total_records': 0,
                    'recent_hour_count': 0,
                    'high_risk_rate': 0,
                    'last_update': None,
                    'monitoring_active': self.monitoring_active
                }
            
            # Get copy of data for processing outside lock
            data_copy = self.data_queue.copy()
            
        recent_df = self.get_recent_data(60)  # Last hour
        
        stats = {
            'total_records': len(data_copy),
            'recent_hour_count': len(recent_df),
            'monitoring_active': self.monitoring_active,
            'last_update': data_copy[-1]['timestamp'] if data_copy else None
        }
        
        if not recent_df.empty:
            risk_factors = [
                'Academic_Performance_Decline', 'Social_Isolation', 'Financial_Issues',
                'Physical_Mental_Health_Problems', 'Legal_Consequences', 'Relationship_Strain',
                'Risk_Taking_Behavior', 'Withdrawal_Symptoms'
            ]
            available_factors = [f for f in risk_factors if f in recent_df.columns]
            
            if available_factors:
                recent_df['risk_count'] = recent_df[available_factors].apply(
                    lambda row: sum(1 for val in row if val == 'Yes'), axis=1
                )
                stats['high_risk_rate'] = (recent_df['risk_count'] >= 5).mean() * 100
                stats['average_risk_factors'] = recent_df['risk_count'].mean()
                stats['addiction_rate'] = (recent_df['Addiction_Class'] == 'Yes').mean() * 100
        
        return stats
    
    def export_monitoring_data(self, filename: str = None) -> str:
        """Export monitoring data to CSV."""
        if not self.data_queue:
            return None
        
        if filename is None:
            filename = f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(self.data_queue)
        df.to_csv(filename, index=False)
        return filename
    
    def configure_alerts(self, thresholds: Dict):
        """Configure alert thresholds."""
        self.alert_thresholds.update(thresholds)
    
    def get_alerts(self) -> List[Dict]:
        """Get current alerts from thread-safe queue."""
        alerts = []
        
        # Drain the alert queue
        while not self.alert_queue.empty():
            try:
                alerts.append(self.alert_queue.get_nowait())
            except:
                break
        
        return alerts
    
    def clear_alerts(self):
        """Clear all current alerts."""
        # Clear the alert queue
        while not self.alert_queue.empty():
            try:
                self.alert_queue.get_nowait()
            except:
                break

class RealTimeVisualization:
    """Handles real-time visualization components."""
    
    def __init__(self, color_palette=None):
        self.color_palette = color_palette or {
            'primary': '#28a745',
            'danger': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
    
    def create_real_time_metrics(self, monitoring_stats: Dict):
        """Create real-time metrics display."""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records", 
                f"{monitoring_stats.get('total_records', 0):,}",
                help="Total records collected since monitoring started"
            )
        
        with col2:
            recent_count = monitoring_stats.get('recent_hour_count', 0)
            st.metric(
                "Last Hour", 
                recent_count,
                delta=f"+{recent_count}" if recent_count > 0 else None,
                help="New records in the last hour"
            )
        
        with col3:
            high_risk_rate = monitoring_stats.get('high_risk_rate', 0)
            st.metric(
                "High Risk Rate", 
                f"{high_risk_rate:.1f}%",
                delta=f"{high_risk_rate - 25:.1f}%" if high_risk_rate > 25 else None,
                delta_color="inverse",
                help="Percentage of high-risk cases in recent data"
            )
        
        with col4:
            status = "🟢 Active" if monitoring_stats.get('monitoring_active', False) else "🔴 Stopped"
            st.metric(
                "Monitoring Status", 
                status,
                help="Current monitoring status"
            )
    
    def create_live_chart(self, recent_data: pd.DataFrame):
        """Create live updating chart."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if recent_data.empty:
            st.info("No recent data available for live chart")
            return
        
        # Create time-based aggregation
        recent_data['timestamp'] = pd.to_datetime(recent_data['timestamp'])
        
        # Aggregate by 5-minute intervals
        recent_data['time_bin'] = recent_data['timestamp'].dt.floor('5min')
        
        time_agg = recent_data.groupby('time_bin').agg({
            'Addiction_Class': lambda x: (x == 'Yes').mean() * 100,
            'user_id': 'count'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Addiction Rate Over Time', 'New Cases Over Time'),
            vertical_spacing=0.1
        )
        
        # Addiction rate
        fig.add_trace(
            go.Scatter(
                x=time_agg['time_bin'],
                y=time_agg['Addiction_Class'],
                mode='lines+markers',
                name='Addiction Rate (%)',
                line=dict(color=self.color_palette['danger'], width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # New cases
        fig.add_trace(
            go.Bar(
                x=time_agg['time_bin'],
                y=time_agg['user_id'],
                name='New Cases',
                marker_color=self.color_palette['primary']
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Real-Time Monitoring Dashboard",
            height=500,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Addiction Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Number of Cases", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_alerts(self, alerts: List[Dict]):
        """Display active alerts."""
        if not alerts:
            st.success("No active alerts")
            return
        
        st.markdown("### 🚨 Active Alerts")
        
        # Group alerts by severity
        critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
        warning_alerts = [a for a in alerts if a.get('severity') == 'warning']
        
        if critical_alerts:
            st.markdown("#### Critical Alerts")
            for alert in critical_alerts:
                st.error(f"🚨 {alert['message']}")
        
        if warning_alerts:
            st.markdown("#### Warnings")
            for alert in warning_alerts:
                st.warning(f"⚠️ {alert['message']}")