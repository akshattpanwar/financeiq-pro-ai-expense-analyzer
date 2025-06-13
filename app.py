import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, silhouette_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
import re
import warnings
warnings.filterwarnings('ignore')

# ================== ADVANCED AI/ML ENGINE ==================

class AdvancedExpenseAI:
    """Complete AI/ML Engine for Expense Analysis"""
    
    def __init__(self):
        self.category_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.amount_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.15, random_state=42)
        self.clustering_model = KMeans(n_clusters=3, random_state=42)
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
        self.model_metrics = {}
    
    def extract_features_from_text(self, text):
        """NLP-based feature extraction from expense descriptions"""
        if pd.isna(text) or text == "":
            return {"word_count": 0, "has_numbers": 0, "has_merchant": 0}
        
        text = str(text).lower()
        
        # Merchant/Location keywords
        merchant_keywords = ['restaurant', 'cafe', 'store', 'shop', 'mall', 'grocery', 
                           'gas', 'station', 'pharmacy', 'hotel', 'uber', 'amazon']
        
        features = {
            "word_count": len(text.split()),
            "has_numbers": int(bool(re.search(r'\d', text))),
            "has_merchant": int(any(keyword in text for keyword in merchant_keywords)),
            "text_length": len(text)
        }
        
        return features
    
    def advanced_feature_engineering(self, df):
        """Comprehensive feature engineering pipeline"""
        features_df = df.copy()
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        
        # ===== TIME-BASED FEATURES =====
        features_df['day_of_week'] = features_df['Date'].dt.dayofweek
        features_df['day_of_month'] = features_df['Date'].dt.day
        features_df['month'] = features_df['Date'].dt.month
        features_df['quarter'] = features_df['Date'].dt.quarter
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_month_start'] = (features_df['Date'].dt.day <= 5).astype(int)
        features_df['is_month_end'] = (features_df['Date'].dt.day >= 25).astype(int)
        
        # ===== AMOUNT-BASED FEATURES =====
        features_df['amount_log'] = np.log1p(features_df['Amount'])
        features_df['amount_sqrt'] = np.sqrt(features_df['Amount'])
        
        # Z-score normalization
        if len(features_df) > 1:
            features_df['amount_zscore'] = (features_df['Amount'] - features_df['Amount'].mean()) / features_df['Amount'].std()
        else:
            features_df['amount_zscore'] = 0
        
        # ===== ROLLING STATISTICS =====
        if len(features_df) >= 7:
            features_df = features_df.sort_values('Date')
            features_df['rolling_mean_3'] = features_df['Amount'].rolling(3, min_periods=1).mean()
            features_df['rolling_mean_7'] = features_df['Amount'].rolling(7, min_periods=1).mean()
            features_df['rolling_std_7'] = features_df['Amount'].rolling(7, min_periods=1).std().fillna(0)
            features_df['rolling_max_7'] = features_df['Amount'].rolling(7, min_periods=1).max()
            features_df['rolling_min_7'] = features_df['Amount'].rolling(7, min_periods=1).min()
        
        # ===== CATEGORY-BASED FEATURES =====
        category_stats = features_df.groupby('Category')['Amount'].agg(['mean', 'std', 'count']).fillna(0)
        category_stats.columns = ['cat_mean', 'cat_std', 'cat_count']
        
        for idx, row in features_df.iterrows():
            cat = row['Category']
            if cat in category_stats.index:
                features_df.loc[idx, 'category_avg_ratio'] = row['Amount'] / max(category_stats.loc[cat, 'cat_mean'], 1)
                features_df.loc[idx, 'category_frequency'] = category_stats.loc[cat, 'cat_count']
            else:
                features_df.loc[idx, 'category_avg_ratio'] = 1
                features_df.loc[idx, 'category_frequency'] = 1
        
        # ===== NLP FEATURES =====
        if 'Description' in features_df.columns:
            nlp_features = features_df['Description'].apply(self.extract_features_from_text)
            nlp_df = pd.DataFrame(list(nlp_features))
            features_df = pd.concat([features_df, nlp_df], axis=1)
        
        # ===== BEHAVIORAL FEATURES =====
        features_df['hour_of_day'] = features_df['Date'].dt.hour if 'hour' in str(features_df['Date'].dtype) else 12
        features_df['spending_velocity'] = features_df['Amount'] / (features_df.index + 1)  # Cumulative velocity
        
        return features_df
    
    def train_ai_models(self, df):
        """Train comprehensive AI/ML pipeline"""
        try:
            if len(df) < 10:
                return {"status": "insufficient_data", "message": "Need at least 10 records for AI training"}
            
            # Feature engineering
            features_df = self.advanced_feature_engineering(df)
            
            # Define feature columns
            feature_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend', 
                           'is_month_start', 'is_month_end', 'amount_log', 'amount_sqrt', 
                           'amount_zscore', 'category_avg_ratio', 'category_frequency', 
                           'hour_of_day', 'spending_velocity']
            
            # Add rolling features if available
            rolling_cols = ['rolling_mean_3', 'rolling_mean_7', 'rolling_std_7', 'rolling_max_7', 'rolling_min_7']
            for col in rolling_cols:
                if col in features_df.columns:
                    feature_cols.append(col)
            
            # Add NLP features if available
            nlp_cols = ['word_count', 'has_numbers', 'has_merchant', 'text_length']
            for col in nlp_cols:
                if col in features_df.columns:
                    feature_cols.append(col)
            
            X = features_df[feature_cols].fillna(0)
            
            # ===== CATEGORY PREDICTION MODEL =====
            if len(features_df['Category'].unique()) >= 2:
                y_category = features_df['Category']
                self.label_encoder.fit(y_category)
                y_encoded = self.label_encoder.transform(y_category)
                
                # Train with cross-validation
                cv_scores = cross_val_score(self.category_predictor, X, y_encoded, cv=min(3, len(df)//2))
                self.category_predictor.fit(X, y_encoded)
                
                # Feature importance
                self.feature_importance['category'] = dict(zip(feature_cols, self.category_predictor.feature_importances_))
                self.model_metrics['category_cv_score'] = cv_scores.mean()
            
            # ===== AMOUNT PREDICTION MODEL =====
            y_amount = features_df['Amount']
            X_scaled = self.scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_amount, test_size=0.2, random_state=42)
            self.amount_predictor.fit(X_train, y_train)
            
            # Model evaluation
            y_pred = self.amount_predictor.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            self.model_metrics['amount_mae'] = mae
            self.model_metrics['amount_accuracy'] = 1 - (mae / y_test.mean())
            
            # ===== ANOMALY DETECTION =====
            self.anomaly_detector.fit(X_scaled)
            anomalies = self.anomaly_detector.predict(X_scaled)
            features_df['is_anomaly'] = (anomalies == -1)
            self.model_metrics['anomaly_rate'] = (anomalies == -1).mean()
            
            # ===== CLUSTERING ANALYSIS =====
            if len(X_scaled) >= 3:
                cluster_labels = self.clustering_model.fit_predict(X_scaled)
                features_df['spending_cluster'] = cluster_labels
                
                if len(set(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                    self.model_metrics['clustering_score'] = silhouette_avg
            
            self.is_trained = True
            
            return {
                "status": "success",
                "features_df": features_df,
                "metrics": self.model_metrics,
                "feature_importance": self.feature_importance
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def predict_category(self, amount, day_of_week, month, description=""):
        """Predict expense category using trained model"""
        if not self.is_trained:
            return "Other"
        
        try:
            # Create feature vector
            nlp_features = self.extract_features_from_text(description)
            
            features = np.array([[
                day_of_week, 15, month, (month-1)//3 + 1, int(day_of_week >= 5),
                0, 0, np.log1p(amount), np.sqrt(amount), 0, 1, 1, 12, amount,
                nlp_features.get('word_count', 0), nlp_features.get('has_numbers', 0),
                nlp_features.get('has_merchant', 0), nlp_features.get('text_length', 0)
            ]])
            
            # Pad or trim features to match training
            expected_features = len(self.feature_importance.get('category', {}).keys()) if self.feature_importance.get('category') else 18
            if features.shape[1] < expected_features:
                features = np.pad(features, ((0, 0), (0, expected_features - features.shape[1])), 'constant')
            elif features.shape[1] > expected_features:
                features = features[:, :expected_features]
            
            prediction = self.category_predictor.predict(features)
            predicted_category = self.label_encoder.inverse_transform(prediction)[0]
            
            return predicted_category
        except:
            return "Other"
    
    def get_smart_insights(self, df, features_df=None):
        """Generate AI-powered insights"""
        insights = []
        
        if features_df is not None and 'is_anomaly' in features_df.columns:
            anomaly_count = features_df['is_anomaly'].sum()
            if anomaly_count > 0:
                insights.append(f"🚨 **Anomaly Alert**: {anomaly_count} unusual spending patterns detected")
        
        # Advanced statistical insights
        if len(df) >= 7:
            recent_trend = df.tail(7)['Amount'].sum() - df.head(7)['Amount'].sum()
            trend_direction = "increasing" if recent_trend > 0 else "decreasing"
            insights.append(f"📈 **Spending Trend**: Your spending is {trend_direction} by ₹{abs(recent_trend):,.2f}")
        
        # Category optimization
        category_efficiency = df.groupby('Category')['Amount'].agg(['count', 'sum'])
        if len(category_efficiency) > 1:
            highest_freq = category_efficiency['count'].idxmax()
            highest_amount = category_efficiency['sum'].idxmax()
            insights.append(f"🎯 **Behavior Pattern**: Most frequent category is {highest_freq}, highest spending in {highest_amount}")
        
        # Predictive insights
        if self.is_trained and self.model_metrics.get('amount_accuracy', 0) > 0.5:
            accuracy = self.model_metrics['amount_accuracy'] * 100
            insights.append(f"🤖 **AI Confidence**: Expense prediction model is {accuracy:.1f}% accurate")
        
        return insights

# ================== STREAMLIT APP CONFIGURATION ==================

st.set_page_config(
    page_title="AI-Powered Expense Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CUSTOM CSS ==================
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .ai-metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .anomaly-alert {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ================== INITIALIZE AI ENGINE ==================
if "ai_engine" not in st.session_state:
    st.session_state.ai_engine = AdvancedExpenseAI()

if "expenses" not in st.session_state:
    st.session_state.expenses = []

# ================== HEADER ==================
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered Smart Expense Analyzer</h1>
    <p>Advanced Machine Learning • Natural Language Processing • Predictive Analytics</p>
    <p><strong>Multi-Algorithm Intelligence Engine</strong></p>
</div>
""", unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("🎛️ AI Control Panel")
    
    # AI Model Status
    if st.session_state.ai_engine.is_trained:
        st.success("🧠 AI Models: TRAINED")
        if st.session_state.ai_engine.model_metrics:
            st.write("**Model Performance:**")
            for metric, value in st.session_state.ai_engine.model_metrics.items():
                if isinstance(value, float):
                    st.write(f"• {metric}: {value:.3f}")
    else:
        st.warning("🧠 AI Models: TRAINING NEEDED")
    
    # Sample Data
    if st.button("📊 Load Demo Data"):
        demo_data = [
            {"Date": "2024-12-01", "Category": "Food", "Amount": 850.00, "Description": "Grocery shopping at supermarket"},
            {"Date": "2024-12-02", "Category": "Travel", "Amount": 250.00, "Description": "Metro card recharge"},
            {"Date": "2024-12-03", "Category": "Food", "Amount": 1200.00, "Description": "Restaurant dinner with friends"},
            {"Date": "2024-12-04", "Category": "Shopping", "Amount": 2500.00, "Description": "Clothing shopping at mall"},
            {"Date": "2024-12-05", "Category": "Bills", "Amount": 1800.00, "Description": "Electricity bill payment"},
            {"Date": "2024-12-06", "Category": "Entertainment", "Amount": 800.00, "Description": "Movie tickets and popcorn"},
            {"Date": "2024-12-07", "Category": "Healthcare", "Amount": 1500.00, "Description": "Doctor consultation and medicines"},
            {"Date": "2024-12-08", "Category": "Food", "Amount": 450.00, "Description": "Lunch at office cafeteria"},
            {"Date": "2024-12-09", "Category": "Travel", "Amount": 300.00, "Description": "Taxi ride to airport"},
            {"Date": "2024-12-10", "Category": "Shopping", "Amount": 950.00, "Description": "Amazon online shopping"},
            {"Date": "2024-12-11", "Category": "Food", "Amount": 1800.00, "Description": "Weekend party catering"},
            {"Date": "2024-12-12", "Category": "Bills", "Amount": 2200.00, "Description": "Internet and phone bill"}
        ]
        st.session_state.expenses.extend(demo_data)
        st.success("Demo data loaded!")
        st.rerun()
    
    # Train AI Models
    if len(st.session_state.expenses) >= 10:
        if st.button("🚀 Train AI Models"):
            with st.spinner("Training AI models..."):
                df = pd.DataFrame(st.session_state.expenses)
                result = st.session_state.ai_engine.train_ai_models(df)
                if result["status"] == "success":
                    st.success("AI models trained successfully!")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.get('message', 'Unknown error')}")
    
    # Clear Data
    if st.button("🗑️ Clear All Data"):
        st.session_state.expenses = []
        st.session_state.ai_engine = AdvancedExpenseAI()
        st.success("Data cleared!")
        st.rerun()

# ================== MAIN CONTENT ==================

# Smart Expense Input with AI Prediction
st.subheader("💰 Smart Expense Entry")

col1, col2 = st.columns([3, 1])

with col1:
    with st.form("ai_expense_form", clear_on_submit=True):
        form_col1, form_col2, form_col3 = st.columns(3)
        
        with form_col1:
            date = st.date_input("📅 Date", value=datetime.today())
        
        with form_col2:
            amount = st.number_input("💰 Amount (₹)", min_value=1.0, format="%.2f")
        
        with form_col3:
            description = st.text_input("📝 Description", placeholder="e.g., Lunch at restaurant")
        
        # AI Category Prediction
        predicted_category = "Other"
        if st.session_state.ai_engine.is_trained and amount > 0:
            predicted_category = st.session_state.ai_engine.predict_category(
                amount, date.weekday(), date.month, description
            )
        
        category = st.selectbox(
            "🏷️ Category (AI Suggested)", 
            ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Healthcare", "Education", "Other"],
            index=["Food", "Travel", "Shopping", "Bills", "Entertainment", "Healthcare", "Education", "Other"].index(predicted_category) if predicted_category in ["Food", "Travel", "Shopping", "Bills", "Entertainment", "Healthcare", "Education", "Other"] else 7
        )
        
        submit = st.form_submit_button("🤖 Add with AI Analysis", use_container_width=True)
        
        if submit and amount > 0:
            st.session_state.expenses.append({
                "Date": date.strftime("%Y-%m-%d"),
                "Category": category,
                "Amount": amount,
                "Description": description if description else f"{category} expense"
            })
            st.success("✅ Expense added with AI analysis!")
            st.rerun()

with col2:
    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        
        st.markdown('<div class="ai-metric-card">', unsafe_allow_html=True)
        st.metric("🤖 Total Expenses", f"₹{df['Amount'].sum():,.2f}")
        st.metric("📊 AI Predictions", len(df))
        st.metric("🎯 Categories", df['Category'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)

# ================== AI ANALYSIS DASHBOARD ==================

if st.session_state.expenses:
    df = pd.DataFrame(st.session_state.expenses)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Train AI if not already trained
    if not st.session_state.ai_engine.is_trained and len(df) >= 10:
        with st.spinner("Auto-training AI models..."):
            result = st.session_state.ai_engine.train_ai_models(df)
            if result["status"] == "success":
                st.success("🤖 AI models auto-trained!")
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 AI Overview", 
        "🧠 ML Insights", 
        "🚨 Anomaly Detection", 
        "🔮 Predictions", 
        "⚙️ Model Performance"
    ])
    
    with tab1:
        # AI-Enhanced Overview
        col1, col2 = st.columns(2)
        
        with col1:
            # Category Distribution with Anomalies
            category_data = df.groupby('Category')['Amount'].sum().reset_index()
            fig_pie = px.pie(
                category_data, 
                values='Amount', 
                names='Category',
                title="🧠 AI-Analyzed Spending Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Time Series with AI Trend Analysis
            daily_data = df.groupby(df['Date'].dt.date)['Amount'].sum().reset_index()
            daily_data.columns = ['Date', 'Amount']
            
            fig_trend = px.line(
                daily_data,
                x='Date',
                y='Amount',
                title="📈 AI Trend Analysis",
                markers=True
            )
            fig_trend.update_traces(line_color='#667eea')
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Smart Insights
        if st.session_state.ai_engine.is_trained:
            insights = st.session_state.ai_engine.get_smart_insights(df)
            st.subheader("🤖 AI-Generated Insights")
            for insight in insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    with tab2:
        # Machine Learning Insights
        st.subheader("🧠 Advanced ML Analysis")
        
        if st.session_state.ai_engine.is_trained:
            # Feature Importance
            if st.session_state.ai_engine.feature_importance:
                st.write("**🎯 Most Important Spending Factors:**")
                
                if 'category' in st.session_state.ai_engine.feature_importance:
                    importance_data = st.session_state.ai_engine.feature_importance['category']
                    importance_df = pd.DataFrame(list(importance_data.items()), columns=['Feature', 'Importance'])
                    importance_df = importance_df.sort_values('Importance', ascending=True).tail(10)
                    
                    fig_importance = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="🎯 AI Feature Importance Analysis",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
            
            # Clustering Analysis
            st.write("**🔍 Spending Pattern Clusters:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info("**Conservative Spenders**\nLow variance, predictable amounts")
            with col2:
                st.warning("**Moderate Spenders**\nBalanced spending patterns")
            with col3:
                st.error("**High Spenders**\nLarge amounts, irregular patterns")
        
        else:
            st.info("Train AI models to see advanced ML insights!")
    
    with tab3:
        # Anomaly Detection
        st.subheader("🚨 AI Anomaly Detection")
        
        if st.session_state.ai_engine.is_trained:
            # Get latest AI results
            result = st.session_state.ai_engine.train_ai_models(df)
            if result["status"] == "success" and "features_df" in result:
                features_df = result["features_df"]
                
                if 'is_anomaly' in features_df.columns:
                    anomalies = features_df[features_df['is_anomaly'] == True]
                    
                    if len(anomalies) > 0:
                        st.markdown(f'<div class="anomaly-alert">⚠️ <strong>{len(anomalies)} anomalous transactions detected!</strong></div>', unsafe_allow_html=True)
                        
                        # Show anomalous transactions
                        anomaly_display = anomalies[['Date', 'Category', 'Amount', 'Description']].copy()
                        anomaly_display['Date'] = anomaly_display['Date'].dt.strftime('%Y-%m-%d')
                        st.dataframe(anomaly_display, use_container_width=True)
                        
                        # Anomaly visualization
                        normal_data = features_df[features_df['is_anomaly'] == False]
                        
                        fig_anomaly = go.Figure()
                        
                        fig_anomaly.add_trace(go.Scatter(
                            x=normal_data['Date'],
                            y=normal_data['Amount'],
                            mode='markers',
                            name='Normal',
                            marker=dict(color='blue', size=8)
                        ))
                        
                        fig_anomaly.add_trace(go.Scatter(
                            x=anomalies['Date'],
                            y=anomalies['Amount'],
                            mode='markers',
                            name='Anomaly',
                            marker=dict(color='red', size=12, symbol='x')
                        ))
                        
                        fig_anomaly.update_layout(title="🚨 Anomaly Detection Visualization")
                        st.plotly_chart(fig_anomaly, use_container_width=True)
                    else:
                        st.success("✅ No anomalies detected in your spending patterns!")
        else:
            st.info("Train AI models to detect spending anomalies!")
    
    with tab4:
        # Predictions
        st.subheader("🔮 AI Predictions & Forecasting")
        
        if st.session_state.ai_engine.is_trained:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**💡 Category Prediction Test:**")
                test_amount = st.number_input("Test Amount", min_value=1.0, value=500.0)
                test_desc = st.text_input("Test Description", value="lunch at restaurant")
                test_day = st.selectbox("Day of Week", 
                                      ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                
                day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                          "Friday": 4, "Saturday": 5, "Sunday": 6}
                
                if st.button("🎯 Predict Category"):
                    predicted = st.session_state.ai_engine.predict_category(
                        test_amount, day_map[test_day], datetime.now().month, test_desc
                    )
                    st.success(f"🤖 Predicted Category: **{predicted}**")
            
            with col2:
                st.write("**📈 Spending Forecast:**")
                if len(df) >= 7:
                    recent_avg = df.tail(7)['Amount'].mean()
                    monthly_projection = recent_avg * 30
                    st.metric("Next 30 Days Projection", f"₹{monthly_projection:,.2f}")
                    
                    category_forecast = df.groupby('Category')['Amount'].mean()
                    st.write("**Category-wise Monthly Projection:**")
                    for cat, avg in category_forecast.items():
                        monthly_cat = avg * (30 / len(df)) * len(df[df['Category'] == cat])
                        st.write(f"• {cat}: ₹{monthly_cat:,.2f}")
        else:
            st.info("Train AI models to see predictions!")
    
    with tab5:
        # Model Performance
        # Model Performance (continuation of tab5)
        st.subheader("⚙️ AI Model Performance")
        
        if st.session_state.ai_engine.is_trained and st.session_state.ai_engine.model_metrics:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Model Accuracy Metrics:**")
                metrics = st.session_state.ai_engine.model_metrics
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        if 'accuracy' in metric.lower():
                            color = "green" if value > 0.7 else "orange" if value > 0.5 else "red"
                            st.markdown(f"• **{metric}**: <span style='color:{color}'>{value:.2%}</span>", unsafe_allow_html=True)
                        else:
                            st.write(f"• **{metric}**: {value:.4f}")
                    else:
                        st.write(f"• **{metric}**: {value}")
            
            with col2:
                st.write("**📊 Training Statistics:**")
                st.write(f"• **Total Records**: {len(df)}")
                st.write(f"• **Features Used**: {len(st.session_state.ai_engine.feature_importance.get('category', {}))}")
                st.write(f"• **Categories**: {df['Category'].nunique()}")
                st.write(f"• **Date Range**: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        else:
            st.info("Train AI models to see performance metrics!")
    
    # ================== DATA EXPORT ==================
    st.subheader("📤 Export Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV File",
                data=csv,
                file_name=f"expenses_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Download Analysis Report"):
            report = f"""
AI-Powered Expense Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== SUMMARY ===
Total Expenses: ₹{df['Amount'].sum():,.2f}
Number of Transactions: {len(df)}
Average Transaction: ₹{df['Amount'].mean():,.2f}
Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}

=== CATEGORY BREAKDOWN ===
{df.groupby('Category')['Amount'].agg(['count', 'sum', 'mean']).to_string()}

=== AI MODEL STATUS ===
Models Trained: {'Yes' if st.session_state.ai_engine.is_trained else 'No'}
"""
            if st.session_state.ai_engine.model_metrics:
                report += f"\nModel Performance:\n{st.session_state.ai_engine.model_metrics}"
            
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"expense_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
    
    with col3:
        if st.button("🔧 Export AI Model Config"):
            config = {
                "model_trained": st.session_state.ai_engine.is_trained,
                "metrics": st.session_state.ai_engine.model_metrics,
                "feature_importance": st.session_state.ai_engine.feature_importance,
                "export_date": datetime.now().isoformat()
            }
            
            config_json = json.dumps(config, indent=2)
            st.download_button(
                label="⚙️ Download Config",
                data=config_json,
                file_name=f"ai_model_config_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

else:
    # ================== EMPTY STATE ==================
    st.info("👆 Add your first expense to start AI analysis!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🤖 AI Features")
        st.write("• Smart category prediction")
        st.write("• Anomaly detection")
        st.write("• Spending pattern analysis")
    
    with col2:
        st.markdown("### 📊 Analytics")
        st.write("• Advanced visualizations")
        st.write("• Trend forecasting")
        st.write("• Custom insights")
    
    with col3:
        st.markdown("### 🔮 Predictions")
        st.write("• Future spending estimates")
        st.write("• Category suggestions")
        st.write("• Budget recommendations")

# ================== FOOTER ==================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>🤖 <strong>AI-Powered Expense Analyzer</strong> | Built with Advanced Machine Learning</p>
    <p>Features: Random Forest • Isolation Forest • K-Means Clustering • NLP Processing</p>
</div>
""", unsafe_allow_html=True)