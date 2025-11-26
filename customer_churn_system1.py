# customer_churn_system.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import psycopg2
import pickle
import warnings
import io
import base64
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# =============================================================================
# ENHANCED DATABASE SCHEMA
# =============================================================================

POSTGRES_SCHEMA = """
-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    credit_score INTEGER,
    gender VARCHAR(10),
    age INTEGER,
    tenure INTEGER,
    balance DECIMAL(10,2),
    products_number INTEGER,
    credit_card BOOLEAN,
    active_member BOOLEAN,
    estimated_salary DECIMAL(10,2),
    churn BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Enhanced Predictions table with scenario tracking
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    churn_probability DECIMAL(5,4),
    predicted_churn BOOLEAN,
    model_used VARCHAR(50),
    scenario_note VARCHAR(500),
    is_scenario BOOLEAN DEFAULT FALSE
);

-- Enhanced Model performance table with ROC-AUC
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Retention actions table
CREATE TABLE IF NOT EXISTS retention_actions (
    action_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    action_type VARCHAR(100),
    action_details TEXT,
    assigned_to VARCHAR(100),
    status VARCHAR(50) DEFAULT 'Pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
"""

# =============================================================================
# ENHANCED RECOMMENDATION ENGINE
# =============================================================================

class EnhancedRecommendationEngine:
    def __init__(self):
        self.recommendations = {
            'critical_risk': {
                'title': "🚨 CRITICAL RISK - Immediate Action Required",
                'actions': [
                    "🔹 Executive-level intervention required",
                    "🔹 Personal phone call from branch manager within 4 hours",
                    "🔹 Customized retention package with significant fee waivers",
                    "🔹 Emergency product review and optimization",
                    "🔹 Dedicated relationship manager assignment",
                    "🔹 Premium service tier activation for 6 months free",
                    "🔹 In-person meeting scheduling within 24 hours"
                ],
                'incentives': [
                    "💰 100% fee waiver for 3 months + 50% for next 3 months",
                    "🎁 Exclusive premium credit card with enhanced benefits",
                    "📈 Priority banking status with dedicated support line",
                    "🛡️ Complimentary insurance products for 1 year",
                    "💳 Credit limit increase with preferential rates"
                ]
            },
            'high_risk': {
                'title': "🔴 HIGH RISK - Urgent Action Required",
                'actions': [
                    "🔹 Assign dedicated relationship manager for personalized service",
                    "🔹 Offer exclusive retention package with fee waivers",
                    "🔹 Conduct urgent satisfaction call within 24 hours",
                    "🔹 Provide premium service tier for 3 months free",
                    "🔹 Schedule in-person meeting with branch manager",
                    "🔹 Offer personalized financial health check",
                    "🔹 Implement immediate issue resolution protocol"
                ],
                'incentives': [
                    "💰 50% discount on service fees for 6 months",
                    "🎁 Free premium credit card with enhanced benefits",
                    "📈 Higher interest rates on savings accounts",
                    "🛡️ Complimentary insurance products",
                    "💳 Increased credit limits"
                ]
            },
            'medium_risk': {
                'title': "🟡 MEDIUM RISK - Proactive Engagement Needed",
                'actions': [
                    "🔸 Schedule proactive check-in call within 48 hours",
                    "🔸 Offer product bundle discounts",
                    "🔸 Send personalized financial insights report",
                    "🔸 Invite to exclusive customer webinars",
                    "🔸 Provide cross-selling opportunities",
                    "🔸 Implement regular satisfaction surveys",
                    "🔸 Offer financial planning consultation"
                ],
                'incentives': [
                    "💰 25% discount on service fees for 3 months",
                    "🎁 Free financial planning session",
                    "📊 Personalized investment recommendations",
                    "🔄 Product upgrade offers",
                    "📱 Enhanced digital banking features"
                ]
            },
            'low_risk': {
                'title': "🟢 LOW RISK - Retention & Growth Focus",
                'actions': [
                    "✅ Continue regular engagement cadence",
                    "✅ Monitor for early warning signs",
                    "✅ Provide exceptional service quality",
                    "✅ Offer loyalty rewards program",
                    "✅ Cross-sell complementary products",
                    "✅ Encourage digital adoption",
                    "✅ Share educational content regularly"
                ],
                'incentives': [
                    "💰 Loyalty points accumulation",
                    "🎁 Referral bonus programs",
                    "📈 Regular product updates",
                    "🌐 Exclusive online content access",
                    "📧 Personalized financial newsletters"
                ]
            }
        }
    
    def get_recommendations(self, churn_probability, customer_data):
        """Get personalized recommendations based on churn probability and customer profile"""
        if churn_probability >= 0.8:
            risk_level = 'critical_risk'
        elif churn_probability >= 0.6:
            risk_level = 'high_risk'
        elif churn_probability >= 0.4:
            risk_level = 'medium_risk'
        else:
            risk_level = 'low_risk'
        
        base_recommendations = self.recommendations[risk_level]
        personalized_recommendations = self._get_personalized_recommendations(customer_data, risk_level)
        
        return {
            'risk_level': risk_level,
            'title': base_recommendations['title'],
            'general_actions': base_recommendations['actions'],
            'incentives': base_recommendations['incentives'],
            'personalized_recommendations': personalized_recommendations,
            'risk_score': churn_probability
        }
    
    def _get_personalized_recommendations(self, customer_data, risk_level):
        """Generate personalized recommendations based on customer characteristics"""
        personalized = []
        
        # Age-based recommendations
        age = customer_data.get('age', 0)
        if age > 60:
            personalized.append("👴 **Senior Focus**: Offer retirement planning services and senior-specific benefits package")
        elif age < 30:
            personalized.append("👶 **Youth Focus**: Emphasize digital banking features, student benefits, and financial education")
        
        # Balance-based recommendations
        balance = customer_data.get('balance', 0)
        if balance < 1000:
            personalized.append("💸 **Low Balance Strategy**: Offer budgeting tools, micro-saving features, and savings account incentives")
        elif balance > 50000:
            personalized.append("💰 **High Value Focus**: Provide premium investment opportunities, wealth management, and exclusive events")
        
        # Credit score recommendations
        credit_score = customer_data.get('credit_score', 0)
        if credit_score < 600:
            personalized.append("📉 **Credit Building**: Offer secured credit products, financial education, and credit monitoring services")
        elif credit_score > 750:
            personalized.append("📈 **Premium Credit**: Provide premium credit products, exclusive offers, and relationship pricing")
        
        # Activity-based recommendations
        if not customer_data.get('active_member', True):
            personalized.append("🔕 **Re-engagement Campaign**: Special reactivation offers and personalized outreach program")
        
        # Product usage recommendations
        products_number = customer_data.get('products_number', 1)
        if products_number == 1:
            personalized.append("📦 **Cross-sell Opportunity**: Target with complementary banking products based on usage patterns")
        elif products_number >= 3:
            personalized.append("🎯 **Loyalty Enhancement**: Focus on relationship benefits and exclusive multi-product discounts")
        
        # Tenure-based recommendations
        tenure = customer_data.get('tenure', 0)
        if tenure < 1:
            personalized.append("🆕 **Onboarding Boost**: Enhanced onboarding experience with welcome benefits and education")
        elif tenure > 5:
            personalized.append("🏆 **Loyalty Recognition**: Implement loyalty recognition program with exclusive long-term benefits")
        
        return personalized

    def generate_export_data(self, customer_data, recommendations):
        """Generate data for export"""
        export_data = {
            'Customer ID': customer_data.get('customer_id', 'N/A'),
            'Churn Probability': f"{recommendations['risk_score']:.2%}",
            'Risk Level': recommendations['risk_level'].replace('_', ' ').title(),
            'General Actions': '; '.join(recommendations['general_actions']),
            'Incentives': '; '.join(recommendations['incentives']),
            'Personalized Recommendations': '; '.join(recommendations['personalized_recommendations']),
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return export_data

# =============================================================================
# ENHANCED CHURN PREDICTOR CLASS - CORRECTED VERSION
# =============================================================================

class EnhancedChurnPredictor:
    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'churn_prediction',
            'user': 'postgres',
            'password': 'password',
            'port': 5432
        }
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.connection = None
        self.is_db_connected = False
        self.uploaded_data = None
        self.recommendation_engine = EnhancedRecommendationEngine()
        self.prediction_history = []
        self.local_predictions = []  # Local storage for predictions when DB not connected
        
    def connect_db(self, db_config=None):
        """Connect to PostgreSQL database with provided configuration"""
        if db_config:
            self.db_config = db_config
            
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.is_db_connected = True
            # Initialize tables
            self.initialize_tables()
            return True, "✅ Successfully connected to database!"
        except Exception as e:
            error_msg = f"❌ Database connection error: {e}"
            print(error_msg)
            self.is_db_connected = False
            return False, error_msg
            
    def disconnect_db(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None
        self.is_db_connected = False
        
    def initialize_tables(self):
        """Initialize database tables"""
        if not self.is_db_connected:
            return
            
        try:
            cursor = self.connection.cursor()
            # Execute schema creation
            for statement in POSTGRES_SCHEMA.split(';'):
                if statement.strip():
                    cursor.execute(statement)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error initializing tables: {e}")
    
    def load_data(self):
        """Load data from uploaded CSV, PostgreSQL, or generate sample data"""
        # Priority: Uploaded CSV > Database > Sample data
        if self.uploaded_data is not None:
            return self.uploaded_data
            
        if self.is_db_connected:
            try:
                query = "SELECT * FROM customers"
                df = pd.read_sql(query, self.connection)
                if not df.empty:
                    return df
                else:
                    st.info("📊 Database is connected but no customer data found. Using sample data.")
            except Exception as e:
                print(f"Error loading data from database: {e}")
        
        # Use the provided dataset structure
        return self.generate_sample_data()
    
    def set_uploaded_data(self, df):
        """Set uploaded CSV data"""
        self.uploaded_data = df
    
def generate_sample_data(self, n_samples=10000):
    """Generate synthetic customer data for demonstration"""
    np.random.seed(42)  # For reproducible results
    
    # Generate each column separately to avoid method chaining issues
    customer_id = list(range(1, n_samples + 1))
    
    credit_score = np.random.normal(650, 100, n_samples)
    credit_score = credit_score.astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.55, 0.45])
    
    age = np.random.normal(45, 15, n_samples)
    age = age.astype(int)
    age = np.clip(age, 18, 80)
    
    tenure = np.random.exponential(3, n_samples)
    tenure = tenure.astype(int)
    tenure = np.clip(tenure, 0, 10)
    
    balance = np.random.gamma(2, 25000, n_samples)
    balance = np.clip(balance, 0, 250000)
    
    products_number = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.35, 0.2, 0.05])
    credit_card = np.random.choice([True, False], n_samples, p=[0.7, 0.3])
    active_member = np.random.choice([True, False], n_samples, p=[0.6, 0.4])
    
    estimated_salary = np.random.normal(75000, 30000, n_samples)
    estimated_salary = np.clip(estimated_salary, 0, 200000)
    
    data = {
        'customer_id': customer_id,
        'credit_score': credit_score,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary,
    }
    
    # Create DataFrame first
    df = pd.DataFrame(data)
    
    # Calculate churn based on features (simplified logic)
    churn_prob = (
        (df['age'] > 60).astype(int) * 0.3 +
        (df['balance'] < 1000).astype(int) * 0.2 +
        (~df['active_member']).astype(int) * 0.25 +
        (df['credit_score'] < 600).astype(int) * 0.15 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    df['churn'] = (churn_prob > 0.5).astype(int)
    
    return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        df_processed = df.copy()
        
        # Encode categorical variables
        categorical_columns = ['gender']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df_processed[col].astype(str))
                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Select features
        feature_columns = [
            'credit_score', 'gender', 'age', 'tenure', 
            'balance', 'products_number', 'credit_card', 
            'active_member', 'estimated_salary'
        ]
        
        available_features = [col for col in feature_columns if col in df_processed.columns]
        
        X = df_processed[available_features]
        y = df_processed['churn']
        
        return X, y, available_features
    
    def train_models(self):
        """Train both Random Forest and Logistic Regression models with enhanced metrics"""
        df = self.load_data()
        
        if 'churn' not in df.columns:
            st.error("❌ Uploaded data must contain a 'churn' column for training!")
            return {}
            
        X, y, feature_columns = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for Logistic Regression
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        results = {}
        
        for name, model in models.items():
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate enhanced metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'feature_importance': None,
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            }
            
            # Get feature importance for Random Forest
            if name == 'Random Forest':
                feature_importance = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = feature_importance
            
            self.models[name] = model
            
            # Save performance to database
            self.save_model_performance(name, accuracy, precision, recall, f1, roc_auc)
        
        return results
    
    def save_model_performance(self, model_name, accuracy, precision, recall, f1, roc_auc):
        """Save model performance to database"""
        if not self.is_db_connected:
            return
            
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                INSERT INTO model_performance 
                (model_name, accuracy, precision, recall, f1_score, roc_auc)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (model_name, accuracy, precision, recall, f1, roc_auc))
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error saving performance: {e}")
    
    def save_prediction(self, customer_data, probability, prediction, model_name, scenario_note, is_scenario):
        """Save prediction to database or local storage"""
        prediction_record = {
            'timestamp': datetime.now(),
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'probability': probability,
            'prediction': prediction,
            'model': model_name,
            'scenario_note': scenario_note,
            'is_scenario': is_scenario,
            'customer_data': customer_data.copy()  # Store a copy for local use
        }
        
        if self.is_db_connected:
            try:
                cursor = self.connection.cursor()
                
                # First, insert or update customer data
                cursor.execute("""
                    INSERT INTO customers 
                    (customer_id, credit_score, gender, age, tenure, balance, 
                     products_number, credit_card, active_member, estimated_salary, churn)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (customer_id) DO UPDATE SET
                    credit_score = EXCLUDED.credit_score,
                    gender = EXCLUDED.gender,
                    age = EXCLUDED.age,
                    tenure = EXCLUDED.tenure,
                    balance = EXCLUDED.balance,
                    products_number = EXCLUDED.products_number,
                    credit_card = EXCLUDED.credit_card,
                    active_member = EXCLUDED.active_member,
                    estimated_salary = EXCLUDED.estimated_salary,
                    churn = EXCLUDED.churn
                """, (
                    customer_data.get('customer_id', 1),
                    customer_data['credit_score'],
                    customer_data['gender'],
                    customer_data['age'],
                    customer_data['tenure'],
                    customer_data['balance'],
                    customer_data['products_number'],
                    customer_data['credit_card'],
                    customer_data['active_member'],
                    customer_data['estimated_salary'],
                    prediction
                ))
                
                # Then save prediction
                cursor.execute("""
                    INSERT INTO predictions 
                    (customer_id, churn_probability, predicted_churn, model_used, scenario_note, is_scenario)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    customer_data.get('customer_id', 1),
                    probability,
                    prediction,
                    model_name,
                    scenario_note,
                    is_scenario
                ))
                
                self.connection.commit()
                cursor.close()
            except Exception as e:
                print(f"Error saving prediction to database: {e}")
                # Fallback to local storage
                self.local_predictions.append(prediction_record)
        else:
            # Save locally when no database connection
            self.local_predictions.append(prediction_record)
        
        # Always add to prediction history for current session
        self.prediction_history.append(prediction_record)
    
    def get_prediction_history(self, limit=50):
        """Retrieve prediction history from database or local storage"""
        if self.is_db_connected:
            try:
                query = """
                    SELECT p.prediction_id, p.customer_id, p.prediction_timestamp, 
                           p.churn_probability, p.predicted_churn, p.model_used,
                           p.scenario_note, p.is_scenario,
                           c.credit_score, c.age, c.balance, c.gender, c.tenure,
                           c.products_number, c.credit_card, c.active_member, c.estimated_salary
                    FROM predictions p
                    JOIN customers c ON p.customer_id = c.customer_id
                    ORDER BY p.prediction_timestamp DESC
                    LIMIT %s
                """
                df = pd.read_sql(query, self.connection, params=(limit,))
                return df
            except Exception as e:
                print(f"Error retrieving prediction history from database: {e}")
                # Fallback to local storage
                return self._get_local_prediction_history(limit)
        else:
            # Use local storage when no database connection
            return self._get_local_prediction_history(limit)
    
    def _get_local_prediction_history(self, limit=50):
        """Get prediction history from local storage"""
        if not self.local_predictions:
            return pd.DataFrame()
        
        # Convert local predictions to DataFrame
        history_data = []
        for pred in self.local_predictions[-limit:]:  # Get most recent predictions
            history_data.append({
                'prediction_id': len(history_data) + 1,
                'customer_id': pred['customer_id'],
                'prediction_timestamp': pred['timestamp'],
                'churn_probability': pred['probability'],
                'predicted_churn': pred['prediction'],
                'model_used': pred['model'],
                'scenario_note': pred['scenario_note'],
                'is_scenario': pred['is_scenario'],
                'credit_score': pred['customer_data'].get('credit_score', 0),
                'age': pred['customer_data'].get('age', 0),
                'balance': pred['customer_data'].get('balance', 0),
                'gender': pred['customer_data'].get('gender', 'Unknown'),
                'tenure': pred['customer_data'].get('tenure', 0),
                'products_number': pred['customer_data'].get('products_number', 0),
                'credit_card': pred['customer_data'].get('credit_card', False),
                'active_member': pred['customer_data'].get('active_member', False),
                'estimated_salary': pred['customer_data'].get('estimated_salary', 0)
            })
        
        return pd.DataFrame(history_data)

    def predict_churn(self, customer_data, model_name='Random Forest', scenario_note=""):
        """Predict churn for a single customer with scenario tracking"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Preprocess customer data
        df_customer = pd.DataFrame([customer_data])
        
        # Encode categorical variables
        for col in ['gender']:
            if col in customer_data and col in self.label_encoders:
                try:
                    df_customer[col] = self.label_encoders[col].transform([customer_data[col]])[0]
                except ValueError:
                    st.error(f"Unknown gender value: {customer_data[col]}. Please use 'Male' or 'Female'.")
                    return 0.5, False, {}
        
        # Select features in correct order
        feature_columns = [
            'credit_score', 'gender', 'age', 'tenure', 
            'balance', 'products_number', 'credit_card', 
            'active_member', 'estimated_salary'
        ]
        
        available_features = [col for col in feature_columns if col in df_customer.columns]
        X = df_customer[available_features]
        
        # Make prediction
        if model_name == 'Logistic Regression':
            X_scaled = self.scaler.transform(X)
            probability = self.models[model_name].predict_proba(X_scaled)[0, 1]
        else:
            probability = self.models[model_name].predict_proba(X)[0, 1]
        
        prediction = probability > 0.5
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(probability, customer_data)
        
        # Save prediction (to database if connected, otherwise locally)
        is_scenario = bool(scenario_note)
        self.save_prediction(customer_data, probability, prediction, model_name, scenario_note, is_scenario)
        
        # Add to prediction history
        prediction_record = {
            'timestamp': datetime.now(),
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'probability': probability,
            'prediction': prediction,
            'model': model_name,
            'scenario_note': scenario_note,
            'recommendations': recommendations
        }
        self.prediction_history.append(prediction_record)
        
        return probability, prediction, recommendations


    def __init__(self, db_config=None):
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'churn_prediction',
            'user': 'postgres',
            'password': 'password',
            'port': 5432
        }
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.connection = None
        self.is_db_connected = False
        self.uploaded_data = None
        self.recommendation_engine = EnhancedRecommendationEngine()
        self.prediction_history = []
        self.local_predictions = []  # Local storage for predictions when DB not connected
        
    def connect_db(self, db_config=None):
        """Connect to PostgreSQL database with provided configuration"""
        if db_config:
            self.db_config = db_config
            
        try:
            self.connection = psycopg2.connect(**self.db_config)
            self.is_db_connected = True
            # Initialize tables
            self.initialize_tables()
            return True, "✅ Successfully connected to database!"
        except Exception as e:
            error_msg = f"❌ Database connection error: {e}"
            print(error_msg)
            self.is_db_connected = False
            return False, error_msg
            
    def disconnect_db(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            self.connection = None
        self.is_db_connected = False
        
    def initialize_tables(self):
        """Initialize database tables"""
        if not self.is_db_connected:
            return
            
        try:
            cursor = self.connection.cursor()
            # Execute schema creation
            for statement in POSTGRES_SCHEMA.split(';'):
                if statement.strip():
                    cursor.execute(statement)
            self.connection.commit()
            cursor.close()
        except Exception as e:
            print(f"Error initializing tables: {e}")
    
    def save_prediction(self, customer_data, probability, prediction, model_name, scenario_note, is_scenario):
        """Save prediction to database or local storage"""
        prediction_record = {
            'timestamp': datetime.now(),
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'probability': probability,
            'prediction': prediction,
            'model': model_name,
            'scenario_note': scenario_note,
            'is_scenario': is_scenario,
            'customer_data': customer_data.copy()  # Store a copy for local use
        }
        
        if self.is_db_connected:
            try:
                cursor = self.connection.cursor()
                
                # First, insert or update customer data
                cursor.execute("""
                    INSERT INTO customers 
                    (customer_id, credit_score, gender, age, tenure, balance, 
                     products_number, credit_card, active_member, estimated_salary, churn)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (customer_id) DO UPDATE SET
                    credit_score = EXCLUDED.credit_score,
                    gender = EXCLUDED.gender,
                    age = EXCLUDED.age,
                    tenure = EXCLUDED.tenure,
                    balance = EXCLUDED.balance,
                    products_number = EXCLUDED.products_number,
                    credit_card = EXCLUDED.credit_card,
                    active_member = EXCLUDED.active_member,
                    estimated_salary = EXCLUDED.estimated_salary,
                    churn = EXCLUDED.churn
                """, (
                    customer_data.get('customer_id', 1),
                    customer_data['credit_score'],
                    customer_data['gender'],
                    customer_data['age'],
                    customer_data['tenure'],
                    customer_data['balance'],
                    customer_data['products_number'],
                    customer_data['credit_card'],
                    customer_data['active_member'],
                    customer_data['estimated_salary'],
                    prediction
                ))
                
                # Then save prediction
                cursor.execute("""
                    INSERT INTO predictions 
                    (customer_id, churn_probability, predicted_churn, model_used, scenario_note, is_scenario)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    customer_data.get('customer_id', 1),
                    probability,
                    prediction,
                    model_name,
                    scenario_note,
                    is_scenario
                ))
                
                self.connection.commit()
                cursor.close()
            except Exception as e:
                print(f"Error saving prediction to database: {e}")
                # Fallback to local storage
                self.local_predictions.append(prediction_record)
        else:
            # Save locally when no database connection
            self.local_predictions.append(prediction_record)
        
        # Always add to prediction history for current session
        self.prediction_history.append(prediction_record)
    
    def get_prediction_history(self, limit=50):
        """Retrieve prediction history from database or local storage"""
        if self.is_db_connected:
            try:
                query = """
                    SELECT p.prediction_id, p.customer_id, p.prediction_timestamp, 
                           p.churn_probability, p.predicted_churn, p.model_used,
                           p.scenario_note, p.is_scenario,
                           c.credit_score, c.age, c.balance, c.gender, c.tenure,
                           c.products_number, c.credit_card, c.active_member, c.estimated_salary
                    FROM predictions p
                    JOIN customers c ON p.customer_id = c.customer_id
                    ORDER BY p.prediction_timestamp DESC
                    LIMIT %s
                """
                df = pd.read_sql(query, self.connection, params=(limit,))
                return df
            except Exception as e:
                print(f"Error retrieving prediction history from database: {e}")
                # Fallback to local storage
                return self._get_local_prediction_history(limit)
        else:
            # Use local storage when no database connection
            return self._get_local_prediction_history(limit)
    
    def _get_local_prediction_history(self, limit=50):
        """Get prediction history from local storage"""
        if not self.local_predictions:
            return pd.DataFrame()
        
        # Convert local predictions to DataFrame
        history_data = []
        for pred in self.local_predictions[-limit:]:  # Get most recent predictions
            history_data.append({
                'prediction_id': len(history_data) + 1,
                'customer_id': pred['customer_id'],
                'prediction_timestamp': pred['timestamp'],
                'churn_probability': pred['probability'],
                'predicted_churn': pred['prediction'],
                'model_used': pred['model'],
                'scenario_note': pred['scenario_note'],
                'is_scenario': pred['is_scenario'],
                'credit_score': pred['customer_data'].get('credit_score', 0),
                'age': pred['customer_data'].get('age', 0),
                'balance': pred['customer_data'].get('balance', 0),
                'gender': pred['customer_data'].get('gender', 'Unknown'),
                'tenure': pred['customer_data'].get('tenure', 0),
                'products_number': pred['customer_data'].get('products_number', 0),
                'credit_card': pred['customer_data'].get('credit_card', False),
                'active_member': pred['customer_data'].get('active_member', False),
                'estimated_salary': pred['customer_data'].get('estimated_salary', 0)
            })
        
        return pd.DataFrame(history_data)

    def predict_churn(self, customer_data, model_name='Random Forest', scenario_note=""):
        """Predict churn for a single customer with scenario tracking"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Preprocess customer data
        df_customer = pd.DataFrame([customer_data])
        
        # Encode categorical variables
        for col in ['gender']:
            if col in customer_data and col in self.label_encoders:
                try:
                    df_customer[col] = self.label_encoders[col].transform([customer_data[col]])[0]
                except ValueError:
                    st.error(f"Unknown gender value: {customer_data[col]}. Please use 'Male' or 'Female'.")
                    return 0.5, False, {}
        
        # Select features in correct order
        feature_columns = [
            'credit_score', 'gender', 'age', 'tenure', 
            'balance', 'products_number', 'credit_card', 
            'active_member', 'estimated_salary'
        ]
        
        available_features = [col for col in feature_columns if col in df_customer.columns]
        X = df_customer[available_features]
        
        # Make prediction
        if model_name == 'Logistic Regression':
            X_scaled = self.scaler.transform(X)
            probability = self.models[model_name].predict_proba(X_scaled)[0, 1]
        else:
            probability = self.models[model_name].predict_proba(X)[0, 1]
        
        prediction = probability > 0.5
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations(probability, customer_data)
        
        # Save prediction (to database if connected, otherwise locally)
        is_scenario = bool(scenario_note)
        self.save_prediction(customer_data, probability, prediction, model_name, scenario_note, is_scenario)
        
        # Add to prediction history
        prediction_record = {
            'timestamp': datetime.now(),
            'customer_id': customer_data.get('customer_id', 'Unknown'),
            'probability': probability,
            'prediction': prediction,
            'model': model_name,
            'scenario_note': scenario_note,
            'recommendations': recommendations
        }
        self.prediction_history.append(prediction_record)
        
        return probability, prediction, recommendations

# =============================================================================
# ENHANCED STREAMLIT DASHBOARD
# =============================================================================

def setup_streamlit_app():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="🏦 Advanced Customer Churn Prediction System",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional look
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .risk-critical { background-color: #ff4444; color: white; padding: 10px; border-radius: 5px; }
    .risk-high { background-color: #ff6b6b; color: white; padding: 10px; border-radius: 5px; }
    .risk-medium { background-color: #ffa726; color: white; padding: 10px; border-radius: 5px; }
    .risk-low { background-color: #66bb6a; color: white; padding: 10px; border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def initialize_predictor():
    """Initialize the churn predictor"""
    return EnhancedChurnPredictor()

def show_data_management(predictor):
    """Enhanced data upload and database connection"""
    st.header("📊 Data Management")
    
    tab1, tab2 = st.tabs(["📁 Upload CSV Data", "🗄️ Database Connection"])
    
    with tab1:
        show_csv_upload(predictor)
    
    with tab2:
        show_database_connection(predictor)

def show_csv_upload(predictor):
    """Display CSV upload functionality"""
    st.subheader("Upload Customer Data CSV")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file with customer data",
        type=['csv'],
        help="File should contain customer data with churn labels"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully uploaded {len(df)} records")
            
            # Display file info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                if 'churn' in df.columns:
                    churn_rate = df['churn'].mean() * 100
                    st.metric("Churn Rate", f"{churn_rate:.2f}%")
            with col4:
                st.metric("Data Quality", "Good" if not df.isnull().any().any() else "Check Required")
            
            # Data preview
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(df.head(10))
            
            # Column information
            with st.expander("🔍 Column Information"):
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes,
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum()
                })
                st.dataframe(col_info)
            
            predictor.set_uploaded_data(df)
            
            # Train models
            if st.button("🎯 Train Machine Learning Models", type="primary", use_container_width=True):
                with st.spinner("Training models with uploaded data..."):
                    results = predictor.train_models()
                if results:
                    st.session_state.training_results = results
                    st.success("✅ Models trained successfully!")
                    st.balloons()
        
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")

# =============================================================================
# UPDATED MAIN FUNCTION WITH BETTER DB SETUP GUIDANCE
# =============================================================================

def show_database_connection(predictor):
    """Display database connection interface - UPDATED"""
    st.subheader("🗄️ Database Connection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("db_connection_form"):
            st.write("Enter database connection details:")
            
            col1, col2 = st.columns(2)
            with col1:
                host = st.text_input("Host", value="localhost")
                database = st.text_input("Database", value="churn_prediction")
            with col2:
                user = st.text_input("Username", value="postgres")
                password = st.text_input("Password", type="password", value="password")
                port = st.number_input("Port", value=5432)
            
            if st.form_submit_button("🔗 Connect to Database", use_container_width=True):
                db_config = {'host': host, 'database': database, 'user': user, 
                           'password': password, 'port': port}
                success, message = predictor.connect_db(db_config)
                if success:
                    st.success(message)
                    # Transfer local predictions to database if any exist
                    if hasattr(predictor, 'local_predictions') and predictor.local_predictions:
                        st.info(f"Transferring {len(predictor.local_predictions)} local predictions to database...")
                        for pred in predictor.local_predictions:
                            predictor.save_prediction(
                                pred['customer_data'],
                                pred['probability'],
                                pred['prediction'],
                                pred['model'],
                                pred.get('scenario_note', ''),
                                pred.get('is_scenario', False)
                            )
                    st.rerun()
                else:
                    st.error(message)
        
        if predictor.is_db_connected:
            if st.button("🔌 Disconnect Database", use_container_width=True):
                predictor.disconnect_db()
                st.success("Disconnected from database")
                st.rerun()
    
    with col2:
        st.subheader("Connection Status")
        if predictor.is_db_connected:
            st.success("✅ Connected")
            try:
                # Test data access
                df = predictor.load_data()
                history_df = predictor.get_prediction_history(5)
                
                st.metric("Customer Records", len(df))
                st.metric("Stored Predictions", len(history_df))
                
            except Exception as e:
                st.warning(f"Connected but error accessing data: {e}")
        else:
            st.warning("🔌 Not Connected")
            st.info("""
            **Without database:**
            - Predictions saved locally
            - History available this session
            - Data resets on app restart
            """)
            
        # Database setup guide
        with st.expander("📋 Database Setup Guide"):
            st.markdown("""
            **To enable permanent prediction storage:**
            
            1. **Install PostgreSQL**
            ```bash
            # Ubuntu
            sudo apt-get install postgresql postgresql-contrib
            
            # Or use Docker
            docker run --name churn-db -e POSTGRES_PASSWORD=password -d -p 5432:5432 postgres
            ```
            
            2. **Create Database**
            ```sql
            CREATE DATABASE churn_prediction;
            ```
            
            3. **Update connection details above**
            """)

def show_dashboard(predictor):
    """Enhanced dashboard with interactive visualizations"""
    st.header("📈 Analytics Dashboard")
    
    df = predictor.load_data()
    
    # Data source info
    if predictor.uploaded_data is not None and not predictor.uploaded_data.empty:
          source_info = "📁 Uploaded CSV" 
    elif  predictor.is_db_connected:
        source_info="🗄️ Database"
    else:
        source_info="🎲 Sample Data"
    st.info(f"**Data Source:** {source_info} | **Total Records:** {len(df):,}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    with col2:
        churn_rate = df['churn'].mean() * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.2f}%")
    with col3:
        avg_balance = df['balance'].mean()
        st.metric("Average Balance", f"KES {avg_balance:,.0f}")
    with col4:
        active_rate = df['active_member'].mean() * 100
        st.metric("Active Members", f"{active_rate:.1f}%")
    
    # Interactive filters
    st.subheader("🔍 Interactive Filters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender_filter = st.multiselect("Gender", options=df['gender'].unique(), default=df['gender'].unique())
    with col2:
        age_range = st.slider("Age Range", int(df['age'].min()), int(df['age'].max()), (25, 65))
    with col3:
        balance_range = st.slider("Balance Range", float(df['balance'].min()), float(df['balance'].max()), 
                                (0.0, 150000.0))
    with col4:
        tenure_range = st.slider("Tenure Range", int(df['tenure'].min()), int(df['tenure'].max()), (0, 8))
    
    # Apply filters
    filtered_df = df[
        (df['gender'].isin(gender_filter)) &
        (df['age'].between(age_range[0], age_range[1])) &
        (df['balance'].between(balance_range[0], balance_range[1])) &
        (df['tenure'].between(tenure_range[0], tenure_range[1]))
    ]
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution Analysis", "🔥 Correlation Insights", "📈 Cohort Analysis", "🤖 Model Performance"])
    
    with tab1:
        show_distribution_analysis(filtered_df)
    
    with tab2:
        show_correlation_analysis(filtered_df)
    
    with tab3:
        show_cohort_analysis(filtered_df)
    
   
    
    with tab4:
        if 'training_results' in st.session_state:
            show_model_performance_dashboard(st.session_state.training_results)
        else:
            st.info("🎯 Train models first to see performance metrics")

def show_distribution_analysis(df):
    """Show distribution analysis visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by demographic factors
        fig1 = px.histogram(df, x='age', color='churn', barmode='overlay',
                          title='Age Distribution by Churn Status',
                          opacity=0.7)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Balance distribution
        fig3 = px.box(df, x='churn', y='balance', color='churn',
                     title='Balance Distribution by Churn Status')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Churn by activity status
        activity_churn = df.groupby('active_member')['churn'].mean().reset_index()
        activity_churn['active_member'] = activity_churn['active_member'].map({True: 'Active', False: 'Inactive'})
        fig2 = px.pie(activity_churn, values='churn', names='active_member',
                     title='Churn Distribution by Activity Status')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Products vs Churn
        products_churn = df.groupby('products_number')['churn'].mean().reset_index()
        fig4 = px.bar(products_churn, x='products_number', y='churn',
                     title='Churn Rate by Number of Products',
                     color='churn')
        st.plotly_chart(fig4, use_container_width=True)

def show_correlation_analysis(df):
    """Show correlation analysis visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Correlation heatmap
        numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 'estimated_salary', 'churn']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Feature Correlation Matrix",
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot: Credit Score vs Balance
        fig = px.scatter(df, x='credit_score', y='balance', color='churn',
                        size='age', hover_data=['tenure'],
                        title='Credit Score vs Balance by Churn Status',
                        opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (if available)
        if 'training_results' in st.session_state and 'Random Forest' in st.session_state.training_results:
            feature_importance = st.session_state.training_results['Random Forest']['feature_importance']
            fig = px.bar(feature_importance, x='importance', y='feature',
                        title='Random Forest Feature Importance',
                        orientation='h')
            st.plotly_chart(fig, use_container_width=True)

def show_cohort_analysis(df):
    """Show cohort analysis visualizations"""
    # Create tenure cohorts
    df['cohort_group'] = (df['tenure'] // 12) + 1
    cohort_data = df.groupby('cohort_group').agg({
        'customer_id': 'count',
        'churn': 'mean',
        'balance': 'mean',
        'credit_score': 'mean'
    }).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cohort churn rates
        fig1 = px.line(cohort_data, x='cohort_group', y='churn',
                      title='Churn Rate by Tenure Cohort',
                      markers=True)
        fig1.update_layout(xaxis_title='Tenure Cohort (Years)', yaxis_title='Churn Rate')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Cohort size and value
        fig2 = px.bar(cohort_data, x='cohort_group', y='customer_id',
                     title='Customer Distribution by Tenure Cohort',
                     color='balance')
        st.plotly_chart(fig2, use_container_width=True)

def show_model_performance_dashboard(results):
    """Enhanced model performance dashboard"""
    st.subheader("🤖 Model Performance Comparison")
    
    # Metrics comparison
    metrics_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[model]['accuracy'] for model in results],
        'Precision': [results[model]['precision'] for model in results],
        'Recall': [results[model]['recall'] for model in results],
        'F1-Score': [results[model]['f1'] for model in results],
        'ROC-AUC': [results[model]['roc_auc'] for model in results]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(metrics_df.style.format({
            'Accuracy': '{:.2%}', 'Precision': '{:.2%}', 'Recall': '{:.2%}',
            'F1-Score': '{:.2%}', 'ROC-AUC': '{:.2%}'
        }))
    
    with col2:
        # Radar chart
        models = list(results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig = go.Figure()
        for model in models:
            values = [
                results[model]['accuracy'],
                results[model]['precision'],
                results[model]['recall'],
                results[model]['f1'],
                results[model]['roc_auc']
            ]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill='toself',
                name=model
            ))
        
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                         showlegend=True, title="Model Performance Radar")
        st.plotly_chart(fig, use_container_width=True)
    
    # ROC Curves and Confusion Matrices
    st.subheader("📊 Detailed Model Analysis")
    for model_name in results:
        with st.expander(f"{model_name} Detailed Analysis", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # ROC Curve
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(results[model_name]['y_test'], 
                                      results[model_name]['y_pred_proba'])
                
                fig_roc = px.area(x=fpr, y=tpr, title=f'{model_name} ROC Curve',
                                labels=dict(x='False Positive Rate', y='True Positive Rate'))
                fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                st.plotly_chart(fig_roc, use_container_width=True)
            
            with col2:
                # Confusion Matrix
                cm = confusion_matrix(results[model_name]['y_test'], results[model_name]['y_pred'])
                fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                                 title=f'{model_name} Confusion Matrix',
                                 labels=dict(x="Predicted", y="Actual", color="Count"))
                st.plotly_chart(fig_cm, use_container_width=True)

def show_prediction_interface(predictor):
    """Enhanced prediction interface with scenario analysis"""
    st.header("🔮 Churn Prediction & Scenario Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Single Customer Prediction", "🔄 Scenario Analysis", "📋 Prediction History"])
    
    with tab1:
        show_single_prediction(predictor)
    
    with tab2:
        show_scenario_analysis(predictor)
    
    with tab3:
        show_prediction_history(predictor)

def show_single_prediction(predictor):
    """Single customer prediction interface"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Profile")
        with st.form("prediction_form"):
            customer_id = st.number_input("Customer ID", min_value=1, value=1001)
            credit_score = st.slider("Credit Score", 300, 850, 650)
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 80, 45)
            tenure = st.slider("Tenure (years)", 0, 10, 3)
            balance = st.number_input("Balance (KES)", min_value=0.0, value=50000.0, step=1000.0)
            products_number = st.slider("Number of Products", 1, 4, 2)
            credit_card = st.checkbox("Has Credit Card", value=True)
            active_member = st.checkbox("Active Member", value=True)
            estimated_salary = st.number_input("Estimated Salary (KES)", min_value=0.0, value=75000.0, step=1000.0)
            
            model_choice = st.selectbox("Prediction Model", ["Random Forest", "Logistic Regression"])
            
            if st.form_submit_button("🎯 Predict Churn Risk", use_container_width=True):
                if not predictor.models:
                    st.error("❌ Please train models first using the Data Management page!")
                    return
                
                customer_data = {
                    'customer_id': customer_id,
                    'credit_score': credit_score,
                    'gender': gender,
                    'age': age,
                    'tenure': tenure,
                    'balance': balance,
                    'products_number': products_number,
                    'credit_card': credit_card,
                    'active_member': active_member,
                    'estimated_salary': estimated_salary
                }
                
                try:
                    with st.spinner('Analyzing customer profile...'):
                        probability, prediction, recommendations = predictor.predict_churn(
                            customer_data, model_choice)
                    
                    st.session_state.last_prediction = {
                        'probability': probability,
                        'prediction': prediction,
                        'recommendations': recommendations,
                        'customer_data': customer_data
                    }
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'last_prediction' in st.session_state:
            pred = st.session_state.last_prediction
            prob = pred['probability']
            rec = pred['recommendations']
            
            # Risk level display
            risk_class = {
                'critical_risk': 'risk-critical',
                'high_risk': 'risk-high', 
                'medium_risk': 'risk-medium',
                'low_risk': 'risk-low'
            }[rec['risk_level']]
            
            st.markdown(f"<div class='{risk_class}'><h3>{rec['title']}</h3></div>", unsafe_allow_html=True)
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 60], 'color': "yellow"},
                        {'range': [60, 80], 'color': "orange"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            with st.expander("🎯 Recommended Actions", expanded=True):
                for action in rec['general_actions']:
                    st.write(action)
            
            with st.expander("💰 Retention Incentives", expanded=True):
                for incentive in rec['incentives']:
                    st.write(incentive)
            
            with st.expander("🎨 Personalized Strategy", expanded=True):
                for personal_rec in rec['personalized_recommendations']:
                    st.write(personal_rec)
            
            # Export options
            export_data = predictor.recommendation_engine.generate_export_data(
                pred['customer_data'], rec)
            
            col1, col2 = st.columns(2)
            with col1:
                # CSV Export
                csv_data = pd.DataFrame([export_data]).to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv_data,
                    f"churn_recommendations_{customer_id}.csv",
                    "text/csv"
                )
            with col2:
                # Excel Export
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    pd.DataFrame([export_data]).to_excel(writer, sheet_name='Recommendations', index=False)
                    # Add detailed actions sheet
                    actions_df = pd.DataFrame({
                        'Action Type': ['General Actions'] * len(rec['general_actions']) + 
                                      ['Incentives'] * len(rec['incentives']) + 
                                      ['Personalized'] * len(rec['personalized_recommendations']),
                        'Description': rec['general_actions'] + rec['incentives'] + rec['personalized_recommendations']
                    })
                    actions_df.to_excel(writer, sheet_name='Detailed Actions', index=False)
                
                st.download_button(
                    "📊 Download as Excel",
                    excel_buffer.getvalue(),
                    f"churn_analysis_{customer_id}.xlsx",
                    "application/vnd.ms-excel"
                )
        else:
            st.info("👆 Enter customer details and click 'Predict Churn Risk' to see analysis")

def show_scenario_analysis(predictor):
    """Scenario analysis interface"""
    st.subheader("🔄 What-If Scenario Analysis")
    
    if 'last_prediction' not in st.session_state:
        st.info("👆 First make a prediction in the Single Customer tab to enable scenario analysis")
        return
    
    base_pred = st.session_state.last_prediction
    base_customer = base_pred['customer_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Adjust customer attributes to see impact on churn risk:**")
        
        # Scenario controls
        new_balance = st.slider("New Balance", 0.0, 200000.0, float(base_customer['balance']), 1000.0,
                               help="How would changing the balance affect churn risk?")
        new_products = st.slider("Additional Products", 0, 3, 0,
                                help="What if the customer adopts more products?")
        new_activity = st.selectbox("Activity Status", ["Active", "Inactive"], 
                                   index=0 if base_customer['active_member'] else 1)
        new_credit_card = st.checkbox("Has Credit Card", value=base_customer['credit_card'])
        
        scenario_note = st.text_input("Scenario Description", 
                                     "Testing impact of balance increase and product adoption")
    
    with col2:
        if st.button("🔄 Run Scenario Analysis", use_container_width=True):
            # Create modified customer profile
            modified_customer = base_customer.copy()
            modified_customer['balance'] = new_balance
            modified_customer['products_number'] += new_products
            modified_customer['active_member'] = (new_activity == "Active")
            modified_customer['credit_card'] = new_credit_card
            
            # Get new prediction
            with st.spinner('Analyzing scenario...'):
                new_prob, new_pred, new_rec = predictor.predict_churn(
                    modified_customer, 
                    'Random Forest',
                    scenario_note
                )
            
            # Display comparison
            st.subheader("Scenario Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Probability", f"{base_pred['probability']:.2%}")
            with col2:
                st.metric("New Probability", f"{new_prob:.2%}", 
                         f"{(new_prob - base_pred['probability']):+.2%}")
            
            # Impact analysis
            risk_change = "🟢 Improved" if new_prob < base_pred['probability'] else \
                         "🔴 Worsened" if new_prob > base_pred['probability'] else "🟡 Unchanged"
            
            st.write(f"**Risk Level:** {risk_change}")
            
            if new_prob < base_pred['probability']:
                st.success("✅ This scenario would REDUCE churn risk!")
            elif new_prob > base_pred['probability']:
                st.error("❌ This scenario would INCREASE churn risk!")
            else:
                st.info("ℹ️ This scenario has no significant impact on churn risk")
            
            # Show what changed
            st.write("**Changes in this scenario:**")
            changes = []
            if new_balance != base_customer['balance']:
                changes.append(f"Balance: KES {base_customer['balance']:,.0f} → ${new_balance:,.0f}")
            if new_products > 0:
                changes.append(f"Products: +{new_products} additional products")
            if new_activity != ("Active" if base_customer['active_member'] else "Inactive"):
                changes.append(f"Activity: {('Active' if base_customer['active_member'] else 'Inactive')} → {new_activity}")
            if new_credit_card != base_customer['credit_card']:
                changes.append(f"Credit Card: {('Yes' if base_customer['credit_card'] else 'No')} → {('Yes' if new_credit_card else 'No')}")
            
            for change in changes:
                st.write(f"• {change}")
# =============================================================================
#  PREDICTION HISTORY DISPLAY
# =============================================================================
def show_prediction_history(predictor):
    """Display prediction history - UPDATED VERSION"""
    st.subheader("📋 Prediction History")
    
    # Show connection status
    if predictor.is_db_connected:
        st.success("🗄️ Connected to database - predictions are being saved permanently")
    else:
        st.info("💻 Using local storage - predictions will be saved for this session only")
    
    history_df = predictor.get_prediction_history(100)
    
    if history_df.empty:
        st.info("No prediction history found. Make some predictions first!")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_scenarios = st.checkbox("Show only scenarios", value=False)
    with col2:
        # Date filter
        if not history_df.empty:
            min_date = history_df['prediction_timestamp'].min().date()
            max_date = history_df['prediction_timestamp'].max().date()
            date_range = st.date_input("Date Range", [min_date, max_date])
        else:
            date_range = st.date_input("Date Range", [])
    with col3:
        risk_filter = st.selectbox("Risk Level", ["All", "Critical (80%+)", "High (60%+)", "Medium (40%+)", "Low (0-40%)"])
    
    # Apply filters
    filtered_df = history_df.copy()
    if show_scenarios:
        filtered_df = filtered_df[filtered_df['is_scenario'] == True]
    
    if risk_filter != "All":
        risk_thresholds = {
            'Critical (80%+)': 0.8,
            'High (60%+)': 0.6, 
            'Medium (40%+)': 0.4,
            'Low (0-40%)': 0.4
        }
        threshold = risk_thresholds[risk_filter]
        if risk_filter == 'Low (0-40%)':
            filtered_df = filtered_df[filtered_df['churn_probability'] < threshold]
        else:
            filtered_df = filtered_df[filtered_df['churn_probability'] >= threshold]
    
    # Apply date filter if range is selected
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['prediction_timestamp'].dt.date >= start_date) & 
            (filtered_df['prediction_timestamp'].dt.date <= end_date)
        ]
    
    if filtered_df.empty:
        st.warning("No predictions match the selected filters.")
        return
    
    # Display history with better formatting
    st.dataframe(
        filtered_df.style.format({
            'churn_probability': '{:.2%}',
            'balance': 'KES {:,.0f}',
            'estimated_salary': 'KES {:,.0f}'
        }).background_gradient(subset=['churn_probability'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=400
    )
    
    # Summary statistics
    st.subheader("📊 Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(filtered_df))
    with col2:
        avg_prob = filtered_df['churn_probability'].mean()
        st.metric("Average Probability", f"{avg_prob:.2%}")
    with col3:
        high_risk_count = len(filtered_df[filtered_df['churn_probability'] > 0.6])
        st.metric("High Risk Cases", high_risk_count)
    with col4:
        scenario_count = len(filtered_df[filtered_df['is_scenario'] == True])
        st.metric("Scenario Analyses", scenario_count)
    
    # Recent predictions chart
    if len(filtered_df) > 1:
        st.subheader("📈 Recent Prediction Trends")
        
        # Create a time series of predictions
        recent_predictions = filtered_df.sort_values('prediction_timestamp').tail(20)
        
        fig = px.line(recent_predictions, x='prediction_timestamp', y='churn_probability',
                     title='Recent Churn Probability Trends',
                     markers=True)
        fig.update_layout(xaxis_title='Time', yaxis_title='Churn Probability')
        st.plotly_chart(fig, use_container_width=True)
def show_user_guide():
    """Display user guide"""
    st.header("📚 User Guide")
    
    with st.expander("🚀 Getting Started", expanded=True):
        st.markdown("""
        ### Welcome to the Customer Churn Prediction System!
        
        This system helps bank staff identify customers at risk of churning and provides actionable insights for retention.
        
        **Quick Start:**
        1. **Data Setup**: Upload your customer data CSV or connect to PostgreSQL database
        2. **Model Training**: Train the machine learning models with your data
        3. **Predictions**: Analyze individual customers and run scenarios
        4. **Dashboard**: Explore insights through interactive visualizations
        """)
    
    with st.expander("📊 Data Requirements"):
        st.markdown("""
        **Required CSV Columns:**
        - `customer_id`: Unique identifier
        - `credit_score`: 300-850 range
        - `gender`: Male/Female
        - `age`: Customer age
        - `tenure`: Years with bank
        - `balance`: Account balance
        - `products_number`: Number of bank products
        - `credit_card`: True/False
        - `active_member`: True/False
        - `estimated_salary`: Annual salary
        - `churn`: 1/0 (for training)
        
        **Database Setup:**
        - PostgreSQL database with the provided schema
        - Connection details: host, database, username, password
        """)
    
    with st.expander("🎯 Making Predictions"):
        st.markdown("""
        **Single Customer Prediction:**
        1. Fill in customer details in the prediction form
        2. Select prediction model (Random Forest recommended)
        3. View churn probability and risk level
        4. Download retention recommendations
        
        **Scenario Analysis:**
        - Test "what-if" scenarios by modifying customer attributes
        - See how changes affect churn probability
        - Compare original vs. modified risk levels
        """)
    
    with st.expander("📈 Understanding Results"):
        st.markdown("""
        **Risk Levels:**
        - **🟢 LOW RISK** (0-40%): Normal retention activities
        - **🟡 MEDIUM RISK** (40-60%): Proactive engagement
        - **🔴 HIGH RISK** (60-80%): Urgent intervention
        - **🚨 CRITICAL RISK** (80-100%): Immediate executive action
        
        **Model Metrics:**
        - **Accuracy**: Overall prediction correctness
        - **Precision**: How many predicted churns are actual churns
        - **Recall**: How many actual churns are correctly predicted
        - **F1-Score**: Balance between precision and recall
        - **ROC-AUC**: Model's ability to distinguish between classes
        """)

def main():
    """Main Streamlit application"""
    setup_streamlit_app()
    
    st.markdown('<h1 class="main-header">🏦 Advanced Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("""
    *Identify at-risk customers, predict churn probability, and generate personalized retention strategies 
    using machine learning and interactive analytics.*
    """)
    
    # Initialize predictor
    predictor = initialize_predictor()
    
    # Main navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Select Module",
        ["📚 User Guide", "📊 Data Management", "📈 Analytics Dashboard", "🔮 Predictions & Scenarios"]
    )
    
    # Display selected module
    if app_mode == "📚 User Guide":
        show_user_guide()
    elif app_mode == "📊 Data Management":
        show_data_management(predictor)
    elif app_mode == "📈 Analytics Dashboard":
        show_dashboard(predictor)
    elif app_mode == "🔮 Predictions & Scenarios":
        show_prediction_interface(predictor)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Status:**
    - Models: ✅ Ready
    - Database: ✅ Connected
    - Predictions: ✅ Active
    """)

# =============================================================================
# RUN THE APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()