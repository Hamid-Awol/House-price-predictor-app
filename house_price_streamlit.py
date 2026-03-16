# house_price_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

class HousePricePredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_col = None
        
    def load_data(self, file):
        """Load dataset from uploaded file"""
        try:
            self.df = pd.read_csv(file)
            return True
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
            return False
    
    def identify_price_column(self):
        """Identify the price/target column"""
        price_keywords = ['price', 'sale_price', 'saleprice', 'sale price', 
                         'house price', 'property price', 'target', 'cost', 'value']
        
        for col in self.df.columns:
            col_lower = col.lower().strip()
            if any(keyword in col_lower for keyword in price_keywords):
                return col
        
        return self.df.columns[-1]
    
    def preprocess_data(self, target_col, test_size=0.2, random_state=42):
        """Preprocess the data"""
        self.target_col = target_col
        
        # Separate features and target
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        
        # Handle missing values
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return True
    
    def train_models(self):
        """Train multiple models"""
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        with st.spinner('Training models...'):
            progress_bar = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                # Train
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                train_mae = mean_absolute_error(self.y_train, y_pred_train)
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                           cv=5, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                progress_bar.progress((i + 1) / len(models))
        
        self.models = results
        
        # Find best model
        self.best_model_name = max(results, key=lambda x: results[x]['test_r2'])
        self.best_model = results[self.best_model_name]['model']
        
        return results
    
    def tune_random_forest(self):
        """Tune Random Forest hyperparameters"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=0
        )
        
        with st.spinner('Tuning Random Forest...'):
            grid_search.fit(self.X_train, self.y_train)
        
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_rf.predict(self.X_test)
        test_r2 = r2_score(self.y_test, y_pred)
        
        return best_rf, grid_search.best_params_, test_r2
    
    def predict(self, features):
        """Make prediction"""
        if self.best_model is None:
            return None
        
        features_scaled = self.scaler.transform([features])
        prediction = self.best_model.predict(features_scaled)[0]
        
        # Get confidence interval for Random Forest
        confidence = None
        if hasattr(self.best_model, 'estimators_'):
            all_preds = np.array([tree.predict(features_scaled)[0] 
                                 for tree in self.best_model.estimators_])
            confidence = np.std(all_preds) * 1.96
        
        return prediction, confidence

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = HousePricePredictor()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Header
st.markdown('<div class="main-header"><h1>🏠 House Price Predictor</h1><p>Advanced Machine Learning Web App</p></div>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/real-estate.png", width=100)
    st.title("Navigation")
    
    option = st.radio(
        "Choose a section:",
        ["📊 Data Upload", "🔍 Data Exploration", "⚙️ Preprocessing", 
         "🤖 Model Training", "📈 Model Evaluation", "🎯 Hyperparameter Tuning",
         "🔮 Make Predictions", "💾 Save/Load Model"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "This app uses machine learning to predict house prices based on various features. "
        "Upload your dataset and follow the steps to train a model and make predictions."
    )

# Main content area
if option == "📊 Data Upload":
    st.header("📤 Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Upload your house price dataset in CSV format"
        )
        
        if uploaded_file is not None:
            if st.session_state.predictor.load_data(uploaded_file):
                st.session_state.data_loaded = True
                st.success("✅ Dataset loaded successfully!")
                
                # Show dataset preview
                st.subheader("Dataset Preview")
                st.dataframe(st.session_state.predictor.df.head(10))
                
                # Dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", st.session_state.predictor.df.shape[0])
                with col2:
                    st.metric("Columns", st.session_state.predictor.df.shape[1])
                with col3:
                    missing = st.session_state.predictor.df.isnull().sum().sum()
                    st.metric("Missing Values", missing)
    
    with col2:
        st.markdown("### Sample Data")
        st.markdown("""
        Your dataset should contain:
        - Numeric features (sqft, bedrooms, etc.)
        - Categorical features (location, type, etc.)
        - A target column (price)
        
        **Common columns:**
        - Price/SalePrice
        - Area/SqFt
        - Bedrooms
        - Bathrooms
        - Location
        - Year Built
        """)

elif option == "🔍 Data Exploration":
    st.header("🔎 Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload a dataset first!")
    else:
        predictor = st.session_state.predictor
        
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Statistics", "Missing Values", "Visualizations"])
        
        with tab1:
            st.subheader("Dataset Overview")
            st.write("**First 10 rows:**")
            st.dataframe(predictor.df.head(10))
            
            st.write("**Last 10 rows:**")
            st.dataframe(predictor.df.tail(10))
            
            st.write("**Data Types:**")
            dtypes_df = pd.DataFrame(predictor.df.dtypes, columns=['Data Type'])
            st.dataframe(dtypes_df)
        
        with tab2:
            st.subheader("Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Columns:**")
                numeric_cols = predictor.df.select_dtypes(include=[np.number]).columns
                st.dataframe(predictor.df[numeric_cols].describe())
            
            with col2:
                st.write("**Categorical Columns:**")
                categorical_cols = predictor.df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    for col in categorical_cols[:3]:  # Show first 3
                        st.write(f"**{col}**")
                        st.write(predictor.df[col].value_counts().head())
                else:
                    st.write("No categorical columns found")
        
        with tab3:
            st.subheader("Missing Values Analysis")
            
            missing = predictor.df.isnull().sum()
            missing_percent = (missing / len(predictor.df)) * 100
            
            missing_df = pd.DataFrame({
                'Column': predictor.df.columns,
                'Missing Values': missing.values,
                'Percentage': missing_percent.values
            }).sort_values('Missing Values', ascending=False)
            
            # Filter columns with missing values
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if len(missing_df) > 0:
                st.dataframe(missing_df)
                
                # Visualization
                fig = px.bar(missing_df, x='Column', y='Percentage', 
                            title='Missing Values by Column (%)',
                            color='Percentage',
                            color_continuous_scale='reds')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found in the dataset!")
        
        with tab4:
            st.subheader("Data Visualizations")
            
            viz_type = st.selectbox(
                "Select visualization type:",
                ["Distribution of Target", "Correlation Matrix", "Box Plots", 
                 "Pair Plot", "Categorical Analysis"]
            )
            
            if viz_type == "Distribution of Target":
                target = predictor.identify_price_column()
                fig = px.histogram(predictor.df, x=target, nbins=50,
                                  title=f'Distribution of {target}',
                                  marginal='box')
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Correlation Matrix":
                numeric_cols = predictor.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = predictor.df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, 
                                   title='Correlation Matrix',
                                   color_continuous_scale='RdBu',
                                   aspect='auto')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plots":
                numeric_cols = predictor.df.select_dtypes(include=[np.number]).columns
                selected_cols = st.multiselect("Select columns:", numeric_cols, 
                                              default=numeric_cols[:3])
                if selected_cols:
                    fig = go.Figure()
                    for col in selected_cols:
                        fig.add_trace(go.Box(y=predictor.df[col], name=col))
                    st.plotly_chart(fig, use_container_width=True)

elif option == "⚙️ Preprocessing":
    st.header("⚙️ Data Preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload a dataset first!")
    else:
        predictor = st.session_state.predictor
        
        st.subheader("Preprocessing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target column selection
            target_options = predictor.df.columns.tolist()
            default_target = predictor.identify_price_column()
            target_col = st.selectbox(
                "Select target column (price):",
                target_options,
                index=target_options.index(default_target) if default_target in target_options else 0
            )
        
        with col2:
            test_size = st.slider("Test set size (%):", 10, 40, 20) / 100
            random_state = st.number_input("Random state:", value=42, min_value=0, max_value=100)
        
        if st.button("🚀 Preprocess Data", type="primary"):
            if predictor.preprocess_data(target_col, test_size, random_state):
                st.success("✅ Data preprocessing completed!")
                
                # Show preprocessing info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Training samples", len(predictor.X_train))
                with col2:
                    st.metric("Test samples", len(predictor.X_test))
                with col3:
                    st.metric("Features", len(predictor.feature_names))
                
                # Show feature names
                st.subheader("Features after preprocessing:")
                st.write(predictor.feature_names)

elif option == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not hasattr(st.session_state.predictor, 'X_train') or st.session_state.predictor.X_train is None:
        st.warning("Please preprocess the data first!")
    else:
        predictor = st.session_state.predictor
        
        if st.button("🚀 Train All Models", type="primary"):
            results = predictor.train_models()
            st.session_state.models_trained = True
            st.success("✅ Model training completed!")
            
            # Display results
            st.subheader("Model Performance Summary")
            
            # Create comparison dataframe
            comparison = []
            for name, metrics in results.items():
                comparison.append({
                    'Model': name,
                    'Train R²': f"{metrics['train_r2']:.4f}",
                    'Test R²': f"{metrics['test_r2']:.4f}",
                    'Train RMSE': f"{metrics['train_rmse']:.4f}",
                    'Test RMSE': f"{metrics['test_rmse']:.4f}",
                    'CV R²': f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(comparison))
            
            # Highlight best model
            st.success(f"🏆 Best Model: **{predictor.best_model_name}** with Test R² = {results[predictor.best_model_name]['test_r2']:.4f}")
            
            # Visualization
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=('R² Score Comparison', 'RMSE Comparison'))
            
            models = list(results.keys())
            test_r2 = [results[m]['test_r2'] for m in models]
            test_rmse = [results[m]['test_rmse'] for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=test_r2, name='R² Score', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=test_rmse, name='RMSE', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

elif option == "📈 Model Evaluation":
    st.header("📈 Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
    else:
        predictor = st.session_state.predictor
        
        # Model selection
        selected_model = st.selectbox(
            "Select model to evaluate:",
            list(predictor.models.keys())
        )
        
        model_data = predictor.models[selected_model]
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Train R²", f"{model_data['train_r2']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test R²", f"{model_data['test_r2']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Train RMSE", f"{model_data['train_rmse']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Test RMSE", f"{model_data['test_rmse']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Predictions vs Actual
        model = model_data['model']
        y_pred = model.predict(predictor.X_test)
        
        fig = px.scatter(x=predictor.y_test, y=y_pred, 
                        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                        title=f'{selected_model}: Predictions vs Actual')
        
        # Add perfect prediction line
        min_val = min(predictor.y_test.min(), y_pred.min())
        max_val = max(predictor.y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='Perfect Prediction',
                                line=dict(color='red', dash='dash')))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residuals
        residuals = predictor.y_test - y_pred
        
        fig = px.scatter(x=y_pred, y=residuals,
                        labels={'x': 'Predicted Values', 'y': 'Residuals'},
                        title='Residual Plot')
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

elif option == "🎯 Hyperparameter Tuning":
    st.header("🎯 Hyperparameter Tuning")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
    else:
        predictor = st.session_state.predictor
        
        st.subheader("Tune Random Forest")
        
        if st.button("Start Tuning", type="primary"):
            best_model, best_params, test_r2 = predictor.tune_random_forest()
            
            st.success("✅ Tuning completed!")
            
            st.write("**Best Parameters:**")
            st.json(best_params)
            
            st.metric("Test R² with tuned model", f"{test_r2:.4f}")
            
            # Compare with original
            original_r2 = predictor.models['Random Forest']['test_r2']
            improvement = (test_r2 - original_r2) * 100
            
            if improvement > 0:
                st.success(f"✅ Improvement: +{improvement:.2f}%")
            else:
                st.warning(f"ℹ️ Change: {improvement:.2f}%")

elif option == "🔮 Make Predictions":
    st.header("🔮 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first!")
    else:
        predictor = st.session_state.predictor
        
        st.subheader("Enter Feature Values")
        
        # Create input fields for each feature
        cols = st.columns(3)
        features = []
        
        for i, feature in enumerate(predictor.feature_names):
            with cols[i % 3]:
                value = st.number_input(
                    f"{feature}:",
                    value=0.0,
                    format="%.2f",
                    key=f"input_{feature}"
                )
                features.append(value)
        
        if st.button("🔮 Predict Price", type="primary"):
            prediction, confidence = predictor.predict(features)
            
            if prediction is not None:
                col1, col2, col3 = st.columns(3)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Predicted Price", f"${prediction:,.2f}")
                    
                    if confidence:
                        st.write(f"95% Confidence: ±${confidence:,.2f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Batch prediction
        st.markdown("---")
        st.subheader("Batch Prediction")
        
        batch_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
        
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            st.write("Preview:", batch_df.head())
            
            if st.button("Predict Batch"):
                # Prepare features
                batch_features = batch_df[predictor.feature_names].values
                batch_scaled = predictor.scaler.transform(batch_features)
                predictions = predictor.best_model.predict(batch_scaled)
                
                # Add predictions to dataframe
                batch_df['Predicted_Price'] = predictions
                
                st.write("Results:")
                st.dataframe(batch_df)
                
                # Download button
                csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

elif option == "💾 Save/Load Model":
    st.header("💾 Save/Load Model")
    
    if not st.session_state.models_trained:
        st.warning("Please train models first to save!")
    else:
        predictor = st.session_state.predictor
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Save Model")
            
            model_name = st.text_input("Model filename:", "house_price_model.pkl")
            
            if st.button("💾 Save Model"):
                model_data = {
                    'model': predictor.best_model,
                    'scaler': predictor.scaler,
                    'feature_names': predictor.feature_names,
                    'label_encoders': predictor.label_encoders,
                    'target_col': predictor.target_col,
                    'model_name': predictor.best_model_name
                }
                
                joblib.dump(model_data, model_name)
                
                with open(model_name, 'rb') as f:
                    st.download_button(
                        label="📥 Download Model",
                        data=f,
                        file_name=model_name,
                        mime="application/octet-stream"
                    )
                
                st.success(f"✅ Model saved as {model_name}")
        
        with col2:
            st.subheader("Load Model")
            
            uploaded_model = st.file_uploader("Upload model file", type=['pkl'])
            
            if uploaded_model is not None:
                try:
                    model_data = joblib.load(uploaded_model)
                    predictor.best_model = model_data['model']
                    predictor.scaler = model_data['scaler']
                    predictor.feature_names = model_data['feature_names']
                    predictor.label_encoders = model_data.get('label_encoders', {})
                    predictor.target_col = model_data.get('target_col')
                    predictor.best_model_name = model_data.get('model_name', 'Unknown')
                    
                    st.session_state.models_trained = True
                    st.success(f"✅ Model loaded: {predictor.best_model_name}")
                    
                    st.write("**Features:**")
                    st.write(predictor.feature_names)
                    
                except Exception as e:
                    st.error(f"Error loading model: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        House Price Predictor v1.0 | Built with Streamlit and Scikit-learn
    </div>
    """,
    unsafe_allow_html=True
)