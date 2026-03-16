# house_price.py

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
from datetime import datetime
import os
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

class HousePricePredictor:
    """
    Professional House Price Prediction Class
    Compatible with GUI, CLI, and Web interfaces
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the House Price Predictor
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the dataset file
        """
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.models_metrics = {}
        self.best_model = None
        self.best_model_name = None
        self.best_model_metrics = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_col = None
        self.numeric_cols = None
        self.categorical_cols = None
        self.preprocessing_log = []
        self.training_log = []
        
        # Model configurations
        self.available_models = {
            'Linear Regression': {
                'class': LinearRegression,
                'params': {},
                'description': 'Simple linear regression model'
            },
            'Ridge Regression': {
                'class': Ridge,
                'params': {'alpha': 1.0},
                'description': 'Linear regression with L2 regularization'
            },
            'Lasso Regression': {
                'class': Lasso,
                'params': {'alpha': 1.0},
                'description': 'Linear regression with L1 regularization'
            },
            'Random Forest': {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1},
                'description': 'Ensemble of decision trees'
            },
            'Gradient Boosting': {
                'class': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'random_state': 42},
                'description': 'Gradient boosting machine'
            }
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'Gradient Boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5]
            },
            'Ridge Regression': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            'Lasso Regression': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            }
        }
        
    def load_data(self, file_path=None):
        """
        Load the dataset from CSV file
        
        Parameters:
        -----------
        file_path : str, optional
            Path to the dataset file (overrides instance file_path)
            
        Returns:
        --------
        bool : True if successful, False otherwise
        dict : Information about the loaded dataset
        """
        if file_path:
            self.file_path = file_path
            
        if not self.file_path:
            return False, {"error": "No file path provided"}
            
        try:
            self.df = pd.read_csv(self.file_path)
            
            # Identify numeric and categorical columns
            self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            
            # Auto-identify target column
            self.target_col = self._identify_price_column()
            
            # Prepare dataset information
            info = {
                "filename": os.path.basename(self.file_path),
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "numeric_features": len(self.numeric_cols),
                "categorical_features": len(self.categorical_cols),
                "missing_values": self.df.isnull().sum().sum(),
                "duplicates": self.df.duplicated().sum(),
                "memory_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
                "target_column": self.target_col,
                "column_names": self.df.columns.tolist(),
                "data_sample": self.df.head(10).to_dict('records')
            }
            
            return True, info
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _identify_price_column(self):
        """
        Identify the price/target column in the dataset
        
        Returns:
        --------
        str : Name of the identified target column
        """
        price_keywords = ['price', 'sale_price', 'saleprice', 'sale price', 
                         'house price', 'property price', 'target', 'cost', 
                         'value', 'selling_price', 'list_price']
        
        for col in self.df.columns:
            col_lower = col.lower().strip().replace('_', ' ').replace('-', ' ')
            if any(keyword in col_lower for keyword in price_keywords):
                return col
        
        # If no price column found, assume last column is target
        return self.df.columns[-1]
    
    def get_data_summary(self):
        """
        Get comprehensive data summary
        
        Returns:
        --------
        dict : Dictionary containing data summary statistics
        """
        if self.df is None:
            return None
            
        summary = {
            "basic_info": {
                "rows": self.df.shape[0],
                "columns": self.df.shape[1],
                "memory_usage": f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "missing_percentage": (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        if self.numeric_cols:
            desc = self.df[self.numeric_cols].describe()
            summary["numeric_summary"] = desc.to_dict()
            
        # Categorical columns summary
        if self.categorical_cols:
            for col in self.categorical_cols[:5]:  # Limit to first 5 for performance
                summary["categorical_summary"][col] = {
                    "unique_values": self.df[col].nunique(),
                    "top_values": self.df[col].value_counts().head(5).to_dict()
                }
                
        return summary
    
    def preprocess_data(self, target_col=None, test_size=0.2, random_state=42, 
                        handle_missing=True, encode_categorical=True, scale_features=True):
        """
        Preprocess the data with various options
        
        Parameters:
        -----------
        target_col : str, optional
            Name of the target column
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
        handle_missing : bool
            Whether to handle missing values
        encode_categorical : bool
            Whether to encode categorical variables
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        bool : True if successful, False otherwise
        dict : Preprocessing results and statistics
        """
        if self.df is None:
            return False, {"error": "No data loaded"}
            
        # Clear previous preprocessing log
        self.preprocessing_log = []
        
        try:
            # Set target column
            if target_col:
                self.target_col = target_col
            elif not self.target_col:
                self.target_col = self._identify_price_column()
                
            # Separate features and target
            X = self.df.drop(columns=[self.target_col])
            y = self.df[self.target_col]
            
            self.preprocessing_log.append(f"Features shape: {X.shape}")
            self.preprocessing_log.append(f"Target shape: {y.shape}")
            
            # Handle missing values
            if handle_missing:
                X, missing_stats = self._handle_missing_values(X)
                self.preprocessing_log.extend(missing_stats)
            
            # Encode categorical variables
            if encode_categorical:
                X, encoding_stats = self._encode_categorical(X)
                self.preprocessing_log.extend(encoding_stats)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            self.preprocessing_log.append(f"Training set size: {len(self.X_train)}")
            self.preprocessing_log.append(f"Test set size: {len(self.X_test)}")
            
            # Scale features
            if scale_features:
                self.X_train = self.scaler.fit_transform(self.X_train)
                self.X_test = self.scaler.transform(self.X_test)
                self.preprocessing_log.append("Features scaled using StandardScaler")
            
            # Prepare results
            results = {
                "target_column": self.target_col,
                "features_count": len(self.feature_names),
                "feature_names": self.feature_names,
                "train_samples": len(self.X_train),
                "test_samples": len(self.X_test),
                "test_size": test_size,
                "random_state": random_state,
                "preprocessing_log": self.preprocessing_log
            }
            
            return True, results
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _handle_missing_values(self, X):
        """Handle missing values in features"""
        stats = []
        
        # Numeric columns: fill with median
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
                stats.append(f"Filled {col} missing with median: {median_val:.2f}")
        
        # Categorical columns: fill with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                mode_val = X[col].mode()[0]
                X[col].fillna(mode_val, inplace=True)
                stats.append(f"Filled {col} missing with mode: {mode_val}")
        
        return X, stats
    
    def _encode_categorical(self, X):
        """Encode categorical variables"""
        stats = []
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            stats.append(f"Encoded {col} ({len(le.classes_)} categories)")
        
        return X, stats
    
    def train_models(self, model_list=None, cv_folds=5):
        """
        Train selected models
        
        Parameters:
        -----------
        model_list : list, optional
            List of model names to train (None = train all)
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Training results for each model
        """
        if self.X_train is None:
            return {"error": "Please preprocess data first"}
            
        self.training_log = []
        self.models = {}
        self.models_metrics = {}
        
        # Determine which models to train
        if model_list is None:
            model_list = list(self.available_models.keys())
        
        results = {}
        
        for model_name in model_list:
            if model_name not in self.available_models:
                continue
                
            try:
                # Get model configuration
                model_config = self.available_models[model_name]
                model_class = model_config['class']
                model_params = model_config['params']
                
                # Create and train model
                model = model_class(**model_params)
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    self.y_train, y_pred_train,
                    self.y_test, y_pred_test
                )
                
                # Cross-validation
                cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                           cv=cv_folds, scoring='r2')
                metrics['cv_mean'] = cv_scores.mean()
                metrics['cv_std'] = cv_scores.std()
                
                # Store model and metrics
                self.models[model_name] = model
                self.models_metrics[model_name] = metrics
                
                # Prepare results
                results[model_name] = {
                    'metrics': metrics,
                    'cv_scores': cv_scores.tolist(),
                    'feature_importance': self._get_feature_importance(model)
                }
                
                self.training_log.append(f"✓ {model_name} trained successfully")
                
            except Exception as e:
                self.training_log.append(f"✗ Error training {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}
        
        # Find best model
        if self.models_metrics:
            self.best_model_name = max(
                self.models_metrics,
                key=lambda x: self.models_metrics[x]['test_r2']
            )
            self.best_model = self.models[self.best_model_name]
            self.best_model_metrics = self.models_metrics[self.best_model_name]
            
            results['best_model'] = {
                'name': self.best_model_name,
                'metrics': self.best_model_metrics
            }
        
        results['training_log'] = self.training_log
        return results
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculate regression metrics"""
        return {
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test)
        }
    
    def _get_feature_importance(self, model):
        """Get feature importance if available"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return dict(zip(self.feature_names, importances))
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            return dict(zip(self.feature_names, coefficients))
        return None
    
    def tune_hyperparameters(self, model_name='Random Forest', param_grid=None, cv_folds=5):
        """
        Perform hyperparameter tuning for a specific model
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        param_grid : dict, optional
            Custom parameter grid (uses default if None)
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Tuning results including best parameters and score
        """
        if self.X_train is None:
            return {"error": "Please preprocess data first"}
            
        if model_name not in self.available_models:
            return {"error": f"Model {model_name} not available"}
            
        try:
            # Get parameter grid
            if param_grid is None:
                if model_name in self.param_grids:
                    param_grid = self.param_grids[model_name]
                else:
                    return {"error": f"No default parameter grid for {model_name}"}
            
            # Get model class
            model_config = self.available_models[model_name]
            model_class = model_config['class']
            base_params = model_config['params']
            
            # Create base model
            base_model = model_class(**base_params)
            
            # Perform grid search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds,
                scoring='r2', n_jobs=-1, verbose=0
            )
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Evaluate on test set
            y_pred = best_model.predict(self.X_test)
            test_r2 = r2_score(self.y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            
            # Prepare results
            results = {
                'best_params': grid_search.best_params_,
                'best_cv_score': grid_search.best_score_,
                'test_r2': test_r2,
                'test_rmse': test_rmse,
                'cv_results': grid_search.cv_results_,
                'best_model': best_model
            }
            
            # Update best model if better
            if test_r2 > self.best_model_metrics.get('test_r2', -np.inf):
                self.best_model = best_model
                self.best_model_name = f"Tuned {model_name}"
                self.best_model_metrics = {
                    'test_r2': test_r2,
                    'test_rmse': test_rmse
                }
                results['is_new_best'] = True
            
            return results
            
        except Exception as e:
            return {"error": str(e)}
    
    def get_feature_importance_dataframe(self):
        """
        Get feature importance as a DataFrame
        
        Returns:
        --------
        pd.DataFrame : Feature importance rankings
        """
        if self.best_model is None:
            return None
            
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return feat_imp_df
        elif hasattr(self.best_model, 'coef_'):
            coefficients = self.best_model.coef_.flatten()
            feat_imp_df = pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': coefficients
            }).sort_values('coefficient', ascending=False)
            return feat_imp_df
        
        return None
    
    def predict(self, features):
        """
        Make prediction for a single instance
        
        Parameters:
        -----------
        features : list or array
            Feature values in the same order as feature_names
            
        Returns:
        --------
        float : Predicted price
        dict : Additional prediction info (confidence interval, etc.)
        """
        if self.best_model is None:
            return None, {"error": "No model trained"}
            
        try:
            # Convert to numpy array and reshape
            features = np.array(features).reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.best_model.predict(features_scaled)[0]
            
            # Calculate confidence interval for tree-based models
            confidence = None
            if hasattr(self.best_model, 'estimators_'):
                all_preds = np.array([
                    tree.predict(features_scaled)[0]
                    for tree in self.best_model.estimators_
                ])
                confidence = {
                    'std': np.std(all_preds),
                    'ci_95': 1.96 * np.std(all_preds),
                    'min': np.min(all_preds),
                    'max': np.max(all_preds)
                }
            
            return prediction, {"confidence": confidence}
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def predict_batch(self, X_batch):
        """
        Make predictions for multiple instances
        
        Parameters:
        -----------
        X_batch : pd.DataFrame or np.array
            Batch of features
            
        Returns:
        --------
        np.array : Array of predictions
        dict : Additional information
        """
        if self.best_model is None:
            return None, {"error": "No model trained"}
            
        try:
            # Convert to DataFrame if necessary
            if isinstance(X_batch, np.ndarray):
                X_batch = pd.DataFrame(X_batch, columns=self.feature_names)
            
            # Ensure correct columns
            X_batch = X_batch[self.feature_names]
            
            # Handle missing values
            X_batch = X_batch.fillna(X_batch.median())
            
            # Scale features
            X_batch_scaled = self.scaler.transform(X_batch)
            
            # Make predictions
            predictions = self.best_model.predict(X_batch_scaled)
            
            # Statistics
            stats = {
                'count': len(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'mean': np.mean(predictions),
                'std': np.std(predictions)
            }
            
            return predictions, stats
            
        except Exception as e:
            return None, {"error": str(e)}
    
    def save_model(self, filename=None):
        """
        Save the trained model and associated data
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save the model
            
        Returns:
        --------
        bool : True if successful, False otherwise
        str : Path to saved file
        """
        if self.best_model is None:
            return False, "No model to save"
            
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"house_price_model_{timestamp}.pkl"
        
        try:
            model_data = {
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'target_col': self.target_col,
                'model_name': self.best_model_name,
                'model_metrics': self.best_model_metrics,
                'timestamp': datetime.now().isoformat(),
                'version': '1.0'
            }
            
            joblib.dump(model_data, filename)
            return True, filename
            
        except Exception as e:
            return False, str(e)
    
    def load_model(self, filename):
        """
        Load a saved model
        
        Parameters:
        -----------
        filename : str
            Path to the saved model file
            
        Returns:
        --------
        bool : True if successful, False otherwise
        dict : Model information
        """
        try:
            model_data = joblib.load(filename)
            
            self.best_model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.label_encoders = model_data.get('label_encoders', {})
            self.target_col = model_data.get('target_col')
            self.best_model_name = model_data.get('model_name', 'Unknown')
            self.best_model_metrics = model_data.get('model_metrics', {})
            
            info = {
                'model_name': self.best_model_name,
                'features': self.feature_names,
                'target': self.target_col,
                'metrics': self.best_model_metrics,
                'timestamp': model_data.get('timestamp', 'Unknown'),
                'version': model_data.get('version', 'Unknown')
            }
            
            return True, info
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def get_model_comparison_dataframe(self):
        """
        Get model comparison as a DataFrame
        
        Returns:
        --------
        pd.DataFrame : Comparison of all trained models
        """
        if not self.models_metrics:
            return None
            
        comparison = []
        for name, metrics in self.models_metrics.items():
            comparison.append({
                'Model': name,
                'Train R²': f"{metrics['train_r2']:.4f}",
                'Test R²': f"{metrics['test_r2']:.4f}",
                'Train RMSE': f"{metrics['train_rmse']:.4f}",
                'Test RMSE': f"{metrics['test_rmse']:.4f}",
                'Train MAE': f"{metrics['train_mae']:.4f}",
                'Test MAE': f"{metrics['test_mae']:.4f}",
                'CV R²': f"{metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}"
            })
        
        return pd.DataFrame(comparison)
    
    def create_visualizations(self, save_path=None):
        """
        Create comprehensive visualizations
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save visualizations
            
        Returns:
        --------
        dict : Dictionary of created figures
        """
        if self.df is None:
            return None
            
        figures = {}
        
        # 1. Distribution of target
        if self.target_col:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.histplot(self.df[self.target_col], kde=True, bins=30, ax=ax1)
            ax1.set_title(f'Distribution of {self.target_col}')
            ax1.set_xlabel(self.target_col)
            ax1.set_ylabel('Frequency')
            figures['target_distribution'] = fig1
        
        # 2. Correlation matrix
        if len(self.numeric_cols) > 1:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            corr_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       square=True, ax=ax2)
            ax2.set_title('Correlation Matrix')
            figures['correlation_matrix'] = fig2
        
        # 3. Feature importance (if available)
        if self.best_model is not None:
            feat_imp_df = self.get_feature_importance_dataframe()
            if feat_imp_df is not None:
                fig3, ax3 = plt.subplots(figsize=(12, 6))
                top_features = feat_imp_df.head(10)
                col_name = 'importance' if 'importance' in feat_imp_df.columns else 'coefficient'
                sns.barplot(data=top_features, x=col_name, y='feature', 
                           palette='viridis', ax=ax3)
                ax3.set_title('Top 10 Feature Importances')
                ax3.set_xlabel('Importance')
                figures['feature_importance'] = fig3
        
        # Save figures if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(f"{save_path}/{name}.png", dpi=300, bbox_inches='tight')
        
        return figures

def main():
    """
    Main function for backward compatibility
    """
    import sys
    
    # Check if running in CLI mode with arguments
    if len(sys.argv) > 1:
        main_cli()
    else:
        # Interactive mode
        predictor = HousePricePredictor('House Price Prediction Dataset.csv')
        
        print("="*60)
        print("HOUSE PRICE PREDICTOR")
        print("="*60)
        
        # Load data
        success, info = predictor.load_data()
        if not success:
            print(f"Failed to load data: {info.get('error')}")
            return
        
        print(f"\n✅ Dataset loaded: {info['filename']}")
        print(f"   Rows: {info['rows']}, Columns: {info['columns']}")
        print(f"   Target column: {info['target_column']}")
        
        # Explore data
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        summary = predictor.get_data_summary()
        if summary:
            print(f"\nMissing values: {sum(summary['missing_values'].values())}")
            print(f"Memory usage: {summary['basic_info']['memory_usage']}")
        
        # Preprocess data
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        test_size = float(input("\nEnter test size (default 0.2): ") or "0.2")
        success, pre_results = predictor.preprocess_data(test_size=test_size)
        
        if not success:
            print(f"Failed to preprocess data: {pre_results.get('error')}")
            return
        
        print("\n".join(pre_results['preprocessing_log']))
        
        # Train models
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        results = predictor.train_models()
        
        # Display results
        comparison_df = predictor.get_model_comparison_dataframe()
        if comparison_df is not None:
            print("\n" + comparison_df.to_string(index=False))
        
        # Feature importance
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        feat_imp = predictor.get_feature_importance_dataframe()
        if feat_imp is not None:
            print("\nTop 10 Features:")
            print(feat_imp.head(10).to_string(index=False))
        
        # Save model
        save = input("\nSave model? (y/n): ").lower()
        if save == 'y':
            success, filename = predictor.save_model()
            if success:
                print(f"✅ Model saved to {filename}")
        
        print("\n" + "="*60)
        print("✅ HOUSE PRICE PREDICTION COMPLETED!")
        print("="*60)

def main_cli():
    """Command-line interface for the house price predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='House Price Prediction')
    parser.add_argument('file', help='Path to the dataset CSV file')
    parser.add_argument('--target', help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--models', nargs='+', help='Models to train')
    parser.add_argument('--tune', action='store_true', help='Perform hyperparameter tuning')
    parser.add_argument('--save', help='Save model to file')
    parser.add_argument('--predict', help='Make predictions on new data')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HousePricePredictor(args.file)
    
    # Load data
    success, info = predictor.load_data()
    if not success:
        print(f"Error loading data: {info.get('error')}")
        return
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Rows: {info['rows']}, Columns: {info['columns']}")
    print(f"   Target column: {info['target_column']}")
    
    # Preprocess data
    target = args.target if args.target else info['target_column']
    success, pre_results = predictor.preprocess_data(
        target_col=target,
        test_size=args.test_size
    )
    
    if not success:
        print(f"Error preprocessing data: {pre_results.get('error')}")
        return
    
    print(f"\n✅ Data preprocessed successfully!")
    print(f"   Training samples: {pre_results['train_samples']}")
    print(f"   Test samples: {pre_results['test_samples']}")
    
    # Train models
    print(f"\n{'='*50}")
    print("TRAINING MODELS")
    print('='*50)
    
    results = predictor.train_models(model_list=args.models)
    
    # Display results
    comparison_df = predictor.get_model_comparison_dataframe()
    if comparison_df is not None:
        print("\n" + comparison_df.to_string(index=False))
    
    if 'best_model' in results:
        best = results['best_model']
        print(f"\n🏆 Best Model: {best['name']}")
        print(f"   Test R²: {best['metrics']['test_r2']:.4f}")
        print(f"   Test RMSE: {best['metrics']['test_rmse']:.4f}")
    
    # Hyperparameter tuning
    if args.tune:
        print(f"\n{'='*50}")
        print("HYPERPARAMETER TUNING")
        print('='*50)
        
        tune_results = predictor.tune_hyperparameters()
        if 'error' not in tune_results:
            print(f"\nBest parameters: {tune_results['best_params']}")
            print(f"Best CV score: {tune_results['best_cv_score']:.4f}")
            print(f"Test R²: {tune_results['test_r2']:.4f}")
    
    # Save model
    if args.save:
        success, filename = predictor.save_model(args.save)
        if success:
            print(f"\n✅ Model saved to {filename}")
        else:
            print(f"\n❌ Error saving model: {filename}")
    
    # Make predictions
    if args.predict:
        try:
            new_data = pd.read_csv(args.predict)
            predictions, stats = predictor.predict_batch(new_data)
            
            if predictions is not None:
                print(f"\n{'='*50}")
                print("PREDICTIONS")
                print('='*50')
                print(f"Total predictions: {stats['count']}")
                print(f"Price range: ${stats['min']:,.2f} - ${stats['max']:,.2f}")
                print(f"Average price: ${stats['mean']:,.2f}")
                
                # Save predictions
                new_data['Predicted_Price'] = predictions
                output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                new_data.to_csv(output_file, index=False)
                print(f"\n✅ Predictions saved to {output_file}")
                
        except Exception as e:
            print(f"\n❌ Error making predictions: {e}")

if __name__ == "__main__":
    main()
