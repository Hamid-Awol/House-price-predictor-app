# house_price_pro_gui.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import seaborn as sns
from datetime import datetime
import threading
warnings.filterwarnings('ignore')

class ProfessionalHousePriceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏠 Professional House Price Predictor")
        self.root.geometry("1200x700")
        self.root.minsize(1000, 600)
        
        # Set style and color scheme
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'white': '#ffffff'
        }
        
        # Configure style
        self.setup_styles()
        
        # Variables
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.target_col = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.current_results = None
        
        # Create GUI
        self.create_widgets()
        self.center_window()
        
    def setup_styles(self):
        """Configure ttk styles for modern look"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Primary.TButton', 
                       background=self.colors['accent'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=('Arial', 10, 'bold'))
        
        style.map('Primary.TButton',
                 background=[('active', '#2980b9')])
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       font=('Arial', 10, 'bold'))
        
        style.configure('Warning.TButton',
                       background=self.colors['warning'],
                       foreground='white',
                       font=('Arial', 10, 'bold'))
        
        style.configure('TLabel', 
                       font=('Arial', 10),
                       background=self.colors['white'])
        
        style.configure('Header.TLabel', 
                       font=('Arial', 16, 'bold'),
                       foreground=self.colors['primary'])
        
        style.configure('Title.TLabel',
                       font=('Arial', 24, 'bold'),
                       foreground=self.colors['primary'])
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def create_widgets(self):
        """Create all GUI widgets with modern layout"""
        
        # Main container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                                text="🏠 PROFESSIONAL HOUSE PRICE PREDICTOR",
                                style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Date/time
        self.time_label = ttk.Label(header_frame, 
                                    text=datetime.now().strftime("%Y-%m-%d %H:%M"),
                                    font=('Arial', 10))
        self.time_label.pack(side=tk.RIGHT)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_preprocessing_tab()
        self.create_modeling_tab()
        self.create_evaluation_tab()
        self.create_prediction_tab()
        
        # Status bar
        self.create_status_bar(main_container)
        
    def create_data_tab(self):
        """Create data loading and exploration tab"""
        data_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(data_frame, text="📊 Data")
        
        # File selection section
        file_section = ttk.LabelFrame(data_frame, text="Dataset Selection", padding="15")
        file_section.pack(fill=tk.X, pady=(0, 20))
        
        file_row = ttk.Frame(file_section)
        file_row.pack(fill=tk.X)
        
        ttk.Label(file_row, text="CSV File:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_path_var = tk.StringVar(value="No file selected")
        file_entry = ttk.Entry(file_row, textvariable=self.file_path_var, width=50, state='readonly')
        file_entry.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
        
        browse_btn = ttk.Button(file_row, text="📂 Browse", 
                               command=self.browse_file,
                               style='Primary.TButton',
                               width=15)
        browse_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        load_btn = ttk.Button(file_row, text="📥 Load Data",
                              command=self.load_data,
                              style='Success.TButton',
                              width=15)
        load_btn.pack(side=tk.LEFT)
        
        # Data info section
        info_section = ttk.LabelFrame(data_frame, text="Dataset Information", padding="15")
        info_section.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Create info display
        info_frame = ttk.Frame(info_section)
        info_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Metrics
        metrics_frame = ttk.Frame(info_frame)
        metrics_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.metrics_text = tk.Text(metrics_frame, height=8, width=40,
                                    font=('Consolas', 10),
                                    bg=self.colors['light'],
                                    relief=tk.FLAT,
                                    wrap=tk.WORD)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Preview
        preview_frame = ttk.Frame(info_frame)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        ttk.Label(preview_frame, text="Data Preview (first 10 rows):",
                 font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        
        # Treeview for data preview
        preview_container = ttk.Frame(preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True)
        
        vsb = ttk.Scrollbar(preview_container, orient="vertical")
        hsb = ttk.Scrollbar(preview_container, orient="horizontal")
        
        self.preview_tree = ttk.Treeview(preview_container,
                                        yscrollcommand=vsb.set,
                                        xscrollcommand=hsb.set,
                                        height=8)
        
        vsb.config(command=self.preview_tree.yview)
        hsb.config(command=self.preview_tree.xview)
        
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        preview_container.grid_rowconfigure(0, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)
        
    def create_preprocessing_tab(self):
        """Create preprocessing tab"""
        pre_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(pre_frame, text="⚙️ Preprocessing")
        
        # Target selection
        target_section = ttk.LabelFrame(pre_frame, text="Target Variable", padding="15")
        target_section.pack(fill=tk.X, pady=(0, 20))
        
        target_row = ttk.Frame(target_section)
        target_row.pack(fill=tk.X)
        
        ttk.Label(target_row, text="Select target column (price):",
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.target_combo = ttk.Combobox(target_row, state='readonly', width=40,
                                         font=('Arial', 10))
        self.target_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Split configuration
        split_section = ttk.LabelFrame(pre_frame, text="Data Split Configuration", padding="15")
        split_section.pack(fill=tk.X, pady=(0, 20))
        
        split_row = ttk.Frame(split_section)
        split_row.pack(fill=tk.X)
        
        ttk.Label(split_row, text="Test size (%):",
                 font=('Arial', 11)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_size_var = tk.DoubleVar(value=20.0)
        test_scale = ttk.Scale(split_row, from_=10, to=40,
                              variable=self.test_size_var,
                              orient=tk.HORIZONTAL,
                              length=200)
        test_scale.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_size_label = ttk.Label(split_row, text="20%",
                                        font=('Arial', 11, 'bold'))
        self.test_size_label.pack(side=tk.LEFT)
        
        test_scale.configure(command=lambda x: self.test_size_label.config(text=f"{int(float(x))}%"))
        
        ttk.Label(split_row, text="Random State:",
                 font=('Arial', 11)).pack(side=tk.LEFT, padx=(20, 10))
        
        self.random_state_var = tk.IntVar(value=42)
        random_spin = ttk.Spinbox(split_row, from_=0, to=999,
                                 textvariable=self.random_state_var,
                                 width=10,
                                 font=('Arial', 10))
        random_spin.pack(side=tk.LEFT)
        
        # Preprocess button
        preprocess_btn = ttk.Button(pre_frame, text="🚀 Run Preprocessing",
                                    command=self.preprocess_data,
                                    style='Primary.TButton',
                                    width=30)
        preprocess_btn.pack(pady=20)
        
        # Preprocessing results
        results_section = ttk.LabelFrame(pre_frame, text="Preprocessing Results", padding="15")
        results_section.pack(fill=tk.BOTH, expand=True)
        
        self.pre_results_text = tk.Text(results_section, height=15,
                                        font=('Consolas', 10),
                                        bg=self.colors['light'],
                                        relief=tk.FLAT,
                                        wrap=tk.WORD)
        self.pre_results_text.pack(fill=tk.BOTH, expand=True)
        
    def create_modeling_tab(self):
        """Create modeling tab"""
        model_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(model_frame, text="🤖 Modeling")
        
        # Model selection
        select_section = ttk.LabelFrame(model_frame, text="Model Selection", padding="15")
        select_section.pack(fill=tk.X, pady=(0, 20))
        
        # Model checkboxes
        self.model_vars = {}
        models = [
            ('Linear Regression', 'LinearRegression'),
            ('Ridge Regression', 'Ridge'),
            ('Lasso Regression', 'Lasso'),
            ('Random Forest', 'RandomForest'),
            ('Gradient Boosting', 'GradientBoosting')
        ]
        
        models_frame = ttk.Frame(select_section)
        models_frame.pack(fill=tk.X)
        
        for i, (display_name, model_key) in enumerate(models):
            var = tk.BooleanVar(value=True)
            self.model_vars[model_key] = var
            cb = ttk.Checkbutton(models_frame, text=display_name,
                                variable=var)
            cb.grid(row=i//3, column=i%3, padx=20, pady=5, sticky=tk.W)
        
        # Train button
        train_btn = ttk.Button(model_frame, text="🎯 Train Selected Models",
                              command=self.train_models,
                              style='Primary.TButton',
                              width=30)
        train_btn.pack(pady=20)
        
        # Results section
        results_section = ttk.LabelFrame(model_frame, text="Training Results", padding="15")
        results_section.pack(fill=tk.BOTH, expand=True)
        
        # Treeview for results
        columns = ('Model', 'R² Score', 'RMSE', 'MAE', 'CV Score')
        self.results_tree = ttk.Treeview(results_section, columns=columns,
                                        show='headings', height=8)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=120, anchor='center')
        
        vsb = ttk.Scrollbar(results_section, orient="vertical",
                           command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=vsb.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Best model label
        self.best_model_label = ttk.Label(model_frame,
                                         text="Best Model: Not yet trained",
                                         font=('Arial', 12, 'bold'),
                                         foreground=self.colors['success'])
        self.best_model_label.pack(pady=10)
        
    def create_evaluation_tab(self):
        """Create evaluation tab"""
        eval_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(eval_frame, text="📈 Evaluation")
        
        # Model selection for evaluation
        select_frame = ttk.Frame(eval_frame)
        select_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(select_frame, text="Select Model:",
                 font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.eval_model_combo = ttk.Combobox(select_frame, state='readonly',
                                            width=30, font=('Arial', 10))
        self.eval_model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.eval_model_combo.bind('<<ComboboxSelected>>', self.update_evaluation)
        
        # Metrics display
        metrics_frame = ttk.LabelFrame(eval_frame, text="Performance Metrics", padding="15")
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.metrics_labels = {}
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        metrics = ['R² Score', 'RMSE', 'MAE', 'CV Mean', 'CV Std']
        positions = [(0,0), (0,1), (1,0), (1,1), (2,0)]
        
        for (metric, (row, col)) in zip(metrics, positions):
            frame = ttk.Frame(metrics_grid, relief=tk.RIDGE, padding=10)
            frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
            
            ttk.Label(frame, text=metric,
                     font=('Arial', 10)).pack()
            
            value_label = ttk.Label(frame, text="--",
                                   font=('Arial', 14, 'bold'),
                                   foreground=self.colors['accent'])
            value_label.pack()
            self.metrics_labels[metric] = value_label
        
        # Feature importance section
        importance_frame = ttk.LabelFrame(eval_frame, text="Feature Importance", padding="15")
        importance_frame.pack(fill=tk.BOTH, expand=True)
        
        self.importance_text = tk.Text(importance_frame, height=8,
                                       font=('Consolas', 10),
                                       bg=self.colors['light'],
                                       relief=tk.FLAT,
                                       wrap=tk.WORD)
        self.importance_text.pack(fill=tk.BOTH, expand=True)
        
    def create_prediction_tab(self):
        """Create prediction tab"""
        pred_frame = ttk.Frame(self.notebook, padding="20")
        self.notebook.add(pred_frame, text="🔮 Predictions")
        
        # Single prediction section
        single_frame = ttk.LabelFrame(pred_frame, text="Single Prediction", padding="15")
        single_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create scrollable frame for features
        canvas = tk.Canvas(single_frame, height=200)
        scrollbar = ttk.Scrollbar(single_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.feature_entries = {}
        self.features_container = scrollable_frame
        
        canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar.pack(side="right", fill="y")
        
        # Predict button
        predict_btn = ttk.Button(single_frame, text="🔮 Predict Price",
                                command=self.predict_single,
                                style='Primary.TButton',
                                width=20)
        predict_btn.pack(pady=10)
        
        # Batch prediction section
        batch_frame = ttk.LabelFrame(pred_frame, text="Batch Prediction", padding="15")
        batch_frame.pack(fill=tk.BOTH, expand=True)
        
        batch_row = ttk.Frame(batch_frame)
        batch_row.pack(fill=tk.X, pady=(0, 10))
        
        self.batch_file_label = ttk.Label(batch_row, text="No file selected",
                                         font=('Arial', 10))
        self.batch_file_label.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(batch_row, text="📂 Select CSV",
                  command=self.select_batch_file).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(batch_row, text="🚀 Predict Batch",
                  command=self.predict_batch,
                  style='Success.TButton').pack(side=tk.LEFT, padx=5)
        
        # Batch results
        self.batch_results_text = tk.Text(batch_frame, height=6,
                                         font=('Consolas', 10),
                                         bg=self.colors['light'],
                                         relief=tk.FLAT,
                                         wrap=tk.WORD)
        self.batch_results_text.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN, padding="2")
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text=" Ready",
                                      font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(status_frame, mode='indeterminate',
                                           length=100)
        
    def update_status(self, message, show_progress=False):
        """Update status bar"""
        self.status_label.config(text=f" {message}")
        if show_progress:
            self.progress_bar.pack(side=tk.RIGHT, padx=5)
            self.progress_bar.start()
        else:
            self.progress_bar.stop()
            self.progress_bar.pack_forget()
        self.root.update()
        
    def browse_file(self):
        """Browse for CSV file"""
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.file_path_var.set(filename)
            
    def load_data(self):
        """Load and display dataset"""
        filename = self.file_path_var.get()
        if filename == "No file selected":
            messagebox.showerror("Error", "Please select a file first!")
            return
        
        self.update_status("Loading data...", True)
        
        try:
            self.df = pd.read_csv(filename)
            
            # Update metrics display
            metrics_text = f"""
📊 DATASET STATISTICS
{'='*40}

Total Rows:      {self.df.shape[0]:,}
Total Columns:   {self.df.shape[1]}
Memory Usage:    {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Data Types:
  Numeric:       {len(self.df.select_dtypes(include=[np.number]).columns)}
  Categorical:   {len(self.df.select_dtypes(include=['object']).columns)}

Missing Values:  {self.df.isnull().sum().sum()}
Duplicates:      {self.df.duplicated().sum()}
"""
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(1.0, metrics_text)
            
            # Update preview
            self.update_preview()
            
            # Update target combo
            self.target_combo['values'] = list(self.df.columns)
            
            # Auto-detect price column
            price_keywords = ['price', 'sale_price', 'saleprice', 'target']
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in price_keywords):
                    self.target_combo.set(col)
                    break
            
            self.update_status("Data loaded successfully")
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            
        except Exception as e:
            self.update_status("Error loading data")
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            
    def update_preview(self):
        """Update data preview"""
        # Clear existing items
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
            
        # Configure columns
        self.preview_tree['columns'] = list(self.df.columns)
        self.preview_tree['show'] = 'headings'
        
        for col in self.df.columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100, anchor='center')
            
        # Add data
        for i, row in self.df.head(10).iterrows():
            values = [row[col] for col in self.df.columns]
            self.preview_tree.insert('', tk.END, values=values)
            
    def preprocess_data(self):
        """Preprocess the data"""
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        target = self.target_combo.get()
        if not target:
            messagebox.showerror("Error", "Please select target column!")
            return
            
        self.target_col = target
        test_size = self.test_size_var.get() / 100
        random_state = self.random_state_var.get()
        
        self.update_status("Preprocessing data...", True)
        self.pre_results_text.delete(1.0, tk.END)
        
        try:
            # Separate features and target
            X = self.df.drop(columns=[target])
            y = self.df[target]
            
            self.pre_results_text.insert(tk.END, "📊 PREPROCESSING STEPS\n")
            self.pre_results_text.insert(tk.END, "="*50 + "\n\n")
            
            # Handle missing values
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isnull().sum() > 0:
                    median_val = X[col].median()
                    X[col].fillna(median_val, inplace=True)
                    self.pre_results_text.insert(tk.END, 
                        f"✓ Filled {col} missing values with median: {median_val:.2f}\n")
            
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    mode_val = X[col].mode()[0]
                    X[col].fillna(mode_val, inplace=True)
                    self.pre_results_text.insert(tk.END,
                        f"✓ Filled {col} missing values with mode: {mode_val}\n")
            
            # Encode categorical
            self.pre_results_text.insert(tk.END, "\n🔤 ENCODING CATEGORICAL VARIABLES\n")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                self.pre_results_text.insert(tk.END,
                    f"✓ Encoded {col} ({len(le.classes_)} categories)\n")
            
            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            self.feature_names = X.columns.tolist()
            
            # Scale features
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            
            self.pre_results_text.insert(tk.END, "\n📈 DATA SPLIT\n")
            self.pre_results_text.insert(tk.END, f"✓ Training samples: {len(self.X_train)}\n")
            self.pre_results_text.insert(tk.END, f"✓ Test samples: {len(self.X_test)}\n")
            self.pre_results_text.insert(tk.END, f"✓ Features: {len(self.feature_names)}\n")
            
            self.update_status("Preprocessing completed")
            messagebox.showinfo("Success", "Data preprocessing completed!")
            
            # Switch to modeling tab
            self.notebook.select(2)
            
            # Update prediction tab with feature entries
            self.update_feature_entries()
            
        except Exception as e:
            self.update_status("Error in preprocessing")
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            
    def update_feature_entries(self):
        """Update feature entries in prediction tab"""
        # Clear existing entries
        for widget in self.features_container.winfo_children():
            widget.destroy()
            
        self.feature_entries = {}
        
        # Create entries for each feature
        for i, feature in enumerate(self.feature_names):
            frame = ttk.Frame(self.features_container)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=feature, width=20,
                     font=('Arial', 10)).pack(side=tk.LEFT, padx=5)
            
            entry = ttk.Entry(frame, width=20, font=('Arial', 10))
            entry.pack(side=tk.LEFT, padx=5)
            entry.insert(0, "0.0")
            
            self.feature_entries[feature] = entry
            
    def train_models(self):
        """Train selected models"""
        if self.X_train is None:
            messagebox.showerror("Error", "Please preprocess data first!")
            return
            
        self.update_status("Training models...", True)
        
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
            
        self.models = {}
        
        # Define models
        model_dict = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        display_names = {
            'LinearRegression': 'Linear Regression',
            'Ridge': 'Ridge Regression',
            'Lasso': 'Lasso Regression',
            'RandomForest': 'Random Forest',
            'GradientBoosting': 'Gradient Boosting'
        }
        
        # Train selected models
        for model_key, var in self.model_vars.items():
            if var.get() and model_key in model_dict:
                model = model_dict[model_key]
                display_name = display_names[model_key]
                
                try:
                    # Train
                    model.fit(self.X_train, self.y_train)
                    
                    # Predict
                    y_pred = model.predict(self.X_test)
                    
                    # Calculate metrics
                    r2 = r2_score(self.y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                    mae = mean_absolute_error(self.y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, self.X_train, self.y_train,
                                               cv=5, scoring='r2')
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    # Store model
                    self.models[display_name] = {
                        'model': model,
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std
                    }
                    
                    # Add to tree
                    self.results_tree.insert('', tk.END, values=(
                        display_name,
                        f"{r2:.4f}",
                        f"{rmse:.4f}",
                        f"{mae:.4f}",
                        f"{cv_mean:.4f} ± {cv_std:.4f}"
                    ))
                    
                except Exception as e:
                    print(f"Error training {display_name}: {str(e)}")
        
        # Find best model
        if self.models:
            self.best_model_name = max(self.models, key=lambda x: self.models[x]['r2'])
            self.best_model = self.models[self.best_model_name]['model']
            
            self.best_model_label.config(
                text=f"🏆 Best Model: {self.best_model_name} (R² = {self.models[self.best_model_name]['r2']:.4f})"
            )
            
            # Update evaluation combo
            self.eval_model_combo['values'] = list(self.models.keys())
            self.eval_model_combo.set(self.best_model_name)
            self.update_evaluation()
            
            self.update_status("Model training completed")
            messagebox.showinfo("Success", "Model training completed!")
            
            # Switch to evaluation tab
            self.notebook.select(3)
            
    def update_evaluation(self, event=None):
        """Update evaluation tab with selected model"""
        selected = self.eval_model_combo.get()
        if not selected or selected not in self.models:
            return
            
        model_data = self.models[selected]
        
        # Update metrics
        self.metrics_labels['R² Score'].config(text=f"{model_data['r2']:.4f}")
        self.metrics_labels['RMSE'].config(text=f"{model_data['rmse']:.4f}")
        self.metrics_labels['MAE'].config(text=f"{model_data['mae']:.4f}")
        self.metrics_labels['CV Mean'].config(text=f"{model_data['cv_mean']:.4f}")
        self.metrics_labels['CV Std'].config(text=f"{model_data['cv_std']:.4f}")
        
        # Update feature importance for tree-based models
        self.importance_text.delete(1.0, tk.END)
        
        if hasattr(model_data['model'], 'feature_importances_'):
            importances = model_data['model'].feature_importances_
            feat_imp = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            self.importance_text.insert(tk.END, "📊 FEATURE IMPORTANCE\n")
            self.importance_text.insert(tk.END, "="*50 + "\n\n")
            
            for i, row in feat_imp.head(10).iterrows():
                self.importance_text.insert(tk.END,
                    f"{i+1}. {row['Feature']:<30} {row['Importance']:.4f}\n")
        else:
            self.importance_text.insert(tk.END, "Feature importance not available for this model.")
            
    def predict_single(self):
        """Make single prediction"""
        if self.best_model is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
            
        try:
            # Collect feature values
            features = []
            for feature in self.feature_names:
                value = float(self.feature_entries[feature].get())
                features.append(value)
                
            # Scale and predict
            features_scaled = self.scaler.transform([features])
            prediction = self.best_model.predict(features_scaled)[0]
            
            # Show result in a nice dialog
            result_window = tk.Toplevel(self.root)
            result_window.title("Prediction Result")
            result_window.geometry("400x250")
            result_window.transient(self.root)
            result_window.grab_set()
            
            # Center the window
            result_window.update_idletasks()
            x = (result_window.winfo_screenwidth() // 2) - (400 // 2)
            y = (result_window.winfo_screenheight() // 2) - (250 // 2)
            result_window.geometry(f'400x250+{x}+{y}')
            
            # Content
            ttk.Label(result_window, text="🏠 PREDICTED HOUSE PRICE",
                     style='Header.TLabel').pack(pady=20)
            
            price_label = ttk.Label(result_window,
                                   text=f"${prediction:,.2f}",
                                   font=('Arial', 32, 'bold'),
                                   foreground=self.colors['success'])
            price_label.pack(pady=20)
            
            ttk.Label(result_window,
                     text=f"Using model: {self.best_model_name}",
                     font=('Arial', 10)).pack()
            
            ttk.Button(result_window, text="Close",
                      command=result_window.destroy,
                      style='Primary.TButton').pack(pady=20)
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numbers for all features!")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            
    def select_batch_file(self):
        """Select file for batch prediction"""
        filename = filedialog.askopenfilename(
            title="Select CSV for Batch Prediction",
            filetypes=[("CSV files", "*.csv")]
        )
        if filename:
            self.batch_file_label.config(text=filename)
            self.batch_filename = filename
            
    def predict_batch(self):
        """Make batch predictions"""
        if self.best_model is None:
            messagebox.showerror("Error", "Please train a model first!")
            return
            
        if not hasattr(self, 'batch_filename'):
            messagebox.showerror("Error", "Please select a batch file!")
            return
            
        try:
            self.update_status("Processing batch predictions...", True)
            
            # Load batch data
            batch_df = pd.read_csv(self.batch_filename)
            
            # Check features
            missing_features = set(self.feature_names) - set(batch_df.columns)
            if missing_features:
                messagebox.showerror("Error", 
                    f"Missing features in batch file: {missing_features}")
                return
                
            # Prepare features
            X_batch = batch_df[self.feature_names].copy()
            
            # Handle missing values
            X_batch = X_batch.fillna(X_batch.median())
            
            # Scale and predict
            X_batch_scaled = self.scaler.transform(X_batch)
            predictions = self.best_model.predict(X_batch_scaled)
            
            # Add predictions to dataframe
            batch_df['Predicted_Price'] = predictions
            
            # Display results
            self.batch_results_text.delete(1.0, tk.END)
            self.batch_results_text.insert(tk.END, "✅ BATCH PREDICTION RESULTS\n")
            self.batch_results_text.insert(tk.END, "="*50 + "\n\n")
            self.batch_results_text.insert(tk.END, f"Total predictions: {len(predictions)}\n")
            self.batch_results_text.insert(tk.END, f"Price range: ${predictions.min():,.2f} - ${predictions.max():,.2f}\n")
            self.batch_results_text.insert(tk.END, f"Average price: ${predictions.mean():,.2f}\n\n")
            
            # Show preview
            self.batch_results_text.insert(tk.END, "Preview (first 5 rows):\n")
            self.batch_results_text.insert(tk.END, batch_df.head().to_string())
            
            # Save option
            save = messagebox.askyesno("Save Results", 
                "Do you want to save the predictions to a CSV file?")
                
            if save:
                save_filename = filedialog.asksaveasfilename(
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")])
                if save_filename:
                    batch_df.to_csv(save_filename, index=False)
                    messagebox.showinfo("Success", f"Predictions saved to {save_filename}")
                    
            self.update_status("Batch prediction completed")
            
        except Exception as e:
            self.update_status("Error in batch prediction")
            messagebox.showerror("Error", f"Batch prediction failed: {str(e)}")

def main():
    root = tk.Tk()
    app = ProfessionalHousePriceGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()