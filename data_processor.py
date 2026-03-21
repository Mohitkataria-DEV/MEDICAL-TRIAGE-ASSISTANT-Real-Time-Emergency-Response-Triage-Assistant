import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os

class DataProcessor:
    def __init__(self, model_path='data_processor.pkl', load_cached=True):
        """Initialize the data processor with caching support"""
        
        self.model_path = model_path
        
        # Try to load cached model if requested
        if load_cached and os.path.exists(self.model_path):
            print(f"Loading cached data processor from {self.model_path}...")
            self._load_from_cache()
        else:
            print("Initializing fresh data processor...")
            self._initialize_fresh()
            # Cache the model if path is provided
            if model_path or load_cached:
                self.save_model()
        
        print("Data Processor initialized successfully!")
    
    def _initialize_fresh(self):
        """Initialize data from scratch"""
        self.data = None
        self.load_data()
    
    def _load_from_cache(self):
        """Load the entire data processor from cache file"""
        try:
            with open(self.model_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Restore all attributes
            self.data = cached_data['data']
            
            print(f"Loaded {len(self.data) if self.data is not None else 0} records from cache!")
            
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Falling back to fresh initialization...")
            self._initialize_fresh()
    
    def save_model(self):
        """Save the data processor to disk"""
        try:
            # Check disk space before saving
            import shutil
            total, used, free = shutil.disk_usage(".")
            if free < 100 * 1024 * 1024:  # Less than 100MB free
                print(f"Warning: Low disk space ({free / (1024**3):.1f} GB free). Skipping save.")
                return
            
            cached_data = {
                'data': self.data
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(cached_data, f)
            
            print(f"Data processor saved successfully to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving data processor: {e}")
    
    def load_data(self, filepath='synthetic_medical_triage.csv'):
        """Load and prepare the dataset with flexible column mapping"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Loaded {len(self.data)} records from {filepath}")
            print(f"Columns found: {list(self.data.columns)}")
            
            # Map columns to expected names if needed
            column_mapping = {
                'SpO2': 'oxygen',
                'O2Sat': 'oxygen',
                'Oxygen': 'oxygen',
                'HeartRate': 'heart_rate',
                'HR': 'heart_rate',
                'SysBP': 'systolic_bp',
                'Systolic': 'systolic_bp',
                'Temp': 'temperature',
                'Pain': 'pain_level'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in self.data.columns and new_name not in self.data.columns:
                    self.data.rename(columns={old_name: new_name}, inplace=True)
            
            # Check if required columns exist, if not create them with default values
            required_columns = {
                'age': 45,
                'heart_rate': 80,
                'systolic_bp': 120,
                'oxygen': 96,
                'temperature': 37.0,
                'pain_level': 5,
                'triage_level': 1
            }
            
            for col, default_value in required_columns.items():
                if col not in self.data.columns:
                    print(f"Warning: '{col}' column not found. Creating with default values.")
                    self.data[col] = default_value
            
            # Clean data types
            numeric_columns = ['age', 'heart_rate', 'systolic_bp', 'oxygen', 'temperature', 'pain_level', 'triage_level']
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce').fillna(required_columns.get(col, 0))
            
            print(f"Data prepared successfully. Shape: {self.data.shape}")
            
        except FileNotFoundError:
            print(f"No data file found at {filepath}, creating sample data")
            self.create_sample_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if file doesn't exist"""
        np.random.seed(42)
        n_samples = 1000
        
        self.data = pd.DataFrame({
            'age': np.random.normal(45, 18, n_samples).clip(0, 95).astype(int),
            'heart_rate': np.random.normal(80, 15, n_samples).clip(40, 150).astype(int),
            'systolic_bp': np.random.normal(125, 20, n_samples).clip(70, 200).astype(int),
            'oxygen': np.random.normal(96, 3, n_samples).clip(85, 100).astype(int),
            'temperature': np.random.normal(37, 0.8, n_samples).clip(35, 41).round(1),
            'pain_level': np.random.randint(0, 11, n_samples),
            'triage_level': np.random.choice([0, 1, 2, 3], n_samples, p=[0.3, 0.4, 0.2, 0.1])
        })
        
        print(f"Created {len(self.data)} synthetic records")
    
    def get_patient_by_id(self, patient_id):
        """Get patient data by ID"""
        if self.data is not None and patient_id < len(self.data):
            return self.data.iloc[patient_id].to_dict()
        return None
    
    def get_statistics(self):
        """Get dataset statistics with safe column access"""
        if self.data is None:
            return {}
        
        # Safely get values with fallbacks
        stats = {
            'total_patients': len(self.data),
            'triage_distribution': self.data['triage_level'].value_counts().to_dict() if 'triage_level' in self.data else {},
            'avg_age': float(self.data['age'].mean()) if 'age' in self.data else 0,
            'age_range': [float(self.data['age'].min()), float(self.data['age'].max())] if 'age' in self.data else [0, 0],
            'avg_heart_rate': float(self.data['heart_rate'].mean()) if 'heart_rate' in self.data else 0,
            'avg_oxygen': float(self.data['oxygen'].mean()) if 'oxygen' in self.data else 0
        }
        
        return stats
    
    def get_sample_queries(self, n=5):
        """Generate sample patient queries for testing"""
        if self.data is None:
            return []
        
        samples = self.data.sample(min(n, len(self.data)))
        queries = []
        
        for _, row in samples.iterrows():
            query_parts = []
            if 'age' in row:
                query_parts.append(f"Patient age {row['age']:.0f}")
            if 'heart_rate' in row:
                query_parts.append(f"HR {row['heart_rate']:.0f}")
            if 'systolic_bp' in row:
                query_parts.append(f"BP {row['systolic_bp']:.0f}")
            if 'oxygen' in row:
                query_parts.append(f"O2 {row['oxygen']:.0f}%")
            if 'temperature' in row:
                query_parts.append(f"Temp {row['temperature']:.1f}°C")
            if 'pain_level' in row:
                query_parts.append(f"Pain {row['pain_level']}/10")
            
            query = ", ".join(query_parts)
            
            queries.append({
                'query': query,
                'actual_triage': int(row['triage_level']) if 'triage_level' in row else 0
            })
        
        return queries
    
    def export_to_csv(self, filepath='medical_triage_data.csv'):
        """Export current data to CSV file"""
        if self.data is not None:
            self.data.to_csv(filepath, index=False)
            print(f"Data exported to {filepath}")
            return True
        return False
    
    def add_patient(self, patient_data):
        """Add a new patient record"""
        if self.data is None:
            self.create_sample_data()
        
        self.data = pd.concat([self.data, pd.DataFrame([patient_data])], ignore_index=True)
        print(f"Added new patient. Total records: {len(self.data)}")
        
        # Auto-save after adding
        self.save_model()