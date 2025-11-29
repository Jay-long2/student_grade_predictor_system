import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from django.conf import settings
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import time

class GradePredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_path = os.path.join(settings.BASE_DIR, 'ml_model', 'trained_model.pkl')
        
    def generate_synthetic_data(self, num_samples=1000):
        """Generate realistic synthetic student data for training"""
        np.random.seed(42)
        
        # Generate realistic distributions
        gpa = np.random.normal(2.8, 0.8, num_samples)
        gpa = np.clip(gpa, 1.0, 4.0)  # GPA between 1.0 and 4.0
        
        completed_units = np.random.randint(30, 200, num_samples)
        
        internship_completed = np.random.choice(['Yes', 'No'], num_samples, p=[0.6, 0.4])
        
        participation = np.random.choice(['Low', 'Medium', 'High'], num_samples, p=[0.2, 0.5, 0.3])
        
        discipline_score = np.random.normal(75, 15, num_samples)
        discipline_score = np.clip(discipline_score, 0, 100)
        
        assignment_score = np.random.normal(70, 20, num_samples)
        assignment_score = np.clip(assignment_score, 0, 100)
        
        # Create target variable based on logical rules
        classes = []
        for i in range(num_samples):
            score = (
                gpa[i] * 0.35 +
                (completed_units[i] / 200) * 0.15 +
                (1 if internship_completed[i] == 'Yes' else 0) * 0.1 +
                ({'Low': 0, 'Medium': 0.5, 'High': 1}[participation[i]]) * 0.1 +
                (discipline_score[i] / 100) * 0.15 +
                (assignment_score[i] / 100) * 0.15
            )
            
            if score > 0.75:
                classes.append('First Class')
            elif score > 0.60:
                classes.append('Second Upper')
            elif score > 0.45:
                classes.append('Second Lower')
            elif score > 0.30:
                classes.append('Pass')
            else:
                classes.append('Fail')
        
        # Create DataFrame
        data = {
            'GPA': gpa,
            'Completed_Units': completed_units,
            'Internship_Completed': internship_completed,
            'Participation': participation,
            'Discipline_Score': discipline_score,
            'Assignment_Score': assignment_score,
            'Class': classes
        }
        
        df = pd.DataFrame(data)
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Create copy to avoid modifying original
        df_processed = df.copy()
        
        # Encode categorical variables
        df_processed['Internship_Completed'] = df_processed['Internship_Completed'].map({'Yes': 1, 'No': 0})
        df_processed['Participation'] = df_processed['Participation'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df_processed['Class'])
        
        # Features
        X = df_processed.drop('Class', axis=1)
        
        return X, y
    
    def train_model(self, df=None, test_size=0.2):
        """Train the Random Forest model"""
        start_time = time.time()
        
        # Generate data if not provided
        if df is None:
            df = self.generate_synthetic_data(1000)
        
        # Preprocess data
        X, y = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': accuracy,
            'training_time': training_time,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': dict(zip(X.columns, self.model.feature_importances_)),
            'data_size': len(df),
            'test_size': len(X_test)
        }
    
    def predict(self, input_data):
        """Make prediction on new data"""
        if not self.is_trained:
            self.load_model()
            if not self.is_trained:
                raise Exception("Model is not trained. Please train the model first.")
        
        # Preprocess input data
        if isinstance(input_data, dict):
            # Single prediction
            df = pd.DataFrame([input_data])
        else:
            # Multiple predictions
            df = input_data.copy()
        
        # Encode categorical variables
        df['Internship_Completed'] = df['Internship_Completed'].map({'Yes': 1, 'No': 0})
        df['Participation'] = df['Participation'].map({'Low': 0, 'Medium': 1, 'High': 2})
        
        # Scale features
        X_scaled = self.scaler.transform(df)
        
        # Make prediction
        predictions_encoded = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode predictions
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        # Get confidence scores
        confidence_scores = np.max(probabilities, axis=1)
        
        return predictions, confidence_scores, probabilities
    
    def save_model(self):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, self.model_path)
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        return False
    
    def generate_visualizations(self, metrics):
        """Generate visualization charts for the model"""
        # Feature Importance Chart
        plt.figure(figsize=(10, 6))
        features = list(metrics['feature_importance'].keys())
        importance = list(metrics['feature_importance'].values())
        
        plt.barh(features, importance)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest Feature Importance')
        plt.tight_layout()
        
        # Convert to base64 for HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        feature_importance_img = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        class_names = self.label_encoder.classes_
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, fmt='d', 
                   xticklabels=class_names,
                   yticklabels=class_names,
                   cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        confusion_matrix_img = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
        
        return {
            'feature_importance': feature_importance_img,
            'confusion_matrix': confusion_matrix_img
        }