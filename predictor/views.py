from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.core.files.storage import FileSystemStorage
import pandas as pd
import json
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np

from .forms import SinglePredictionForm, CSVUploadForm
from .models import StudentPrediction, BulkPrediction, PredictionResult
from ml_model.ml_utils import GradePredictor

# Global predictor instance (same as in ml_model)
predictor = GradePredictor()

@login_required
def dashboard(request):
    """Main dashboard showing prediction history and stats"""
    # Load recent predictions
    recent_predictions = StudentPrediction.objects.filter(user=request.user).order_by('-created_at')[:5]
    
    # Get prediction stats
    total_predictions = StudentPrediction.objects.filter(user=request.user).count()
    bulk_predictions = BulkPrediction.objects.filter(user=request.user).count()
    
    # Check if model is trained
    model_loaded = predictor.load_model()
    
    context = {
        'recent_predictions': recent_predictions,
        'total_predictions': total_predictions,
        'bulk_predictions': bulk_predictions,
        'model_loaded': model_loaded,
    }
    return render(request, 'predictor/dashboard.html', context)

@login_required
def predict_single(request):
    """Single student prediction form"""
    if not predictor.load_model():
        messages.error(request, 'Please train the ML model first before making predictions.')
        return redirect('train_model')
    
    if request.method == 'POST':
        form = SinglePredictionForm(request.POST)
        if form.is_valid():
            try:
                # Prepare data for prediction
                input_data = {
                    'GPA': form.cleaned_data['gpa'],
                    'Completed_Units': form.cleaned_data['completed_units'],
                    'Internship_Completed': form.cleaned_data['internship_completed'],
                    'Participation': form.cleaned_data['participation'],
                    'Discipline_Score': form.cleaned_data['discipline_score'],
                    'Assignment_Score': form.cleaned_data['assignment_score'],
                }
                
                # Make prediction
                predicted_class, confidence, probabilities = predictor.predict(input_data)
                
                # Save prediction to database
                prediction = form.save(commit=False)
                prediction.user = request.user
                prediction.predicted_class = predicted_class[0]
                prediction.confidence = confidence[0] * 100  # Convert to percentage
                prediction.probabilities = {
                    predictor.label_encoder.inverse_transform([i])[0]: float(prob)
                    for i, prob in enumerate(probabilities[0])
                }
                prediction.save()
                
                # Generate feature comparison chart
                feature_chart = generate_single_prediction_chart(input_data, predicted_class[0])
                
                # Prepare context for results page
                context = {
                    'prediction': prediction,
                    'input_data': input_data,
                    'probabilities': prediction.probabilities,
                    'feature_chart': feature_chart,
                }
                
                messages.success(request, f'Prediction completed! Result: {predicted_class[0]}')
                return render(request, 'predictor/prediction_results.html', context)
                
            except Exception as e:
                messages.error(request, f'Error making prediction: {str(e)}')
    else:
        form = SinglePredictionForm()
    
    return render(request, 'predictor/predict_single.html', {'form': form})

def generate_single_prediction_chart(input_data, predicted_class):
    """Generate feature comparison chart for single prediction"""
    # Prepare data for chart
    features = ['GPA', 'Completed Units', 'Discipline Score', 'Assignment Score']
    values = [
        input_data['GPA'],
        input_data['Completed_Units'],
        input_data['Discipline_Score'],
        input_data['Assignment_Score']
    ]
    
    # Normalize values for better visualization
    normalized_values = [
        values[0] / 4.0 * 100,  # GPA out of 4.0
        values[1] / 200 * 100,  # Units out of 200
        values[2],              # Already 0-100
        values[3]               # Already 0-100
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, normalized_values, color=['#007bff', '#28a745', '#ffc107', '#dc3545'])
    plt.title(f'Student Feature Profile\nPredicted: {predicted_class}')
    plt.ylabel('Normalized Score (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for bar, value, orig_value in zip(bars, normalized_values, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{orig_value}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    chart_base64 = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    
    return chart_base64

@login_required
def upload_csv(request):
    """Bulk prediction via CSV upload"""
    if not predictor.load_model():
        messages.error(request, 'Please train the ML model first before making predictions.')
        return redirect('train_model')
    
    if request.method == 'POST':
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                csv_file = request.FILES['csv_file']
                
                # Debug: Print file info
                print(f"File name: {csv_file.name}")
                print(f"File size: {csv_file.size}")
                
                # Check if file is empty
                if csv_file.size == 0:
                    messages.error(request, 'The uploaded file is empty.')
                    return redirect('upload_csv')
                
                # Try to read CSV with different parameters
                try:
                    df = pd.read_csv(csv_file)
                except pd.errors.EmptyDataError:
                    messages.error(request, 'The CSV file is empty or has no columns.')
                    return redirect('upload_csv')
                except pd.errors.ParserError as e:
                    messages.error(request, f'Error parsing CSV file: {str(e)}')
                    return redirect('upload_csv')
                except UnicodeDecodeError:
                    # Try with different encoding
                    csv_file.seek(0)
                    try:
                        df = pd.read_csv(csv_file, encoding='latin-1')
                    except Exception as e:
                        messages.error(request, f'Encoding error: {str(e)}')
                        return redirect('upload_csv')
                
                print(f"DataFrame shape: {df.shape}")
                print(f"DataFrame columns: {df.columns.tolist()}")
                
                # Save bulk prediction record
                bulk_prediction = form.save(commit=False)
                bulk_prediction.user = request.user
                bulk_prediction.save()
                
                bulk_prediction.total_records = len(df)
                bulk_prediction.save()
                
                # Validate CSV structure
                required_columns = ['GPA', 'Completed_Units', 'Internship_Completed', 
                                  'Participation', 'Discipline_Score', 'Assignment_Score']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    messages.error(request, f'Missing columns in CSV: {", ".join(missing_columns)}. Found columns: {", ".join(df.columns)}')
                    bulk_prediction.delete()
                    return redirect('upload_csv')
                
                # Process predictions
                results = []
                for index, row in df.iterrows():
                    try:
                        # Handle potential NaN values
                        input_data = {
                            'GPA': float(row['GPA']) if pd.notna(row['GPA']) else 0.0,
                            'Completed_Units': int(row['Completed_Units']) if pd.notna(row['Completed_Units']) else 0,
                            'Internship_Completed': str(row['Internship_Completed']) if pd.notna(row['Internship_Completed']) else 'No',
                            'Participation': str(row['Participation']) if pd.notna(row['Participation']) else 'Medium',
                            'Discipline_Score': float(row['Discipline_Score']) if pd.notna(row['Discipline_Score']) else 0.0,
                            'Assignment_Score': float(row['Assignment_Score']) if pd.notna(row['Assignment_Score']) else 0.0,
                        }
                        
                        # Validate data ranges
                        input_data['GPA'] = max(0.0, min(4.0, input_data['GPA']))
                        input_data['Discipline_Score'] = max(0.0, min(100.0, input_data['Discipline_Score']))
                        input_data['Assignment_Score'] = max(0.0, min(100.0, input_data['Assignment_Score']))
                        
                        # Make prediction
                        predicted_class, confidence, probabilities = predictor.predict(input_data)
                        
                        # Create result record
                        result = PredictionResult(
                            bulk_prediction=bulk_prediction,
                            student_id=row.get('Student_ID', f'STU{index+1:03d}'),
                            gpa=input_data['GPA'],
                            completed_units=input_data['Completed_Units'],
                            internship_completed=input_data['Internship_Completed'],
                            participation=input_data['Participation'],
                            discipline_score=input_data['Discipline_Score'],
                            assignment_score=input_data['Assignment_Score'],
                            predicted_class=predicted_class[0],
                            confidence=confidence[0] * 100,
                        )
                        result.save()
                        results.append(result)
                        
                        bulk_prediction.processed_records = len(results)
                        bulk_prediction.save()
                        
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        continue
                
                # Mark as completed
                bulk_prediction.completed_at = datetime.now()
                bulk_prediction.save()
                
                messages.success(request, f'Successfully processed {len(results)} out of {len(df)} records.')
                return redirect('prediction_results')
                
            except Exception as e:
                messages.error(request, f'Error processing CSV file: {str(e)}')
                print(f"Detailed error: {str(e)}")
                if 'bulk_prediction' in locals():
                    bulk_prediction.delete()
    else:
        form = CSVUploadForm()
    
    return render(request, 'predictor/upload_csv.html', {'form': form})

@login_required
def prediction_results(request):
    """Display prediction results"""
    # Get latest single prediction
    latest_single = StudentPrediction.objects.filter(user=request.user).order_by('-created_at').first()
    
    # Get latest bulk prediction
    latest_bulk = BulkPrediction.objects.filter(user=request.user).order_by('-created_at').first()
    bulk_results = []
    if latest_bulk:
        bulk_results = PredictionResult.objects.filter(bulk_prediction=latest_bulk)[:20]  # Limit for display
    
    context = {
        'latest_single': latest_single,
        'latest_bulk': latest_bulk,
        'bulk_results': bulk_results,
    }
    
    return render(request, 'predictor/prediction_results.html', context)

@login_required
def visualization(request):
    """Data visualization for predictions with actual charts"""
    # Get user's prediction history
    predictions = StudentPrediction.objects.filter(user=request.user)
    
    if not predictions:
        messages.info(request, 'No prediction data available for visualization.')
        return render(request, 'predictor/visualization.html')
    
    # Prepare data for charts
    class_distribution = {}
    confidence_scores = []
    feature_data = {
        'GPA': [], 'Completed_Units': [], 'Discipline_Score': [], 
        'Assignment_Score': [], 'Class': []
    }
    
    for prediction in predictions:
        class_name = prediction.predicted_class
        class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
        confidence_scores.append(prediction.confidence)
        
        # Collect feature data for charts
        feature_data['GPA'].append(prediction.gpa)
        feature_data['Completed_Units'].append(prediction.completed_units)
        feature_data['Discipline_Score'].append(prediction.discipline_score)
        feature_data['Assignment_Score'].append(prediction.assignment_score)
        feature_data['Class'].append(prediction.predicted_class)
    
    # Generate actual charts using Matplotlib/Seaborn
    charts = generate_prediction_charts(feature_data, class_distribution, confidence_scores)
    
    context = {
        'class_distribution': class_distribution,
        'total_predictions': len(predictions),
        'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
        'predictions': predictions,
        'charts': charts,  # Add the generated charts
    }
    
    return render(request, 'predictor/visualization.html', context)

def generate_prediction_charts(feature_data, class_distribution, confidence_scores):
    """Generate actual Matplotlib/Seaborn charts and convert to Base64"""
    charts = {}
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Chart 1: Class Distribution Pie Chart
    plt.figure(figsize=(8, 6))
    if class_distribution:
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        colors = ['#28a745', '#17a2b8', '#ffc107', '#007bff', '#dc3545']
        plt.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors[:len(classes)])
        plt.title('Predicted Class Distribution')
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        charts['class_distribution'] = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close()
    
    # Chart 2: Feature Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    features = ['GPA', 'Completed Units', 'Discipline Score', 'Assignment Score']
    avg_values = [
        sum(feature_data['GPA']) / len(feature_data['GPA']),
        sum(feature_data['Completed_Units']) / len(feature_data['Completed_Units']),
        sum(feature_data['Discipline_Score']) / len(feature_data['Discipline_Score']),
        sum(feature_data['Assignment_Score']) / len(feature_data['Assignment_Score'])
    ]
    
    bars = plt.bar(features, avg_values, color=['#007bff', '#28a745', '#ffc107', '#dc3545'])
    plt.title('Average Feature Values Across Predictions')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['feature_comparison'] = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    
    # Chart 3: Confidence Distribution Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(confidence_scores, bins=10, alpha=0.7, color='#6f42c1', edgecolor='black')
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['confidence_distribution'] = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    
    # Chart 4: GPA vs Assignment Score Scatter Plot
    plt.figure(figsize=(8, 6))
    
    # Create color mapping for classes
    color_map = {'First Class': '#28a745', 'Second Upper': '#17a2b8', 
                 'Second Lower': '#ffc107', 'Pass': '#007bff', 'Fail': '#dc3545'}
    colors = [color_map.get(cls, '#6c757d') for cls in feature_data['Class']]
    
    plt.scatter(feature_data['GPA'], feature_data['Assignment_Score'], 
                c=colors, alpha=0.6, s=60)
    plt.xlabel('GPA')
    plt.ylabel('Assignment Score')
    plt.title('GPA vs Assignment Score by Predicted Class')
    plt.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=8, label=cls)
                      for cls, color in color_map.items() if cls in set(feature_data['Class'])]
    plt.legend(handles=legend_elements)
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    charts['gpa_vs_assignment'] = base64.b64encode(buffer.getvalue()).decode()
    buffer.close()
    plt.close()
    
    return charts

@login_required
def download_bulk_results(request, bulk_id):
    """Download bulk prediction results as CSV"""
    try:
        bulk_prediction = BulkPrediction.objects.get(id=bulk_id, user=request.user)
        results = PredictionResult.objects.filter(bulk_prediction=bulk_prediction)
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename="prediction_results_{bulk_id}.csv"'
        
        writer = csv.writer(response)
        writer.writerow([
            'Student_ID', 'GPA', 'Completed_Units', 'Internship_Completed',
            'Participation', 'Discipline_Score', 'Assignment_Score',
            'Predicted_Class', 'Confidence'
        ])
        
        for result in results:
            writer.writerow([
                result.student_id,
                result.gpa,
                result.completed_units,
                result.internship_completed,
                result.participation,
                result.discipline_score,
                result.assignment_score,
                result.predicted_class,
                f"{result.confidence:.2f}%"
            ])
        
        return response
        
    except BulkPrediction.DoesNotExist:
        messages.error(request, 'Bulk prediction not found.')
        return redirect('prediction_results')

@login_required
def clear_predictions(request):
    """Clear user's prediction history"""
    if request.method == 'POST':
        StudentPrediction.objects.filter(user=request.user).delete()
        BulkPrediction.objects.filter(user=request.user).delete()
        messages.success(request, 'All prediction history has been cleared.')
    
    return redirect('dashboard')