from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
import json
from .ml_utils import GradePredictor
from .models import MLModel, TrainingSession

# Global predictor instance
predictor = GradePredictor()

@login_required
def train_model(request):
    if request.method == 'POST':
        try:
            # Train the model
            metrics = predictor.train_model()
            
            # Save training session to database
            ml_model, created = MLModel.objects.get_or_create(
                name='Random Forest Classifier',
                defaults={'version': '1.0'}
            )
            
            training_session = TrainingSession.objects.create(
                model=ml_model,
                training_data_size=metrics['data_size'],
                test_data_size=metrics['test_size'],
                accuracy=metrics['accuracy'],
                training_time=metrics['training_time']
            )
            
            # Generate visualizations
            visualizations = predictor.generate_visualizations(metrics)
            
            messages.success(request, f'Model trained successfully! Accuracy: {metrics["accuracy"]:.2%}')
            
            context = {
                'metrics': metrics,
                'visualizations': visualizations,
                'training_session': training_session,
            }
            return render(request, 'ml_model/train_model.html', context)
            
        except Exception as e:
            messages.error(request, f'Error training model: {str(e)}')
    
    # Load existing model info if available
    context = {}
    try:
        if predictor.load_model():
            ml_model = MLModel.objects.filter(name='Random Forest Classifier').first()
            latest_training = TrainingSession.objects.filter(model=ml_model).order_by('-created_at').first()
            context['model_loaded'] = True
            context['latest_training'] = latest_training
    except:
        context['model_loaded'] = False
    
    return render(request, 'ml_model/train_model.html', context)

@login_required
def evaluate_model(request):
    # Generate new test data for evaluation
    test_data = predictor.generate_synthetic_data(200)
    X_test, y_test = predictor.preprocess_data(test_data)
    
    if not predictor.is_trained:
        messages.warning(request, 'Please train the model first.')
        return redirect('train_model')
    
    # Make predictions on test data
    predictions, confidence_scores, probabilities = predictor.predict(test_data.drop('Class', axis=1))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    accuracy = accuracy_score(y_test, predictor.label_encoder.transform(predictions))
    
    context = {
        'accuracy': accuracy,
        'test_data_size': len(test_data),
        'predictions': list(zip(
            test_data['Class'].tolist(),
            predictions.tolist(),
            confidence_scores.tolist()
        ))[:10],  # Show first 10 examples
        'class_distribution': test_data['Class'].value_counts().to_dict(),
    }
    
    return render(request, 'ml_model/evaluate_model.html', context)

@login_required
def download_training_data(request):
    """Download synthetic training data as CSV"""
    import csv
    from django.http import HttpResponse
    
    # Generate data
    data = predictor.generate_synthetic_data(500)
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="student_training_data.csv"'
    
    writer = csv.writer(response)
    writer.writerow(data.columns)
    
    for index, row in data.iterrows():
        writer.writerow(row)
    
    return response

@login_required
def model_status(request):
    """API endpoint to check model status"""
    status = {
        'is_trained': predictor.is_trained,
        'model_loaded': predictor.load_model(),
    }
    
    if predictor.is_trained:
        ml_model = MLModel.objects.filter(name='Random Forest Classifier').first()
        if ml_model:
            latest_training = TrainingSession.objects.filter(model=ml_model).order_by('-created_at').first()
            status['latest_accuracy'] = latest_training.accuracy if latest_training else 0
            status['last_trained'] = latest_training.created_at.isoformat() if latest_training else None
    
    return JsonResponse(status)