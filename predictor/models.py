from django.db import models
from django.contrib.auth.models import User

class StudentPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    gpa = models.FloatField()
    completed_units = models.IntegerField()
    internship_completed = models.CharField(max_length=3, choices=[('Yes', 'Yes'), ('No', 'No')])
    participation = models.CharField(max_length=10, choices=[('Low', 'Low'), ('Medium', 'Medium'), ('High', 'High')])
    discipline_score = models.FloatField()
    assignment_score = models.FloatField()
    
    # Prediction results
    predicted_class = models.CharField(max_length=20)
    confidence = models.FloatField()
    probabilities = models.JSONField(default=dict)  # Store all class probabilities
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username} - {self.predicted_class} ({self.confidence:.2f}%)"

class BulkPrediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='bulk_predictions/')
    total_records = models.IntegerField(default=0)
    processed_records = models.IntegerField(default=0)
    results_file = models.FileField(upload_to='prediction_results/', null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return f"Bulk Prediction - {self.user.username} - {self.created_at}"

class PredictionResult(models.Model):
    bulk_prediction = models.ForeignKey(BulkPrediction, on_delete=models.CASCADE, related_name='results')
    student_id = models.CharField(max_length=50, blank=True, null=True)
    gpa = models.FloatField()
    completed_units = models.IntegerField()
    internship_completed = models.CharField(max_length=3)
    participation = models.CharField(max_length=10)
    discipline_score = models.FloatField()
    assignment_score = models.FloatField()
    
    predicted_class = models.CharField(max_length=20)
    confidence = models.FloatField()
    
    def __str__(self):
        return f"Result - {self.student_id or 'Unknown'} - {self.predicted_class}"