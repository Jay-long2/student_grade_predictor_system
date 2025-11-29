from django.db import models
from django.contrib.auth.models import User

class MLModel(models.Model):
    name = models.CharField(max_length=100, default='Random Forest Classifier')
    version = models.CharField(max_length=20, default='1.0')
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.name} v{self.version}"

class TrainingSession(models.Model):
    model = models.ForeignKey(MLModel, on_delete=models.CASCADE)
    training_data_size = models.IntegerField()
    test_data_size = models.IntegerField()
    accuracy = models.FloatField()
    training_time = models.FloatField()  # in seconds
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Training {self.model.name} - {self.created_at}"