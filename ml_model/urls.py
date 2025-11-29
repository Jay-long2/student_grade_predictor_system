from django.urls import path
from . import views

urlpatterns = [
    path('train/', views.train_model, name='train_model'),
    path('evaluate/', views.evaluate_model, name='evaluate_model'),
    path('download-data/', views.download_training_data, name='download_training_data'),
    path('status/', views.model_status, name='model_status'),
]