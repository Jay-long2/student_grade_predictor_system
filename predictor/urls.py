from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('predict/', views.predict_single, name='predict_single'),
    path('results/', views.prediction_results, name='prediction_results'),
    path('visualization/', views.visualization, name='visualization'),
    path('download-results/<int:bulk_id>/', views.download_bulk_results, name='download_bulk_results'),
    path('clear-predictions/', views.clear_predictions, name='clear_predictions'),
]