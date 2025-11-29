from django import forms
from .models import StudentPrediction, BulkPrediction

class SinglePredictionForm(forms.ModelForm):
    class Meta:
        model = StudentPrediction
        fields = [
            'gpa', 'completed_units', 'internship_completed', 
            'participation', 'discipline_score', 'assignment_score'
        ]
        widgets = {
            'gpa': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '0.01',
                'min': '0.0',
                'max': '4.0',
                'placeholder': 'Enter GPA (0.0 - 4.0)'
            }),
            'completed_units': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0',
                'placeholder': 'Enter completed units'
            }),
            'internship_completed': forms.Select(attrs={
                'class': 'form-control'
            }),
            'participation': forms.Select(attrs={
                'class': 'form-control'
            }),
            'discipline_score': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '1',
                'min': '0',
                'max': '100',
                'placeholder': 'Enter discipline score (0-100)'
            }),
            'assignment_score': forms.NumberInput(attrs={
                'class': 'form-control',
                'step': '1',
                'min': '0',
                'max': '100',
                'placeholder': 'Enter assignment score (0-100)'
            }),
        }
        labels = {
            'gpa': 'Grade Point Average',
            'completed_units': 'Completed Units',
            'internship_completed': 'Internship Completed',
            'participation': 'Participation Level',
            'discipline_score': 'Discipline Score',
            'assignment_score': 'Assignment Score',
        }

class CSVUploadForm(forms.ModelForm):
    class Meta:
        model = BulkPrediction
        fields = ['csv_file']
        widgets = {
            'csv_file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            })
        }
        labels = {
            'csv_file': 'CSV File'
        }