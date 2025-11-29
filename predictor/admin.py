from django.contrib import admin
from .models import StudentPrediction, BulkPrediction, PredictionResult

class PredictionResultInline(admin.TabularInline):
    model = PredictionResult
    extra = 0
    readonly_fields = ('student_id', 'gpa', 'completed_units', 'internship_completed', 
                      'participation', 'discipline_score', 'assignment_score', 
                      'predicted_class', 'confidence')
    can_delete = False
    
    def has_add_permission(self, request, obj):
        return False

@admin.register(StudentPrediction)
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'gpa', 'completed_units', 'internship_completed', 
                   'participation', 'predicted_class', 'confidence', 'created_at')
    list_filter = ('predicted_class', 'internship_completed', 'participation', 'created_at', 'user')
    search_fields = ('user__username', 'predicted_class')
    readonly_fields = ('created_at', 'probabilities')
    fieldsets = (
        ('User Information', {
            'fields': ('user', 'created_at')
        }),
        ('Academic Features', {
            'fields': ('gpa', 'completed_units', 'internship_completed', 
                      'participation', 'discipline_score', 'assignment_score')
        }),
        ('Prediction Results', {
            'fields': ('predicted_class', 'confidence', 'probabilities')
        }),
    )
    list_per_page = 25
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('user')

@admin.register(BulkPrediction)
class BulkPredictionAdmin(admin.ModelAdmin):
    list_display = ('user', 'total_records', 'processed_records', 'created_at', 'completed_at', 'status')
    list_filter = ('created_at', 'completed_at', 'user')
    search_fields = ('user__username',)
    readonly_fields = ('created_at', 'completed_at', 'total_records', 'processed_records')
    inlines = (PredictionResultInline,)
    list_per_page = 25
    
    def status(self, obj):
        if obj.completed_at:
            return 'Completed'
        elif obj.processed_records > 0:
            return f'Processing ({obj.processed_records}/{obj.total_records})'
        else:
            return 'Pending'
    status.short_description = 'Status'

@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'bulk_prediction', 'gpa', 'completed_units', 
                   'internship_completed', 'predicted_class', 'confidence')
    list_filter = ('predicted_class', 'internship_completed', 'participation', 'bulk_prediction')
    search_fields = ('student_id', 'bulk_prediction__user__username', 'predicted_class')
    readonly_fields = ('bulk_prediction', 'student_id', 'gpa', 'completed_units', 
                      'internship_completed', 'participation', 'discipline_score', 
                      'assignment_score', 'predicted_class', 'confidence')
    list_per_page = 25
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('bulk_prediction', 'bulk_prediction__user')
    
    def has_add_permission(self, request):
        return False
    
    def has_delete_permission(self, request, obj=None):
        return False