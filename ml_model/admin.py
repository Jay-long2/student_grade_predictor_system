from django.contrib import admin
from .models import MLModel, TrainingSession

class TrainingSessionInline(admin.TabularInline):
    model = TrainingSession
    extra = 0
    readonly_fields = ('training_data_size', 'test_data_size', 'accuracy', 'training_time', 'created_at')
    can_delete = False
    
    def has_add_permission(self, request, obj):
        return False

@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'version', 'accuracy', 'precision', 'recall', 'f1_score', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at', 'updated_at')
    search_fields = ('name', 'version')
    readonly_fields = ('created_at', 'updated_at')
    inlines = (TrainingSessionInline,)
    list_editable = ('is_active',)
    list_per_page = 25
    
    fieldsets = (
        ('Model Information', {
            'fields': ('name', 'version', 'is_active')
        }),
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at')
        }),
    )
    
    def get_queryset(self, request):
        return super().get_queryset(request).prefetch_related('trainingsession_set')

@admin.register(TrainingSession)
class TrainingSessionAdmin(admin.ModelAdmin):
    list_display = ('model', 'training_data_size', 'test_data_size', 'accuracy', 
                   'training_time', 'created_at', 'performance_status')
    list_filter = ('created_at', 'model')
    search_fields = ('model__name',)
    readonly_fields = ('created_at', 'training_data_size', 'test_data_size', 
                      'accuracy', 'training_time', 'model')
    list_per_page = 25
    
    def performance_status(self, obj):
        if obj.accuracy >= 0.9:
            return 'Excellent'
        elif obj.accuracy >= 0.8:
            return 'Good'
        elif obj.accuracy >= 0.7:
            return 'Fair'
        else:
            return 'Poor'
    performance_status.short_description = 'Performance'
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('model')
    
