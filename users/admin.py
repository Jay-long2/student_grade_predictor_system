from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import Profile

class ProfileInline(admin.StackedInline):
    model = Profile
    can_delete = False
    verbose_name_plural = 'Profile'
    fk_name = 'user'
    fields = ('student_id', 'department', 'phone_number', 'profile_picture')
    readonly_fields = ('created_at', 'updated_at')

class CustomUserAdmin(UserAdmin):
    inlines = (ProfileInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'get_student_id', 'get_department', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_active', 'is_superuser', 'date_joined', 'last_login')
    search_fields = ('username', 'first_name', 'last_name', 'email', 'profile__student_id', 'profile__department')
    list_per_page = 25
    
    def get_student_id(self, obj):
        return obj.profile.student_id if hasattr(obj, 'profile') else 'N/A'
    get_student_id.short_description = 'Student ID'
    
    def get_department(self, obj):
        return obj.profile.department if hasattr(obj, 'profile') and obj.profile.department else 'N/A'
    get_department.short_description = 'Department'
    
    def get_inline_instances(self, request, obj=None):
        if not obj:
            return list()
        return super().get_inline_instances(request, obj)

@admin.register(Profile)
class ProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'student_id', 'department', 'phone_number', 'created_at')
    list_filter = ('department', 'created_at')
    search_fields = ('user__username', 'student_id', 'department', 'phone_number')
    readonly_fields = ('created_at', 'updated_at')
    list_per_page = 25

# Re-register UserAdmin
admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)