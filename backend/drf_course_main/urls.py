from django.contrib import admin
from django.urls import path, include
from backend.core import views as core_views
from rest_framework import routers

router = routers.DefaultRouter()
# Nếu bạn muốn register viewsets, vd:
# router.register(r'items', core_views.YourViewSet)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('ai/', include('backend.core.urls')),  # <-- sửa từ 'core.urls' thành 'backend.core.urls'
]

urlpatterns += router.urls
