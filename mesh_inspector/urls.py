from django.urls import path
from .views import inspect_mesh

urlpatterns = [
    path('inspect/', inspect_mesh),
]
