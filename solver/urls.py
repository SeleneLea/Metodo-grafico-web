from django.urls import path 
from . import views

urlpatterns = [
    path('', views.solver_view, name='solver'),
]
