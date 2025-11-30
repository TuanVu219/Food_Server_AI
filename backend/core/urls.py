from django.urls import path
from . import views
from .views import FoodRecogniseView, draw_view
urlpatterns = [
    path('recognise/', FoodRecogniseView.as_view(), name='food_recognise'),
    path('draw/', views.draw_view, name='draw'),

]
