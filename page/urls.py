from django.urls import path
from . import views

# URL conf
urlpatterns = [
    path('stock/', views.stock_view, name='stock'),
]