from django.urls import path
from . import views

# URL conf
urlpatterns = [
    path('get_prediction/', views.get_prediction),
    path('check/', view=views.check)

]