from django.shortcuts import render
from django.http import HttpResponse
from stock_values import ai_predictor
# Create your views here.

def get_prediction(request):
    stock = 'AAPL'
    prediction = ai_predictor.get_stock_value()
    ret_values = prediction
    return HttpResponse(ret_values)

def check(request):
    return HttpResponse('test')