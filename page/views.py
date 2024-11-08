from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from stock_values import ai_predictor
from form import stockForm
from django.views.decorators.csrf import ensure_csrf_cookie
import time

RESULT = 0
STOCK = ""

# Create your views here.
@ensure_csrf_cookie
def stock_view(request):
    stock = None  # To store the submitted stock input
    form = stockForm(request.POST)
    context = {
        'form' : form,
        'valid' : False,
        'stock' : None,
        'result' : 0.0,
    }
    if request.method == 'POST':
        if form.is_valid():
            context['valid'] = True

            # Save the form data to the variable
            stock = form.cleaned_data['stock']

            result = 7.3  #ai_predictor.get_stock_value(str(stock))
            context['result'] = result

            return redirect(f'/page/stock/result/?stock={stock}')
    else:
        form = stockForm()
    return render(request, 'stock_form.html', context)

def result(request):
    stock = request.GET.get('stock')
    result =  ai_predictor.get_stock_value(stock)
    context = {
        'result' : result,
        'stock' : stock,
        'prediction' : 0,
        'confidence' : round((1 - result)*100,1),
    }

    if result > 0.5:
        context['prediction'] = 1
        context['confidence'] = round(result*100,1)

    return render(request, 'result.html', context)
