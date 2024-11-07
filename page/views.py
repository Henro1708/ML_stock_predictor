from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from stock_values import ai_predictor
from form import stockForm
from django.views.decorators.csrf import ensure_csrf_cookie

# Create your views here.
@ensure_csrf_cookie
def stock_view(request):
    stock = None  # To store the submitted stock input
    form = stockForm(request.POST)
    context = {
        'form' : form,
        'loaded' : False,
        'valid' : False,
        'stock' : None,
        'result' : 0.0,
    }
    if request.method == 'POST':
        if form.is_valid():
            context['valid'] = True

            # Save the form data to the variable
            stock = form.cleaned_data['stock']
            context['stock'] = stock

            result = 7.3  #ai_predictor.get_stock_value(str(stock))
            context['result'] = result

            return render(request, 'stock_form.html', context)
    else:
        form = stockForm()

    
    return render(request, 'stock_form.html', context)
