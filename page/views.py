from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from stock_values import ai_predictor
from form import NameForm
from django.views.decorators.csrf import ensure_csrf_cookie

# Create your views here.
stock = 'AAPL'
prediction = ai_predictor.get_stock_value()
ret_values = prediction

def get_prediction(request):
    return HttpResponse(ret_values)

@ensure_csrf_cookie
def check(request):
    return render(request, 'get_stock.html', {'result': str(prediction) })

@ensure_csrf_cookie
def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect("/thanks/")

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm()

    return render(request, "form.html", {"form": form})
