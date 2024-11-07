from django import forms


class stockForm(forms.Form):
    stock = forms.CharField(label="Stock", max_length=100)