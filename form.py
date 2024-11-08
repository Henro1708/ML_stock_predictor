from django import forms


class stockForm(forms.Form):
    stock = forms.CharField(required=False,label="Stock to predict:", max_length=100)