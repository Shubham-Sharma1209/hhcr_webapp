from django import forms  
from .models import Document

class UploadForm(forms.ModelForm):
    class Meta:
        model=Document
        fields=['document']
        widgets={
            'document': forms.FileInput(attrs={'id':'doc_id','hidden':True})
        }