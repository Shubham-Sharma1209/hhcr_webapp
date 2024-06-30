from django.shortcuts import render
from django.shortcuts import HttpResponseRedirect
from .forms import UploadForm
from .models import Document
from .predictor import Predictor
# Create your views here.
def index(request):
    if request.method == 'POST':
        form=UploadForm(request.POST,request.FILES)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('upload')
    else:
        form=UploadForm()
    return render(request, 'hhcr/index.html',{'form':form})


p=Predictor()

def upload(response):
    # doc=Documents()
    latest_doc=Document.objects.order_by("-upload_date",)[0]
    p.set_document(latest_doc)
    out_file_path=p.segment_and_predict()
    Document.objects.filter(id=latest_doc.id).update(predicted_document=out_file_path)
    params={'output':out_file_path}
    return render(response,'hhcr/upload.html',params)

