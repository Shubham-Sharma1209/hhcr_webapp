from django.db import models
from pdf2image import convert_from_path
import os
from mysite.settings import MEDIA_ROOT

# Create your models here.
class Document(models.Model):
    document=models.FileField(upload_to='hhcr',default="")
    predicted_document=models.FileField(upload_to='predicted_image',default='out.jpg')
    upload_date=models.DateTimeField(auto_now=True)

    def __str__(self):
        return f'{self.document}'
    
    def is_pdf(self):
        fpath=self.fpath()
        return fpath.endswith('.pdf')  # assuming all PDFs are saved with .pdf extension in the media folder. Adjust this as needed.
    
    def pdf_to_img(self):
        if self.is_pdf():
            self.document = convert_from_path("/media/"+str(self))
    def file_path(self):
        return os.path.join(MEDIA_ROOT, str(self.document))

    def delete_file(self):
        os.remove(self.fpath())

        
    # def __del__(self):
    #     os.remove(self.fpath())
    #     super().__del__()


