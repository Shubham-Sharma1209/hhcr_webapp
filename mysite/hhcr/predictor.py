from imutils import paths
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import os
import re
from keras.saving import load_model
from mysite.settings import MEDIA_ROOT


import warnings
warnings.filterwarnings("ignore")

class Predictor():
    
    def __init__(self):
        self.loaded_model=load_model('hhcr\model_cnn\CNN_model_2-32-30-numclasses-36.h5')
    
    def set_document(self,Document):
        self.Document=Document
        self.get_image()
    
    def get_image(self):
        try:
            self.image=cv.imread(self.Document.file_path())
        except FileNotFoundError:
            print("File Not Found")

    def convert_to_grayscale(self,img): 
        median=cv.medianBlur(img, 3, 15)
        gray=cv.cvtColor(median,cv.COLOR_BGR2GRAY)
        return gray
    
    def get_adaptive_thresh(self,gray):
        adaptive_thresh=cv.adaptiveThreshold(gray,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,15, 11)
        return adaptive_thresh
    
    def get_otsu(self,img):
        blur = cv.GaussianBlur(img,(5,5),0)
        grayscale=cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
        ret2,th2 = cv.threshold(grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('Otsu',th2)
        return th2
    
    def erode(self,img):
        ero_shape=cv.MORPH_RECT
        ero_size=2
        element=cv.getStructuringElement(ero_shape,(2*ero_size+1,2*ero_size+1))
        return cv.erode(img,element)
    
    def com_area(self,rect1,rect2):
        widthoverlap = min(rect1[2],rect2[2]) >= max(rect1[0],rect2[0]) 
        heightoverlap = min(rect1[3],rect2[3]) >= max( rect1[1],rect2[1])
        return (widthoverlap or ( (max(rect1[0],rect2[0]) - min(rect1[2],rect2[2]) ) < 0) ) and heightoverlap


    def merge_rect(self,rect1,rect2):
        return (min(rect1[0],rect2[0]),min(rect1[1],rect2[1]),max(rect1[2],rect2[2]),max(rect1[3],rect2[3]))

    def area(self,word):
        return (word[2]-word[0])*(word[3]-word[1])

    def add_padding(self,img):
        return cv.copyMakeBorder(img,2,2,2,2,cv.BORDER_REPLICATE,value=(255,255,255))
    

    def find_contours(self,eroded_img):
        contours, hierarchy = cv.findContours(eroded_img, cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

        print(f'{len(contours)} Contours found!')
        # cont_img=cv.drawContours(eroded_img,contours,-1,0,2)
        # cv.imshow('contours image',cont_img)

        img_with_contours=eroded_img.copy()
        possible_words=[]

        for c,h in zip(contours,hierarchy[0]):
            x,y,w,h=cv.boundingRect(c)
            x,y,x_r,y_b=x,y,x+w,y+h
            if max((x_r-x)/(y_b-y),(y_b-y)/(x_r-x))>10  or cv.contourArea(c)<100  or cv.contourArea(c)>(self.image.shape[0]*self.image.shape[1])/100:
                continue
            img_with_contours = cv.rectangle(img_with_contours,(x,y),(x_r,y_b),(0,0,255),2)
            possible_words.append((x,y,x_r,y_b))

        # cv.imshow('All contours',img_with_contours)
        while True:
            check=0
            for i,word1 in enumerate(possible_words):
                for word2 in possible_words[i+1:]:
                    if self.com_area(word1,word2):
                        check=1
                        possible_words.remove(word2)
                        word1 = self.merge_rect(word1,word2)
                        possible_words[i]=word1
                if self.area(word1)>8000:
                    possible_words.remove(word1)
                    check=1
            if check==0:
                break


        possible_words=list(set(possible_words))
        return possible_words

    def segment_and_predict(self):

        self.characters=pd.read_csv("hhcr\static\hhcr\characters.csv",index_col="index")

        fontpath="hhcr\static\hhcr\MartelSans-Light.ttf"
        font = ImageFont.truetype(fontpath, 32)

        
        
        gray=self.convert_to_grayscale(self.image)
        adaptive_thresh=self.get_adaptive_thresh(gray)
        eroded_img=self.erode(adaptive_thresh)
        word_contours=self.find_contours(eroded_img)
        # self.char_splitter(word_contours)
        # print(word_contours)
        # cv.imshow('gray',gray)
        # img_pil = Image.fromarray(out_img)
        # draw = ImageDraw.Draw(img_pil)
        # cv.imshow('eroded',eroded_img)
        out_img=self.image.copy()

        for x,y,x_r,y_b in word_contours:
            # result=predict_images((img[y:y_b,x:x_r]))
            result=self.predict_images(self.add_padding(self.image[y:y_b,x:x_r]))
            out_img=cv.rectangle(out_img,(x,y),(x_r,y_b),(0,0,255),1)
            cv.putText(out_img,self.characters.iloc[int(result),2] , (x,y-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.6, (26,4,135), 1)
            # draw.rectangle((x,y,x_r,y_b),(0,0,255))
            # draw.text((x,y-10),characters.iloc[int(result),1],font=font,fill=(12,255,212,0))
        # out_img=np.array(img_pil)

        out_image_path='pred_'+str(self.Document.document).split('/')[-1]
        res_path=os.path.join('predicted_image',out_image_path)
        out_image_path=os.path.join(MEDIA_ROOT,res_path)
        cv.imwrite(out_image_path,out_img)
        return res_path
        # cv.waitKey(0)
        
    def see_words(self,word_contours):
        words=self.image.copy()
        for word in word_contours:
            x,y,x_r,y_b=word
            cv.rectangle(words,(x,y),(x_r,y_b),(0,0,255),1,0)
        cv.imshow('words',words)


        

    def char_splitter(self,word_contours):
        k=0
        for x,y,x_r,y_b in word_contours:
            word=self.image[y:y_b,x:x_r]
            word=self.get_adaptive_thresh(self.convert_to_grayscale(word))
            # deskew(word)
            if 0 in word.shape:
                continue
            word=255-word
            horizontal_histogram=word.sum(axis=1)
            height,width=word.shape
            horizontal_histogram//=255    
            # print(word.shape,horizontal_histogram.shape)
            upper_end=-1
            no_upper=False
            # cv.imwrite(f'./words/{x}-{y}-{x_r}-{y_b}.png',word)
            for loc, v in enumerate(horizontal_histogram):
                # v=v//255
                if v>=width*0.85:
                    # print(k,1)
                    upper_end=loc+1
                    break
                if loc<height-2:
                    # print(    k,2)
                    # if (horizontal_histogram[loc-1]+horizontal_histogram[loc-2]+v)//3>width*1.5:
                    if (horizontal_histogram[loc+1]+horizontal_histogram[loc+2]+v)>width:
                        upper_end=loc+1
                        break
                if loc>=height*0.3:
                    # print(k,3)
                    upper_end=loc
                    no_upper=True
                    break
            if no_upper or loc<height*0.15:
                lower_start=height*0.8
            else:
                lower_start=height*0.7

            # word=word[int(upper_end)+1:int(lower_start),:]
            # word=word[int(upper_end)+1:,:]
            # y+=int(upper_end)+1
            vertical_histogram=word.sum(axis=0)
            vertical_histogram//=255
            start=0
            cheight,width=word.shape
            n=1

            # cv.imwrite(f'./without_upper/{x}-{y}-{w}-{h}-{loc}.png',word)
            word=255-word
            for bel,v in enumerate(vertical_histogram):
                # print(start,v,bel)
                if v<=1:
                    # if v<5 and bel>0:
                    #     if abs(vertical_histogram[bel]-v)<=1:
                    #         continue
                    # if v>=5:
                    #     continue
                    if (bel-start)>=cheight*0.7:
                        # print(bel)
                        if word[:,start:bel].shape[0]*word[:,start:bel].shape[1]>300:
                            # cv.imwrite(f'./characters/{n}_{x}_{y}_{x_r}_{y_b}.png',img[y:y_b,x+start:x+bel+2])
                            # cv.imwrite(f'./characters/{x+start}_{y}_{x+bel+2}_{y_b}.png',img[y:y_b,x+start:x+bel+2])
                            # try:
                                # print(x+start,y,x_r+bel+5,y)
                            result=self.predict_images(self.add_padding(self.image[y-1:y_b+1,x+start-2:x+bel+2]))
                            out_img=cv.rectangle(out_img,(x+start,y),(x+bel,y_b),(0,0,255),1)
                            cv.putText(out_img,self.characters.iloc[int(result),2] , (x+start,y-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.6, (212,20  ,12), 1)

                            # except:
                            #     pass
                            n+=1
                            start=bel+1
                # else:
                #     start+=1
            k+=1
            # print(k)   
    def predict_images(self,image,verbose=False):
        img_height,img_width=32,32
        image_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image_gray=cv.adaptiveThreshold(image_gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,5)
        image = cv.resize(image_gray, (img_height, img_width))

        img=np.array(image)
        img = img.reshape(1, img_height, img_width, 1)
        img = img.astype('float32')
        img /= 255
        result=self.loaded_model.predict(img,verbose=0)
        res=self.loaded_model(img)
        predicted_class = int(np.argmax(result, axis=1))
        return predicted_class