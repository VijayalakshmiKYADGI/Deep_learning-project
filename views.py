from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from tensorflow.keras.utils import to_categorical 
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler  
from sklearn.svm import SVC 
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import model_from_json
# Create your views here.
import numpy as np

def binary(request):
    if request.method=='POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs=FileSystemStorage()
        filename=fs.save("uploads//"+myfile.name,myfile)
        json_file = open('C:\\Users\\vijay\\DesktoP\\FOOT_CRIC\\model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("C:\\Users\\vijay\\Desktop\\FOOT_CRIC\\model.h5")
        img_name =  filename
        test_image = image.load_img(img_name, target_size = (64, 64))

        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'INDIAN_CRICKET_PLAYER'
        else:
            prediction = 'FOOTBALL_PLAYERS'
        print(prediction,img_name)
        result= "Predicted result is: " +prediction
        
        return render(request, 'binary.html',context={'result':result})

    return render(request, 'binary.html')


def rice(request):
       
    if request.method=='POST' :
        data=request.POST
        area=data.get('area')
        maj=data.get('ma')
        min=data.get('mi')
        ecc=data.get('ecc')
        ca=data.get('ca')
        ed=data.get('ed')
        ex=data.get('ex')
        pe=data.get('per')
        rou=data.get('rou')
        asp=data.get('asp')
        
        path="C:\\Users\\vijay\\Desktop\\RICE_CLASS\\riceClassification.csv"
        data=pd.read_csv(path)
        inputs= data.drop(['id','Class'],'columns')
        output= data.drop(['Area','MajorAxisLength','AspectRation','MinorAxisLength','Eccentricity','Extent','Perimeter','Roundness','ConvexArea','EquivDiameter','id'],'columns')
        x_train,x_test,y_train,y_test = train_test_split(inputs,output,train_size=0.8)
        sc= StandardScaler()                    
        x_train=sc.fit_transform(x_train)
        x_test=sc.transform(x_test)
            
        model=SVC()
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
            
            #
        newinputs=np.array([[float(area),float(maj),float(min),float(ecc),float(ca),float(ed),float(ex),float(pe),float(rou),float(asp)]])
        newinputs=sc.transform(newinputs)
        result = model.predict(newinputs)
        classe= "Predicted class is: " +str(result)
                        
        if ('buttonsubmit' in request.POST):
            return render(request, 'rice.html',context={'result':classe})
    return render(request,'rice.html')

def index(request):
    
    return render(request,'index.html')
