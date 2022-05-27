from asyncio.windows_events import NULL
from flask import Flask, redirect, render_template, request
from matplotlib.pyplot import get
from sqlalchemy import false, true
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder 
import torch
import torch.nn as nn
from torchvision import models
import os
import shutil
import cv2
import imghdr

class ImageClassificationBase(nn.Module):                # Classes for our trained image classification model (Resnet34)
    pass

class ResNet34(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet34(pretrained=pretrained)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

app = Flask(__name__)

PATH = 'final_maybe.pth'                                # Loading the image classification model onto the gpu
model = ResNet34(8)
model.load_state_dict(torch.load(PATH))
model.eval()
model.cuda()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def get_default_device():                                  # Note: A machine with gpu is must for this webapp to work
    """Pick GPU if available, else CPU"""                  # Pytorch with cuda support must be installed
    if torch.cuda.is_available():                          # GPU must be supported by pytorch
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
device = get_default_device()           

def predict_image(img, model):                             # Returns the prediction(name of anime character) from the model
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return clasing[preds[0].item()]


clasing = ['Asuka Langley Soryu',                           # Image classification classes
 'Chitoge Kirisaki',
 'Kosaki Onodera',
 'Kurumi Tokisaki',
 'Rei Ayanami',
 'Saber Fate',
 'Tobichi Origami',
 'Tohsaka Rin']

classifier = cv2.CascadeClassifier('lbpcascade_animeface.xml')         # Classifier for seperating multiple Anime faces from an image

def pre_processing():                               # This function returns a list of predictions for each recognised face

    all_preds = []
    data_dir = "static/Dataset/final/"
    list = os.listdir("static/Dataset/final/hello/")
    l = len(list)

    if l == 0:
        return all_preds

    something = ImageFolder(data_dir)               # Normalising our data according to imagenet dataset. Since our model is based on it.
    img_size = 224
    imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    something = ImageFolder(data_dir, tt.Compose([tt.Resize(img_size),tt.RandomCrop(img_size),tt.ToTensor(),tt.Normalize(*imagenet_stats)]))

    x = 0
    for i in something:
        imgg,label = something[x]
        all_preds.append(predict_image(imgg, model))
        x = x+1
    return all_preds

def seperate_faces(img_path):                  #This function seperates all the faces in the given image, crops and saves them 
    image = cv2.imread(img_path)                
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    faces = classifier.detectMultiScale(gray_image)           

    output_dir = 'static/Dataset/final/hello'

    for i, (x,y,w,h) in enumerate(faces):
        face_image = image[y:y+h, x:x+w]
        output_path = os.path.join(output_dir,'{0}.jpg'.format(i))
        cv2.imwrite(output_path,face_image)

def clear_stuff():                              # This function clears all the stored images used for every prediction
    dir = 'static/Dataset/final/hello/'
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

    another_dir = 'static/Dataset/Seperate/'
    for files in os.listdir(another_dir):
        path = os.path.join(another_dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)

def get_images(l):                            # Returns a list containing paths of all the seperated faces(images)  
    img_paths = []
    for i in range(0,l):
        img_paths.append("static/Dataset/final/hello/{num}.jpg".format(num = i))
    return img_paths

def get_text(l,all_preds):                   # Returns a list containing paths of .txt files containing information about the repective Anime character
    text = []
    for i in range(0,l):
        to_put = "static/Info/{name}.txt".format(name = all_preds[i])
        to_put = to_put.replace(" ", "%20", 2)
        text.append(to_put)
    
    return text

# routes

@app.route("/", methods=['GET', 'POST'])                   
def main():
    return render_template("base.html")

@app.route("/model1", methods=['GET','POST'])               
def first():
    return render_template("index.html")

@app.route("/model2", methods = ['GET', 'POST'])           

def second():
    return render_template("model2.html")

@app.route("/submit", methods = ['GET', 'POST'])
            
def get_output():

    clear_stuff()
            
    if request.method == 'POST':                                     # saving the file uploaded by the user
        img = request.files['my_image']
        if img.filename == "":
            return redirect('/model1');
        img_path = "static/Dataset/Seperate/" + img.filename	
        img.save(img_path)

        type = imghdr.what(img_path)                                 # checking if the file uploaded by the user is a valid image type
        if (type == "png" or type == "jpg" or type == "jpeg"):
            bool = true
        else:
            bool = false
        
        if bool == false:
            return render_template("index.html", bool = bool)        # if not a valid file type display error message

        seperate_faces(img_path)                          
        all_preds = pre_processing()                      
        l =len(all_preds)                           
        img_paths = get_images(l)                         
        text = get_text(l,all_preds)                      
    else:
        return redirect('/');        
    
    predictions = []            # Creating a list whose elements are themselves lists with 3 elements (image paths, predictions and path of .txt files)
    
    for i in range(len(img_paths)): 
        temp = [img_paths[i], all_preds[i], text[i]]
        predictions.append(temp)

    return render_template("index.html", predictions = predictions, l = l, img_path = img_path) # img_path is the path of parent image
    
@app.route("/submit2", methods = ['GET', 'POST'])

def model22():

    clear_stuff()

    if request.method == 'POST':                                        # saving the file uploaded by the user
        img = request.files['my_image']
        if img.filename == "":
            return redirect('/model2');
        img_path = "static/Dataset/final/hello/" + img.filename	
        img.save(img_path)

        type = imghdr.what(img_path)                                    # checking if the file uploaded by the user is a valid image type
        if (type == "png" or type == "jpg" or type == "jpeg"):
            bool = true
        else:
            bool = false
        
        if bool == false:
            return render_template("model2.html", bool = bool)          # if not a valid file type display error message

        all_preds = pre_processing()
        img_paths = get_images(1)
        text = get_text(1,all_preds)
    else:
        return redirect("/")

    predictions = []                                            # Creating a list whose elements are themselves lists with 3 elements (image paths, predictions and path of .txt files)
    for i in range(len(img_paths)): 
        temp = [img_paths[i], all_preds[i], text[i]]
        predictions.append(temp)

    return render_template("model2.html", predictions = predictions, l = 1, img_path = img_path)  # img_path is the path of parent image

@app.route("/README", methods = ['GET', 'POST'])

def readme():
    return render_template("README.html")

if __name__ =='__main__':
    app.debug = True
    app.run(debug = True)