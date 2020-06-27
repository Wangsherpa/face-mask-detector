# Implementing our COVID-19 face mask detector training script with Pytorch and Deep Learning

I have used Anaconda's Jupyter notebook to write and visualize the result of the python program.

## This face mask detector project has four parts:
  1. Detect the face and extract the ROI
  2. Using the above face ROI detect the results (mask or not_mask)
  3. Detect COVID-19 face mask in images
  4. Detect COVID-19 mask in real-time video streams
  
In order to train a custom face mask detector, we need to break our project into two distinct phases, each 
with its own repective sub-steps
  1. **Training**: Here we'll focus on loading our face mask detection dataset from the disk and train the model (pretrained model)
  on this dataset
  2. **Deployment**: Once the face mask detector is trained, we can then move on to loading the mask detector, performing face detection, and classifying each face as with_mask or without_mask
  
# Let's move on to the notebook now:-

# Training Phase:-

**cell-1:** Import all the required libraries and modules
```
import os
import cv2
import time
import torch
import imutils
import numpy as np
import torch.nn.functional as F

from torch import nn, optim
from tqdm.notebook import tqdm
from torchvision import models, datasets, transforms
```
**Cell-2:** Loading the dataset and implementing transforms (resize, convert images to tensor and normalize)
            . Also define the DataLoader for training and testing dataset
```
# define data directory
data_dir = 'dataset/'

# define train, test directory
train_dir = os.path.join(data_dir, 'train/')
test_dir = os.path.join(data_dir, 'test/')

# define transformations
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])
                               ])

# load train and test data
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

# define dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10,
                                           shuffle = True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10,
                                          shuffle = True)

# Print the number of pictures in the datasets
print(f"No of Training Images: {len(train_data)},",
      f"No of Testing Images: {len(test_data)}")
```

**Cell-3:** Loading the pretrained VGG16 model (I have used VGG16 here but you can test with other pretrained model also)
```
# load the model
model = models.vgg16(pretrained=True)

# freeze the weights
for param in model.parameters():
    param.requires_grad = False
    
# Check if gpu is available
use_cuda = torch.cuda.is_available()

# no of parameters comming from last cnn layer
n_inputs = model.classifier[6].in_features

# Change the last layer according to our need
model.classifier[6] = nn.Linear(n_inputs, 2)

# if gpu is available then add model to gpu
if use_cuda:
    model = model.cuda()

# verify if the model architecture is as expected
print(model)
```
**Cell-4:** Define loss and optimizer
```
# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
```
**Cell-5:** Training the model (This script can be improved by adding validation and early stopping)
This is a very simple script in pytorch to train the model
```
def train(model):
    for e in range(n_epochs):
        train_loss = 0
        for images, labels in tqdm(train_loader):
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()

            # Clear previous accumulated gradients
            optimizer.zero_grad()

            # forward pass
            output = model(images)

            # calculate loss
            loss = criterion(output, labels)

            # backward pass (backpropagation)
            loss.backward()

            # Update weights
            optimizer.step()

            # Update train_loss
            train_loss += loss.item()

        print(f" Epochs: {e+1}/{n_epochs}, Train_loss : {train_loss / len(train_loader)}")
        # Saving the model weights
        torch.save(model.state_dict(), 'model-1-acc-99.pt')
train(model)
```
**Cell-6:** Testing the model on testing set which the model has never seen before
```
test_loss = 0.
correct = 0.
total = 0.

model.eval()

for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
    
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    
    # forward pass
    output = model(data)
    
    # calculate the loss
    loss = criterion(output, target)
    
    # update the averae test loss
    test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
    
    # convert the output probabilities to predicted class
    pred = output.data.max(1, keepdim=True)[1]
    
    # compare the prediction to true label
    correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
    total += data.size(0)
    
print('Test Loss: {:.6f}\n'.format(test_loss))
print('\n Test Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
```
** Test Accuracy **
![Accuracy](https://github.com/Wangsherpa/face-mask-detector/blob/master/images/test_accuracy.jpg)

# Deployment Phase:-
1. This function will take an image and trained model, and return the predicted class index and corresponding probability
```
# Function to test custom images
from PIL import Image

def with_or_without_mask(image, model):
    input_img = Image.fromarray(image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(input_img)
    input_batch = input_tensor.unsqueeze(0)
    
    # Move the input to gpu if available
    if use_cuda:
        input_batch = input_batch.cuda()
        
    with torch.no_grad():
        output = model(input_batch)
        
    prob, index = torch.max(F.softmax(output[0]), 0)
    
    # predicted class index and probability
    return prob, index
```

2. Face detection and prediction in images.
```
# loading our serialized face detector model from disk
print("Loading face detector model...")
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("Loading face mask detector model")
model.load_state_dict(torch.load('model-1-acc-99.pt'))

# load the input image from disk, clone it, and grab the image dimensions
# image = cv2.imread('examples/example_01.png')
# Using ip camera
cap = cv2.VideoCapture('http://192.168.225.24:8080/video')
# cap = cv2.VideoCapture(0) # Using device camera
_, image = cap.read()
# orig = image.copy()
(h, w) = image.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the face detections
print("Computing face detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e, probability) associated with the detection
    confidence = detections[0, 0, i, 2]
    
    # filter out weak detections by ensuring the confidence is greater
    # than the minimum confidence
    if confidence > 0.5:
        # compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7]  * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # ensure the bounding boxes fall within the dimensions of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w-1, endX), min(h-1, endY))
        
        # preprocess the image and get the class and proba
        prob, class_ = with_or_without_mask(image[startY:endY, startX:endX], model)
        
        # determine the class label and color we'll use to draw the bounding
        # box and text
        label = "Mask" if class_ == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, prob)
        
        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# show the output image
cv2.imshow("output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

3. Face detection and prediction on a real time video.
This function will take an frame, face detector model and a face mask detector model and returns list of face coordinates and predictions
```
def detect_and_predict(frame, faceNet, maskNet):
    # grab the dimension of the frame and then construct a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    # initialize our list of faces, their corresponding locations
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence associated with the detections
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence greater than
        # the minimum confidence
        if confidence > 0.7:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')
            
            # ensure the bounding boxex fall within the dimension of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            
            # extract the face ROI and preprocess and predict
            face = frame[startY:endY, startX:endX]
            faces.append(face)
            locs.append((startX, startY, endX, endY))
            
    # only make a predictions if atleast one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on all the
        # faces at the same time rather than one-by-one prediction
        # in the above for loop
        for face in faces:
            pred = with_or_without_mask(face, maskNet)
            preds.append(pred)
    return (locs, preds)
```

# Face detection on a live video stream
```
# loading our serialized face detector model from disk
print("Loading face detector model...")
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
model.load_state_dict(torch.load('model-1-acc-99.pt'))

# Starting video stream
vs = cv2.VideoCapture('http://192.168.225.24:8080/video')
# vs = cv2.VideoCapture(0)
# time.sleep(2.0)

# loop over the frames from the video stream
while True:
    
    # grab the frame from the threaded video stram and resize to have a
    # minimum of 400 px
    _, frame = vs.read()
#     frame = imutils.resize(frame, width=400)
    
    # detect faces in the frame and determine if they are wearing
    # a face mask or not
    (locs, preds) = detect_and_predict(frame, faceNet, model)
    
    # loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (prob, clas) = pred
        
        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if clas == 0 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, prob)
        
        # display the label and bounding box rectangle on the original frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    # show the output frame
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key is pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
```

# Real outputs:-
- With Mask:

![without mask](https://github.com/Wangsherpa/face-mask-detector/blob/master/images/with_mask.jpg)

- Without Mask:

![without mask](https://github.com/Wangsherpa/face-mask-detector/blob/master/images/without_mask.jpg)

# Suggestions for imporvement
As we can see from the results sections above, our face mask detector is working quite well despite:
1. Having limited training data
2. The with_mask class being artificially generated.

To improve our face detection model, we can add more images and not the artificial images.
The actual images of person wearing the mask and the actual images of a person without wearing a mask.











