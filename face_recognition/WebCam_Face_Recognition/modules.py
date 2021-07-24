import os
import pickle
from face_recognition import app
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from numpy import load, asarray, savez_compressed
from numpy import expand_dims
from keras.models import load_model
from scipy import ndimage
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from datetime import datetime
import os
import tensorflow as tf
import psutil
from reportlab.platypus import SimpleDocTemplate, Table
from reportlab.lib.pagesizes import letter
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from openpyxl import Workbook # pip install openpyxl 
from openpyxl.styles import Font, Color

path = "./face_recognition/8_5-Dataset"
model_path = "./face_recognition/WebCam_Face_Recognition/models/"
npz_path = "./face_recognition/WebCam_Face_Recognition/npz_data/"

# path = "./8_5-Dataset"
# model_path = "./WebCam_Face_Recognition/models/"
# npz_path = "./WebCam_Face_Recognition/npz_data/"

def make_pdf(data, filename):
    pdf = SimpleDocTemplate(
                filename,
                pagesize = letter
    )
    table = Table(data)
    
    # add style

    style = TableStyle([
        ('BACKGROUND', (0,0), (4,0), colors.green),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),

        ('ALIGN',(0,0),(-1,-1),'CENTER'),

        ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 14),

        ('BOTTOMPADDING', (0,0), (-1,0), 12),

        ('BACKGROUND',(0,1),(-1,-1),colors.beige),
    ])
    table.setStyle(style)

    # 2) Alternate backgroud color
    rowNumb = len(data)
    for i in range(1, rowNumb):
        if i % 2 == 0:
            bc = colors.burlywood
        else:
            bc = colors.beige
        
        ts = TableStyle(
            [('BACKGROUND', (0,i),(-1,i), bc)]
        )
        table.setStyle(ts)

    # 3) Add borders
    ts = TableStyle(
        [
        ('BOX',(0,0),(-1,-1),2,colors.black),

        ('LINEBEFORE',(2,1),(2,-1),2,colors.red),
        ('LINEABOVE',(0,2),(-1,2),2,colors.green),

        ('GRID',(0,1),(-1,-1),2,colors.black),
        ]
    )
    table.setStyle(ts)
    elems = []
    elems.append(table)

    pdf.build(elems)  # SAVES PDF OF ATTENDANCE NAMED {filename}

def make_excel(data, filename):
    wb = Workbook()
    ws1 = wb.active # work sheet
    ws1.title = "Pyxl"
    for i in range(len(data)):
        for j in range(len(data[i])):
            #print(i, j)
            _ = ws1.cell(column=j+1, row=i+1, value=data[i][j])
            #ws1.append([data[row]])

    wb.save(filename)

def load_facenet_model():
    try:
        facenet_model = load_model(model_path+'facenet_keras.h5')
        facenet_model.keras_model._make_predict_function()
    except:
        new_model_path = os.path.join(os.getcwd(),model_path[2:])
        print(new_model_path)
        facenet_model = load_model(new_model_path + 'facenet_keras.h5')
    if facenet_model:
        print("facenet model loaded!")
        return facenet_model

def load_caffe_model():
    print("loading model...")
    model = cv2.dnn.readNetFromCaffe(model_path+"deploy.prototxt.txt", model_path+"res10_300x300_ssd_iter_140000.caffemodel")
    if model:
        print("caffe model loaded")
        return model


def get_slot(time=datetime.now()):
    # '2021-04-03 17:03:37.159053'
    start = time.replace(hour=10, minute=20, second=0, microsecond=0)
    slot1 = time.replace(hour=11, minute=00, second=0, microsecond=0)
    slot2 = time.replace(hour=11, minute=40, second=0, microsecond=0)
    slot3 = time.replace(hour=13, minute=10, second=0, microsecond=0)
    slot4 = time.replace(hour=15, minute=25, second=0, microsecond=0)
    slot5 = time.replace(hour=16, minute=45, second=0, microsecond=0)
    #slot5 = time.replace(hour=4, minute=25, second=0, microsecond=0)
    if time < start:
        return 2
    elif time < slot1:
        return 2
    elif time < slot2:
        return 2
    elif time < slot3:
        return 2
    elif time < slot4:
        return 2
    elif time < slot5:
        return 2
    else:
        return 2


def dataAugmentation(original):
    augmented = [original]
    # horizontal flip
    flipped = cv2.flip(original, 1)
    augmented.append(flipped)
    # rotate clockvise & counterclock
    augmented.append(ndimage.rotate(original, -20, (1, 0)))
    augmented.append(ndimage.rotate(original, 20, (1, 0)))
    # brightness
    alpha_1, alpha_2 = 1, 1.5
    beta_1, beta_2 = 50, 0
    augmented.append(cv2.convertScaleAbs(flipped, alpha=alpha_1, beta=beta_1))
    # contrast
    augmented.append(cv2.convertScaleAbs(original, alpha=alpha_2, beta=beta_2))
    return augmented


def data_augmentation_of_all_images(dir_path):
    affected = []
    for name in os.listdir(dir_path)[::-1]:
        if name == "augmented" or name[0] == "$":
            continue
        Path(os.path.join(dir_path,"augmented",name)).mkdir(parents=True, exist_ok=True)
        i = 0
        for ind, filename in enumerate(os.listdir(os.path.join(dir_path,name))):
            i += 1
            # dst = dir_path +name+ "/"+name+"_train_"+str(ind)
            # src = dir_path + name + "/" + filename
            src = os.path.join(dir_path,name,filename)
            print(src)
            img = Image.open(src)
            img = img.convert('RGB')
            img = asarray(img)
            L = dataAugmentation(img)
            j = 1
            for each in L:
                aug_iamge = Image.fromarray(each)
                save_path = os.path.join(dir_path,"augmented",name,name+"_aug_"+str(i)+"_"+str(j)+".jpg")
                #aug_iamge.save(dir_path + "augmented/" + name + "/" + name + "_aug_" + str(i) + "_" + str(j) + ".jpg")
                aug_iamge.save(save_path)
                j += 1

        os.rename(os.path.join(dir_path,name),os.path.join(dir_path,"$"+name))
        affected.append(name)
    return affected

def get_embeddings(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    # Z-score
    mean = face_pixels.mean()
    std = face_pixels.std()
    face_pixels = (face_pixels - mean) / std

    # convert in required dimension for facenet model
    # print("before:",face_pixels.shape)
    samples = expand_dims(face_pixels, axis=0)
    # print("after:",samples.shape)
    graph = app.config['GRAPH']
    with graph.as_default():
        embs = model.predict(samples)
    #print(embs[0].shape)
    # print(embs[0])
    return embs[0]
# load dataset

def extract_faces(model, filepath, required_size=(160, 160)):
    '''
    :param model: face detection loaded model
    :param filepath: path of directory where all images is stored
    :param required_size: crop faces size
    :return: tuple of original images numpy array & cropped faces numpy array
    '''
    cropped = []
    original = []
    files = os.listdir(filepath)
    for filename in files:
        # open image & convert into an numpy array
        image = cv2.imread(filepath + "/" + filename)
        im = Image.open(filepath + "/" + filename)
        image = np.array(im)

        # print("++++ {} ++++".format(filename))
        # print("original image shape=",image.shape)

        # resize the image
        image = cv2.resize(image, (300, 300))

        # image preprocessing like mean substraction
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        # print(blob.shape)

        # pass the blob through the network and obtain the detections and
        # predictions
        model.setInput(blob)
        try:
            detections = model.forward()
        except:
            print("skip {} due to model error".format(filename))
            continue
        # print(detections.shape)
        '''
        detection.shape =  (1, 1, 200, 7)
        7 columns:

        3rd column = confidence of pixel
        4th column = (startX)/width
        5th column = (startY)/height
        6th column = (endX)/width
        7th column = (endY)/height

        '''

        height, width = image.shape[:2]
        # print("resized image shape:",image.shape)
        # Image.fromarray(image, 'RGB').show()

        # maximum of all detections -> detect most accurate one face
        confidence = detections.max(axis=2)[0][0][2]
        arg = detections.argmax(axis=2)[0][0][2]
        # print(arg)
        # take last axis
        each = detections[0][0][arg]
        # print("Maximum confidence:",confidence)
        # multiple image detection
        #     for each in detections[0][0]:
        #         confidence = each[2]
        if confidence > 0.5:
            try:
                # print(confidence)
                startX_factor, startY_factor, endX_factor, endY_factor = each[3:]
                # print(startX_factor)
                startX, startY, endX, endY = int(startX_factor * width), int(startY_factor * height), int(
                    endX_factor * width), int(endY_factor * height)
                # print((startX, startY, endX, endY))
                # cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),2)
                # text = str(round(confidence*100,2))
                # cv2.putText(image,text,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
                crop = image[startY:endY, startX:endX]
                # Image.fromarray(image,'RGB').show()
                # Image.fromarray(crop, 'RGB').show()

                # resize face to (160,160,3)
                crop_img = Image.fromarray(crop)
                crop_img = crop_img.resize(required_size)
                face = asarray(crop_img)
                # print("face shape",face.shape)
                if confidence < 0.5:
                    Image.fromarray(face, 'RGB').show()

                cropped.append(face)
                original.append(image)
            except:
                print("face not detected in", filename)
        else:
            print("low confidence:{},face not detected in {}".format(confidence, filename))
    print("{}/{} faces detected in {}".format(len(cropped), len(files), filepath))
    return (cropped,original)


def load_face_classifier():
    print("loading SVM model....")
    data = load(os.path.join(path,npz_path,'8_5-face-embeddings.npz'))
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    #l-2 normalization
    encoder = Normalizer()
    trainX = encoder.transform(trainX)
    testX = encoder.transform(testX)


    #conver string labels into numeric
    label_encoder = LabelEncoder()
    label_encoder.fit(trainy)
    trainy = label_encoder.transform(trainy)
    testy = label_encoder.transform(testy)

    #load SVM Classifier
    svm_model = SVC(kernel='linear',probability=True)
    svm_model.fit(trainX,trainy)

    #prediction
    train_pred = svm_model.predict(trainX)
    test_pred = svm_model.predict(testX)

    #accuracy score
    print("train accuracy:",accuracy_score(trainy,train_pred))
    print("test accuracy:",accuracy_score(testy,test_pred))
    return (svm_model,label_encoder)

def load_train_face_classifier():
    print("loading SVM model....")
    data = load(os.path.join(npz_path,'8_5-train_face-embeddings.npz'))
    trainX, trainy = data['arr_0'], data['arr_1']
    print('Dataset: train=%d' % (trainX.shape[0]))

    #l-2 normalization
    encoder = Normalizer()
    trainX = encoder.transform(trainX)



    #conver string labels into numeric
    label_encoder = LabelEncoder()
    label_encoder.fit(trainy)
    trainy = label_encoder.transform(trainy)


    #load SVM Classifier
    svm_model = SVC(kernel='linear',probability=True)
    svm_model.fit(trainX,trainy)

    #prediction
    train_pred = svm_model.predict(trainX)


    #accuracy score
    print("train accuracy:",accuracy_score(trainy,train_pred))
    return (svm_model,label_encoder)
#model = load_caffe_model()

def add_new_persons():

    #load models
    #facenet_model = load_facenet_model()
    facenet_model = app.config['FACENET_MODEL']
    model = load_caffe_model()


    #Data augmentation
    train_path = os.path.join(path,"train")
    train_aug_path = os.path.join(path, "train/augmented")
    affected_labels = data_augmentation_of_all_images(train_path)

    #Extract Faces
    X_face, y_face = [], []
    X_original, y_original = [], []
    for label in affected_labels:
        label_path = os.path.join(train_aug_path,label)
        x_cropped, x_original = extract_faces(model,label_path)
        X_face.extend(x_cropped)
        X_original.extend(x_original)
        y_face.extend([label]*len(x_cropped))

    #append these faces in existing data
    try:
        data = load(npz_path+"8_5-train_faces.npz",allow_pickle=True)
    except:
        print("no such file named 8_5-train_faces.npz")
        data = {'arr_0':asarray([]),'arr_1':asarray([])}

    X_faces_train, Y_faces_train = data['arr_0'].tolist(), data['arr_1'].tolist()
    X_faces_train.extend(X_face)
    Y_faces_train.extend(y_face)
    savez_compressed(npz_path+"8_5-train_faces.npz", asarray(X_faces_train), asarray(Y_faces_train))

    #getting face embedings
    try:
        embs = load(npz_path+"8_5-train_face-embeddings.npz",allow_pickle=True)
    except:
        print("no file named 8_5-train_face-embeddings.npz")
        embs = {'arr_0':asarray([]),'arr_1':asarray([])}

    X_embs_train , Y_embs_train = embs['arr_0'].tolist(), embs['arr_1'].tolist()

    for face in X_face:
        embeddings = get_embeddings(facenet_model, face)
        X_embs_train.append(embeddings)

    Y_embs_train = asarray(Y_faces_train)
    savez_compressed(npz_path+"8_5-train_face-embeddings.npz", asarray(X_embs_train), asarray(Y_embs_train))

    #retrain svm_model
    svm_model, encoder = load_train_face_classifier()
    pickle.dump(svm_model,open(model_path+"svm_train_face_model","wb"))
    pickle.dump(encoder,open(model_path+"label_train_face_encoder","wb"))
    app.config['SVM_MODEL'] = svm_model
    app.config['LABEL_ENCODER'] = encoder
    print(f"Affected labels:{affected_labels}\nModel Successfully Retrained!")
    return affected_labels
