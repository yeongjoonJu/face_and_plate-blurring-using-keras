from . import model
import glob
import os
import cv2
import numpy as np

def create_input_image_embeddings(email, images):
    input_embeddings = {}
    inception_model = model.create_model()
    base_path = 'images/' + email
    for image in images:
        filename = os.path.join(base_path, image)
        image_file = cv2.imread(filename, 1)
        input_embeddings[filename] = image_to_embedding(image_file, inception_model)

    return input_embeddings, inception_model


def image_to_embedding(image, model):
    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) 
    #image = cv2.resize(image, (96, 96)) 
    img = image[...,::-1]
    img = np.around(np.transpose(img, (0,1,2))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def recognize_face(face_image, input_embeddings, model):
    embedding = image_to_embedding(face_image, model)
    
    minimum_distance = 200
    name = None
    
    # Loop over  names and encodings.
    for (input_name, input_embedding) in input_embeddings.items():       
        euclidean_distance = np.linalg.norm(embedding-input_embedding)
        print('Euclidean distance from %s is %s' %(input_name, euclidean_distance))
        if euclidean_distance < minimum_distance:
            minimum_distance = euclidean_distance
            name = input_name
    
    if minimum_distance < 0.68:
        return str(name)
    else:
        return None


def detect_face(image, input_embeddings, model, face_detector, font=cv2.FONT_HERSHEY_SIMPLEX):
    height, width, cmap = image.shape
    gray = cv2.cvtColor(cv2.UMat(image), cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    # Loop through all the faces detected
    identities = []
    
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x+w
        y2 = y+h

        face_image = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
        identity = recognize_face(face_image, input_embeddings, model)

        if identity is not None:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, str(identity), (x1+5, y1-5), font, 1, (255,255,255), 2)
    
    return image