
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import imutils
from time import sleep
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.ndimage import zoom
import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization


#model = model_from_json(open('models/Face_model_architecture.json').read())
model = model_from_json(open('models/model_arch.json').read())
model.load_weights('./weights_201.hdf5')

# Prepara o Modelo para Compilação
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# Compila o modelo
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Extrai as Features
def extract_face_features(gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
    
        extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face

# Detecta as Faces
def detect_face(frame):
        cascPath = "models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        return gray, detected_faces

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)


video_capture = cv2.VideoCapture(1) 


while True:
    # Captura frame-by-frame
    ret, frame = video_capture.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video


    frame = imutils.resize(frame, width = 800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta as Faces
    gray, detected_faces = detect_face(frame)
    
    face_index = 0
    
    # Previsões
    for face in detected_faces:
        (x, y, w, h) = face
        if w > 100:
            # Desenha um retângulo em torno das faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            
            # Extrai as features
            extracted_face = extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)

            # Prevendo sorrisos
            prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))

            # Desenha o rosto extraído no canto superior direito
            frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

            # Anota a imagem principal com uma etiqueta
            if prediction_result == 3:
                cv2.putText(frame, "Feliz!",(x,y), cv2.FONT_ITALIC, 2, 200, 10)                        
            elif prediction_result == 2:
                cv2.putText(frame, "Com Medo",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 10)
            elif prediction_result == 4:
                cv2.putText(frame, "Triste",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 10)
            elif prediction_result == 0:
                cv2.putText(frame, "Surpreso",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 10)
            else :
                cv2.putText(frame, "Neutro",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 10)
            

            # Incrementa o contador
            face_index += 1
                

    # Exibe o frame resultante
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver pronto, libera a captura
video_capture.release()
cv2.destroyAllWindows()
