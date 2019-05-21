import cv2
import os
import argparse
from face_recognition.recognition import create_input_image_embeddings, detect_face
from face_recognition.model import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)   
    '''
    Command line arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=True, default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="./results",
        help = "Video output path"
    )
    parser.add_argument(
        "--email", nargs='?', type=str, default="",
        help = "[Optional] user email"
    )    

    args = parser.parse_args()
    user_email = 'go1217jo@naver.com'
    face_images = ['yeongjoon1.jpg']
    img = cv2.imread(args.input)
    input_embeddings, inception = create_input_image_embeddings(user_email, face_images)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    predicted = detect_face(img, input_embeddings, inception, face_cascade)
    cv2.imshow('frame', predicted)