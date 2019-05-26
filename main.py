import cv2
import os
from PIL import Image
import argparse

from detection_model import Yolo_Ensemble
from face_recognize import Face_Recognition

def mask_rectangle(img, rect):
    (x1, y1, x2, y2) = rect
    
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 0)
    dst = img[y1:y2, x1:x2]
    dst = cv2.GaussianBlur(dst, (99,99), 30)
    img[y1:y2, x1:x2]= dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)   
    '''
    Command line arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help = "Video input path"
    )
    parser.add_argument(
        "--output", nargs='?', type=str, default="./results",
        help = "Video output path"
    )
    parser.add_argument(
        "--email", nargs='?', type=str, default="",
        help = "user email"
    )

    args = parser.parse_args()
    
    use_recognition = True
    if args.email == "":
        use_recognition = False

    # load model
    if use_recognition:
        face_recog_model = Face_Recognition(args.email)    
    detect_model = Yolo_Ensemble()
    detect_model.load_model('model_data/yolo_face_model.h5', 'model_data/yolo_plate_model.h5')
    
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        face_boxes, plate_boxes = detect_model.detect(image)
        
        # recognize faces
        if use_recognition:
            face_boxes = face_recog_model.recognize(frame, face_boxes)

        predicted = face_boxes + plate_boxes

        for box in predicted:
            mask_rectangle(frame, box)

        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
   
    detect_model.close_session()
 
