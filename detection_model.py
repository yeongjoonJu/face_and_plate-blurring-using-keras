import os
import numpy as np
import cv2
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
from keras.utils import multi_gpu_model

class Yolo_Ensemble(object):
    def __init__(self, allow_score=0.15, iou=0.45, gpu_num=1):
        self.sess = K.get_session()
        self.tiny_anchors = np.array([10.0,14.0, 23.0,27.0, 37.0,58.0, 81.0,82.0, 135.0,169.0, 344.0,319.0]).reshape(-1, 2)
        self.anchors = np.array([10.0,13.0, 16.0,30.0, 33.0,23.0, 30.0,61.0, 62.0,45.0, 59.0,119.0, 116.0,90.0, 156.0,198.0, 373.0,326.0]).reshape(-1, 2)
        self.model_image_size = (480,480)
        self.ensemble_score = allow_score
        self.iou = iou
        self.input_image_shape = K.placeholder(shape=(2,))
        self.gpu_num = gpu_num
    
    def generate(self, model_path, anchors, score):
        print(model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        num_anchors = len(anchors)
        num_classes = 1

        model = None
        try:
            model = load_model(model_path, compile=False)
        except:
            if num_anchors > 6:
                model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            else:
                model = tiny_yolo_body(Input(shape=(None, None, 3)), num_anchors//2, num_classes)
            model.load_weights(model_path)
        
        
        if self.gpu_num >=2:
            model = multi_gpu_model(model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(model.output, anchors,
                num_classes, self.input_image_shape,
                score_threshold=score, iou_threshold=self.iou)

        return {'model':model, 'boxes':boxes, 'scores': scores, 'classes': classes}


    def load_model(self, yolo_face_path, tiny_yolo_face_path, yolo_plate_path):
        self.gen_face_model = self.generate(yolo_face_path, self.anchors, 0.08)
        #self.tiny_face_model = self.generate(tiny_yolo_face_path, self.tiny_anchors, 0.24)
        self.gen_plate_model = self.generate(yolo_plate_path, self.anchors, 0.14)


    def get_overapped_portion(self, x1, y1, x2, y2, x3, y3, x4, y4):
        x_list = [x1,x2,x3,x4]
        y_list = [y1,y2,y3,y4]
        min_x = min(x_list)
        min_y = min(y_list)
        max_x = max(x_list)
        max_y = max(y_list)
        all_range = np.zeros((max_y-min_y, max_x-min_x),dtype=int)
        all_range[y1-min_y:y2-min_y, x1-min_x:x2-min_x] = 1
        all_range[y3-min_y:y4-min_y, x3-min_x:x4-min_x] += 1
        overlapped_area = len(all_range[all_range==2])
        area1 = (x2-x1) * (y2-y1)
        return overlapped_area / area1

    def detect(self, frame):
        real_boxes = []
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        new_image_size = (frame.width - (frame.width % 32),
                          frame.height - (frame.height % 32))
        boxed_frame = letterbox_image(frame, new_image_size)
        image_data = np.array(boxed_frame, dtype='float32')

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        face_boxes, face_scores, _ = self.sess.run(
            [self.gen_face_model['boxes'], self.gen_face_model['scores'], self.gen_face_model['classes']],
            feed_dict={
                self.gen_face_model['model'].input: image_data,
                self.input_image_shape: [frame.size[1], frame.size[0]],
                K.learning_phase(): 0
            })
        """
        tiny_face_boxes, tiny_face_scores, _ = self.sess.run(
            [self.tiny_face_model['boxes'], self.tiny_face_model['scores'], self.gen_face_model['classes']],
            feed_dict={
                self.tiny_face_model['model'].input: image_data,
                self.input_image_shape: [frame.size[1], frame.size[0]],
                K.learning_phase(): 0
            })

        plate_boxes, plate_scores, _ = self.sess.run(
            [self.gen_plate_model['boxes'], self.gen_plate_model['scores'], self.gen_face_model['classes']],
            feed_dict={
                self.gen_plate_model['model'].input: image_data,
                self.input_image_shape: [frame.size[1], frame.size[0]],
                K.learning_phase(): 0
            })
        """
        # face detection
        for i, score in enumerate(face_scores):
            gen_box = face_boxes[i]
            y1, x1, y2, x2 = gen_box
            y1 = max(0, np.floor(y1 + 2.5).astype('int32'))
            x1 = max(0, np.floor(x1 + 2.5).astype('int32'))
            y2 = min(frame.size[1], np.floor(y2 + 2.5).astype('int32'))
            x2 = min(frame.size[0], np.floor(x2 + 2.5).astype('int32'))
            """
            allow = False
            # gen_face_model에서 예측한 확률이 낮으면 tiny모델과 겹치는 부분을 확인,
            # 일정 부분 이상 겹치지 않으면 오인예측으로 봄
            if score < 0.24:
                for tiny_box in tiny_face_boxes:
                    y3, x3, y4, x4 = tiny_box
                    y3 = max(0, np.floor(y3 + 2.5).astype('int32'))
                    x3 = max(0, np.floor(x3 + 2.5).astype('int32'))
                    y4 = min(frame.size[1], np.floor(y4 + 2.5).astype('int32'))
                    x4 = min(frame.size[0], np.floor(x4 + 2.5).astype('int32'))
                    portion =  self.get_overapped_portion(x1, y1, x2, y2, x3, y3, x4, y4)
                    if portion >= self.ensemble_score:
                        allow = True
                        break
            else:
                allow = True
            """
            if True:
                real_boxes.append((x1, y1, x2, y2))
        """
        for i, score in enumerate(tiny_face_scores):
            if score > 0.4:
                y3, x3, y4, x4 = tiny_box
                y3 = max(0, np.floor(y3 + 2.5).astype('int32'))
                x3 = max(0, np.floor(x3 + 2.5).astype('int32'))
                y4 = min(frame.size[1], np.floor(y4 + 2.5).astype('int32'))
                x4 = min(frame.size[0], np.floor(x4 + 2.5).astype('int32'))
                real_boxes.append((x3,y3,x4,y4))
                self.mask_rectangle(frame, real_boxes[-1])
        """
        return real_boxes


    def close_session(self):
        self.sess.close()