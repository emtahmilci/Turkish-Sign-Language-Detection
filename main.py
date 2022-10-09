import os
import cv2 
import numpy as np
import tensorflow as tf
from tkinter import *
import PIL.Image
import PIL.ImageTk
from gtts import gTTS
from pygame import mixer
import time

from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt
# %matplotlib inline

bg_color= '#9a8c98'
bg_color2='#f2e9e4'
tx_color= '#22223b'
sub_tx_color= '#4a4e69'



CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'

paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }

files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)


# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-41')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
print(LABEL_MAP_NAME)
def sign_lang():
     cap = cv2.VideoCapture(0)
     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
     time_b=0
     while cap.isOpened(): 
         ret, frame = cap.read()
         image_np = np.array(frame)
        
         input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
         detections = detect_fn(input_tensor)
        
         num_detections = int(detections.pop('num_detections'))
         detections = {key: value[0, :num_detections].numpy()
                       for key, value in detections.items()}
         detections['num_detections'] = num_detections
    
         # detection_classes should be ints.
         detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    
         label_id_offset = 1
         image_np_with_detections = image_np.copy()
         viz_utils.visualize_boxes_and_labels_on_image_array(
                     image_np_with_detections,
                     detections['detection_boxes'],
                     detections['detection_classes']+label_id_offset,
                     detections['detection_scores'],
                     category_index,
                     use_normalized_coordinates=True,
                     max_boxes_to_draw=5,
                     min_score_thresh=.8,
                     agnostic_mode=False)
         for i in range(0,100):
             if(detections['detection_scores'][i]>.8) and time_b+2<time.time():    
                 title = category_index[detections['detection_classes'][i]+1]['name']
                  mixer.init()
                  mixer.music.load('sounds/'+title+'.mp3')
                  mixer.music.play()
                 time_b = time.time()
                
         cv2.imshow('Isaret Dili Algilama',  cv2.resize(image_np_with_detections, (800, 600)))
        
         if cv2.waitKey(10) & 0xFF == ord('q'):
             cap.release()
             cv2.destroyAllWindows()
             break
    

window = Tk()
window.title("İşaret Dili Algılama")
canvas = Canvas(window, height=450, width=750, bg=bg_color)
canvas.pack()

frame_ust = Frame(window, bg=bg_color2)
frame_ust.place(relx=0.1, rely=0.1, relwidth=0.80, relheight=0.1)

frame_sol_alt = Frame(window, bg=bg_color2)
frame_sol_alt.place(relx=0.1, rely=0.21, relwidth=0.30, relheight=0.60)

frame_sag_alt = Frame(window, bg=bg_color2)
frame_sag_alt.place(relx=0.41, rely=0.21, relwidth=0.49, relheight=0.60)

Label(frame_sag_alt, bg=bg_color2, fg=tx_color, text = "Program Hakkında", font = "Verdana 12 bold").pack()
Label(frame_sag_alt, bg=bg_color2, text ='Program Türkçe işaret dilindeki 20 hareketi algılamak üzere eğitilmiştir.\n Programın algıladığı hareketler aşağıda sıralanmıştır:\n\n Merhaba, Nasılsın, İyiyim\n Güle güle, Gezmek  \n Aile, Kardeş\n Ev, Lazım \n Evet, Hayır \n Öğretmek, Misafir \n Yardım, Sormak  \n Sevmek, Özür dilemek \n Taksi, Üçgen, Saat \n\n Bu Program Ekin Mete Tahmilci, Ahmet Can Işıklar \n ve Ecenur Karakaya tarafından geliştirilmiştir.').pack(side = LEFT)

im = PIL.Image.open("icon/icons.png")
photo = PIL.ImageTk.PhotoImage(im)
label = Label(frame_sol_alt, image=photo)
label.image = photo
label.pack(pady=30)

etiket = Label(frame_ust, bg=bg_color2, fg=tx_color, text= "İşaret Dili Algılama", font = "Verdana 14 bold")
etiket.pack(padx=10, pady=10)

run_button = Button(frame_sol_alt, bg=bg_color, fg=tx_color, text="Programı Başlat", command=sign_lang) #command
run_button.pack(pady= 40, side = BOTTOM)

window.mainloop()




