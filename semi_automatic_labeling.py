import os.path
import labelme.label_file as lf
import base64
import json
from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from skimage import measure
import numpy as np
import cv2

classes = [
'front_left_window',
'front_right_window',
'back_left_window',
'back_right_window',
'back_bumper',
'back_glass',
'back_left_door',
'back_left_light',
'back_right_door',
'back_right_light',
'front_bumper',
'front_glass',
'front_left_door',
'front_left_light',
'front_right_door',
'front_right_light',
'hood',
'left_mirror',
'right_mirror',
'wheel',
'tailgate',
'left_body',
'right_body',
'bus_left_window',
'bus_right_window',
'left_mid_window',
'right_mid_window',
'back_body',
'trunk',
]


IMA_PATH = './imgs/'
# Specify the path to model config and checkpoint work_dir
config_file = 'work_dir/mask_rcnn_r101_fpn_mstrain-poly_3x_coco_tag.py'
checkpoint_file = 'work_dir/epoch_12.pth'

# build the model from a config work_dir and a checkpoint work_dir
model = init_detector(config_file, checkpoint_file, device='cuda:0')


file_name_list = os.listdir(IMA_PATH)

def get_height_width(img_path):
    img = Image.open(img_path)
    return img.height,img.width


def get_img_json(shapes,imagePath=None,imageData=None,imageHeight=512,imageWidth=512):
    article_info = {}
    data = json.loads(json.dumps(article_info))

    data['version'] = '4.6.0'
    data['flags'] = {}
    data['shapes'] = shapes
    data['imagePath'] = imagePath
    data['imageData'] = imageData
    data['imageHeight'] = imageHeight
    data['imageWidth'] = imageWidth


    return data

def get_shape(label,points,shape_type='polygon',group_id=None):
    article_info = {}
    data = json.loads(json.dumps(article_info))

    data['label'] = label
    data['points'] = points
    data['group_id'] = group_id
    data['shape_type'] = shape_type
    data['flags'] = {}


    return data




SCORE_THR = 0.9
FIND_CONTOURS_LEVEL = 0.5

if __name__ == '__main__':


    for file in file_name_list:
        print('process '+file)

        img_path = os.path.join(IMA_PATH,file)
        result = inference_detector(model, img_path)
        result_bbox = result[0]
        result_segment_binary = result[1]



        result_dir_list = []

        for class_num in range(len(result_bbox)):
            result_dir = {}
            result_dir['class_name'] = classes[class_num]
            result_dir['bbox'] = []
            result_dir['segment'] = []
            for instance in range(len(result_bbox[class_num])):

                if result_bbox[class_num][instance][4] >= SCORE_THR:
                    result_dir['bbox'].append(result_bbox[class_num][instance][:4])

                    contours = measure.find_contours(np.array(result_segment_binary[class_num][instance]).astype(int).T,
                                                     FIND_CONTOURS_LEVEL)

                    contours = cv2.convexHull(contours[0].astype(np.int32))

                    result_dir['segment'].append(contours.squeeze())

            if len(result_dir['bbox']) >= 1:
                result_dir_list.append(result_dir)

        shape_list = []
        for res_dir in result_dir_list:
            for bbox in res_dir['bbox']:
                shape = get_shape(res_dir['class_name'],bbox.reshape(2,2).tolist(),'rectangle')
                shape_list.append(shape)
            for segment  in  res_dir['segment']:
                shape = get_shape(res_dir['class_name'],segment.tolist(),'polygon')
                shape_list.append(shape)


        imageData = lf.LabelFile.load_image_file(img_path)
        imageData = base64.b64encode(imageData).decode("utf-8")
        imageHeight,imageWidth = get_height_width(img_path)

        img_json = get_img_json(shape_list,file,imageData,imageHeight,imageWidth)
        json_path = os.path.join(IMA_PATH,file.split('.')[0]+'.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(img_json, f, ensure_ascii=False)






