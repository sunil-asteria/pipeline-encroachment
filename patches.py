import matplotlib.pyplot as plt
import numpy as np
import cv2
import uuid
import os
import fiftyone as fo
import json

patch_height = 1280
patch_width = 1280

def rect_to_bbox(patch):
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        x, y, width, height = cv2.boundingRect(contours[0])
        return True,(x, y, width, height)
    return False,None

def pad_image(img, patch_height, patch_width):
    img_height, img_width, _= img.shape
    pad_top = 0
    pad_bottom = patch_height - (img_height % patch_height)
    pad_left = 0
    pad_right = patch_width - (img_width % patch_width)
    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    return padded_img
    
def crop_patches(padded_img, patch_height, patch_width):
    no_of_rows = padded_img.shape[0] // patch_height
    no_of_columns = padded_img.shape[1] // patch_width
    patches = []
    for i in range(no_of_rows):
        for j in range(no_of_columns):
            top = i * patch_height
            bottom = (i + 1) * patch_height
            left = j * patch_width
            right = (j + 1) * patch_width
            cropped_image = padded_img[top:bottom, left:right]
            patches.append(cropped_image)
    return patches, no_of_rows, no_of_columns
    
def generate_patches_with_bboxes(image,bboxes):
    annotations = []
    image_patches,no_of_rows_image, no_of_columns_image = crop_patches(image,patch_height,patch_width)
    
    for bbox in bboxes:
        all_patches = []
        x, y, w, h, cls = bbox
       
        empty_img =  np.zeros(image.shape[:2], dtype=np.uint8) 
        cv2.rectangle(empty_img, (int(x),int(y)),(int(x+w),int(y+h)), (255,255, 255), -1)
        black_patches,no_of_rows_black, no_of_columns_black = crop_patches(empty_img,patch_height,patch_width)
        for i,black_patch in enumerate(black_patches):
            flag,bbox = rect_to_bbox(black_patch)
            if flag:
                patch_bbox_info = {}
                patch_bbox_info['patch_id'] = i
                patch_bbox_info['class'] = cls
                rect_x,rect_y,rect_w,rect_h = bbox
                patch_bbox_info['x'] = rect_x
                patch_bbox_info['y'] = rect_y
                patch_bbox_info['w'] = rect_w
                patch_bbox_info['h'] = rect_h
                all_patches.append(patch_bbox_info)
        annotations.append(all_patches)
    return image_patches,annotations, no_of_rows_image, no_of_columns_image


dataset = fo.load_dataset("pipeline_data")

copy = dataset.clone()

output_directory = "all_patches_data"
os.makedirs(output_directory, exist_ok=True)
image_id = 0
annotation_id = 0

category_data = [{"id":0, "name":"TREES","supercategory":None},{"id":1, "name":"ELECTRIC_POLES","supercategory":None},
                 {"id":2, "name":"PERMENANT_STRUCTURES","supercategory":None},{"id":3, "name":"TEMPORARY_STRUCTURES","supercategory":None},
                 {"id":4, "name":"WELL","supercategory":None},{"id":5, "name":"PYLON","supercategory":None},
                 {"id":6, "name":"SOLAR","supercategory":None},{"id":7, "name":"TLP","supercategory":None}]
image_data = []
annotations_data = []

with fo.ProgressBar() as pb:

    for sample in copy:
        img = cv2.imread(sample.filepath)
        
        padded_img = pad_image(img, patch_height, patch_width)
        
        bboxes = []
        for annotation in sample.ground_truth.detections:
            cls = annotation.label
            (x, y, w, h) = (annotation.bounding_box[0] * img.shape[1],
                            annotation.bounding_box[1] * img.shape[0],
                            annotation.bounding_box[2] * img.shape[1],
                            annotation.bounding_box[3] * img.shape[0])
            bboxes.append([x, y, w, h, cls])
        image_patches,annotations,no_of_rows_image,no_of_columns_image = generate_patches_with_bboxes(padded_img,bboxes)
    
        for i,patch in enumerate(image_patches):
            filename = f"{uuid.uuid4()}.jpg"
            patch_path = os.path.join(output_directory, filename)
            cv2.imwrite(patch_path,patch)
            patch_data = {
                "id": image_id,
                "file_name": filename,
                "width": patch.shape[1],
                "height": patch.shape[0],
            }
            image_data.append(patch_data)
            
    
            for annotation in annotations:
                for sub_obj in annotation:
                    if sub_obj["patch_id"] == i:
                        x,y,w,h,cls = sub_obj['x'],sub_obj['y'],sub_obj['w'],sub_obj['h'],sub_obj['class']
                        for category in category_data:
                            if cls == category["name"]:
                                class_id = category["id"]
                        annotation_data = {"id": annotation_id,
                                           "image_id": image_id,
                                           "category_id": class_id,
                                           "bbox": [x,y,w,h],
                                           "score": 1.0,
                                           "area": w*h,
                                           "iscrowd": 0
                                          }
                        annotations_data.append(annotation_data)
                        annotation_id += 1
                        
            image_id += 1


coco_format = {
        "images": image_data,
        "annotations": annotations_data,
        "categories": category_data
    }

with open(('coco_format.json'), 'w') as f:
        json.dump(coco_format, f)

