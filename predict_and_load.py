import cv2
import fiftyone as fo
dataset = fo.load_dataset("pipeline_data")

from ultralytics import YOLO

detection_model = YOLO("Best_version_2.pt")

# import cv2


with fo.ProgressBar() as pb:
    for sample in dataset:
        image = cv2.imread(sample.filepath)
        image_height,image_width,c = image.shape
        
        results = detection_model(sample.filepath)
        
        # print(results)
        detections = []
        for idx, prediction in enumerate(results[0].boxes.xywhn):
            cls = int(results[0].boxes.cls[idx].item())
            class_name = detection_model.names[cls]
            # print(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
            score = float(results[0].boxes.conf[idx])
            rel = [prediction[0].item(), prediction[1].item(), prediction[2].item(), prediction[3].item()]
            # print(class_name,rel,score)
            x,y,w,h = rel[0]*image_width,rel[1]*image_height, rel[2]*image_width,rel[3]*image_height
            x1 = x-w/2
            y1 = y-h/2
            new_rel = [x1/image_width,y1/image_height,w/image_width,h/image_height]
            # if class_name == "ELECTRIC POLES":
            #     class_name = "ELECTRIC_POLES"

            # if class_name == "PERMENANT STRUCTURES":
            #     class_name = "PERMENANT_STRUCTURES"

            # if class_name == "TEMPORARY STRUCTURES":
            #     class_name = "TEMPORARY_STRUCTURES"
            
            detections.append(
                    fo.Detection(
                        label=class_name,
                        bounding_box=new_rel,
                        confidence=score
                    )
                )
        sample["sahi_predictions_v2_1280x1280_2"] = fo.Detections(detections=detections)
        sample.save()
    
    # break
    # break