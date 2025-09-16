import cv2
import fiftyone as fo
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict

dataset = fo.load_dataset("pipeline_data")

yolov8_model_path = "Best_version_2.pt"

sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=yolov8_model_path,
    confidence_threshold=0.3,
    device="cuda:0",
)


for sample in dataset:
    
    result = get_sliced_prediction(
        sample.filepath,
        sahi_model,
        slice_height = 1280,
        slice_width = 1280,
        overlap_height_ratio = 0.2,
        overlap_width_ratio = 0.2
    )

    fiftyone_detections = result.to_fiftyone_detections()
    
    sample["sahi_predictions_v2_1280x1280"] = fo.Detections(detections=fiftyone_detections)
    sample.save()
  


