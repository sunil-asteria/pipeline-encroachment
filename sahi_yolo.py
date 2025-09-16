#from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
#from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "Pipeline_models/Best_version_2.pt"
#yolov8_model_path = "Pipeline_models/Best_version_2.pt"
print("running model ",yolov8_model_path)
result=predict(
    model_type="yolov8",
    model_path=yolov8_model_path,
    model_device="cuda:0",  # or 'cuda:0'
    model_confidence_threshold=0.4,
    source="test_data/images/val",#Pipline_workspace/test_data"
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)