import fiftyone as fo
import fiftyone.utils.labels as fol

dataset = fo.load_dataset("copy_pipeline_data")

fol.perform_nms(sample_collection=dataset, in_field="sahi_predictions_v3_640x640", out_field="nms_patch", confidence_thresh=0.15, iou_thresh=0.10,classwise=True, progress=True)

fol.perform_nms(sample_collection=dataset, in_field="sahi_predictions_v2_1280x1280", out_field="nms_1280", confidence_thresh=0.15, iou_thresh=0.10,classwise=True, progress=True)

dataset.save()