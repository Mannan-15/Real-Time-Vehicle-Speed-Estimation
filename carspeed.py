import argparse
import supervision as sv
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict, deque

# Define the polygon for the region of interest
source = np.array([[1313, 763], [2432, 773], [2529, 1079], [811, 1073]])

target_width = 25
target_height = 250

target = np.array([
    [0, 0],
    [target_width - 1, 0],
    [target_width - 1, target_height - 1],
    [0, target_height - 1]
])

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed Estimation Project")
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Load the model and move it to the GPU
    model = YOLO("yolov8x.pt").to("cuda")

    box_annotator = sv.BoxAnnotator(thickness=4)
    
    label_annotator = sv.LabelAnnotator(
        text_thickness=2,
        text_scale=0.5,
        text_position=sv.Position.BOTTOM_CENTER
    )
    
    trace_annotator = sv.TraceAnnotator(
        thickness=2,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )

    poly = sv.PolygonZone(polygon=source)
    
    view_transformer = ViewTransformer(source=source, target=target)
    
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Use the model as a generator directly over the video file for efficient GPU usage
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    for frame in frame_generator:
        # Run a prediction on the frame
        result = model(frame, device=0)[0]
        
        detections = sv.Detections.from_ultralytics(result)
        
        # Filter detections by the polygon zone
        detections = detections[poly.trigger(detections)]
        
        # Update tracker with the filtered detections
        detections = byte_track.update_with_detections(detections=detections)
        
        if detections.xyxy.size > 0:
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)
            
            labels = []
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f'#{tracker_id}')
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f'#{tracker_id} {speed:.2f} km/h')
        else:
            labels = []
        
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=source)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        
        cv2.imshow("Car Speed Detection", annotated_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
