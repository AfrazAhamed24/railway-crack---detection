import sys
import json
import os
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO

# Suppress ultralytics logging
os.environ['YOLO_VERBOSE'] = 'False'

# Load once and reuse.
DEFAULT_MODEL_PATH = os.environ.get(
    "CRACK_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "best.railway-crack.pt")
)
INFERENCE_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.03"))
model = YOLO(DEFAULT_MODEL_PATH)

# A crack detector must expose a crack/defect class. The bundled file was
# previously a COCO detector, which produced false "crack" detections such as
# bird/person. Detect that configuration once and fail clearly.
CRACK_CLASS_TOKENS = (
    "crack", "rail_crack", "railway_crack", "track_crack",
    "fracture", "defect", "damage"
)
COCO_CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
}

MODEL_NAMES = {int(k): str(v).strip().lower() for k, v in model.names.items()}
CRACK_CLASS_IDS = {
    class_id for class_id, name in MODEL_NAMES.items()
    if any(token in name for token in CRACK_CLASS_TOKENS)
}
IS_CRACK_MODEL = bool(CRACK_CLASS_IDS) or (
    len(MODEL_NAMES) == 1 and not set(MODEL_NAMES.values()).issubset(COCO_CLASS_NAMES)
)


def model_status():
    return {
        "path": DEFAULT_MODEL_PATH,
        "classes": list(MODEL_NAMES.values()),
        "crackClassIds": sorted(CRACK_CLASS_IDS),
        "isCrackModel": IS_CRACK_MODEL,
    }


def ensure_crack_model():
    if not IS_CRACK_MODEL:
        raise RuntimeError(
            "Configured YOLO weights are not a crack detector. "
            f"Found classes: {', '.join(MODEL_NAMES.values())}. "
            "Replace best.pt with crack-trained weights."
        )


def _run_variant_inference(frame, conf=INFERENCE_CONFIDENCE):
    """Try multiple visual variants to improve detection on phone-screen captures."""
    variants = [frame]

    # Contrast-enhanced grayscale reprojected to BGR.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    variants.append(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR))

    # Local contrast enhancement helps thin dark cracks survive uneven rail
    # lighting and phone-camera glare.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_frame = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    variants.append(clahe_frame)

    # Mild sharpening to highlight thin crack lines.
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    variants.append(cv2.filter2D(frame, -1, sharpen_kernel))

    best_results = None
    best_score = -1.0
    best_conf = 0.0

    for candidate in variants:
        results = model(candidate, conf=conf, imgsz=1280, augment=True, verbose=False)
        current_boxes = 0
        current_max_conf = 0.0
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                valid = [
                    i for i, class_id in enumerate(boxes.cls.int().tolist())
                    if not CRACK_CLASS_IDS or class_id in CRACK_CLASS_IDS
                ]
                if valid:
                    current_boxes += len(valid)
                    current_max_conf = max(float(boxes.conf[i]) for i in valid)

        # Prefer higher confidence first, then more boxes.
        score = current_max_conf * 10.0 + current_boxes
        if score > best_score:
            best_score = score
            best_conf = current_max_conf
            best_results = results

    return best_results, best_conf


def detect_crack(frame, conf=INFERENCE_CONFIDENCE):
    """Run crack detection on a single frame and return normalized metadata."""
    ensure_crack_model()
    results, _ = _run_variant_inference(frame, conf=conf)

    for r in results:
        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            valid = [
                i for i, class_id in enumerate(boxes.cls.int().tolist())
                if not CRACK_CLASS_IDS or class_id in CRACK_CLASS_IDS
            ]
            if not valid:
                continue
            first = valid[0]
            return {
                "crack": True,
                "confidence": float(boxes.conf[first]),
                "bbox": boxes.xyxy[first].tolist()
            }

    return {"crack": False}


def run_inference(image_path):
    try:
        ensure_crack_model()
        frame = cv2.imread(image_path)
        if frame is None:
            print(json.dumps({"error": "Unable to read input image"}))
            return

        results, max_conf = _run_variant_inference(frame, conf=INFERENCE_CONFIDENCE)
        
        predictions = []
        for r in results:
            for box in r.boxes:
                b = box.xywh[0].tolist() # get box coordinates in (top, left, width, height) format
                c = box.cls.item()
                conf = box.conf.item()
                if CRACK_CLASS_IDS and int(c) not in CRACK_CLASS_IDS:
                    continue
                name = model.names[int(c)]
                
                predictions.append({
                    "x": b[0],
                    "y": b[1],
                    "width": b[2],
                    "height": b[3],
                    "confidence": conf,
                    "class": name,
                    "label": "CRACK"
                })
        
        result = {
            "crackDetected": len(predictions) > 0,
            "confidence": max_conf,
            "predictions": predictions,
            "imageSize": {
                "width": results[0].orig_shape[1],
                "height": results[0].orig_shape[0]
            },
            "model": model_status()
        }
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e), "model": model_status()}))


def run_camera_detection(camera_index=0, location="Track KM 120"):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    last_saved_second = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detect_crack(frame)
        time_now = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {time_now}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Location: {location}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if result["crack"]:
            x1, y1, x2, y2 = map(int, result["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "CRACK DETECTED", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Alert: Crack detected (confidence={result['confidence']:.2f})")

            # Save at most one frame per second to avoid duplicate burst writes.
            current_second = datetime.now().strftime("%Y%m%d_%H%M%S")
            if current_second != last_saved_second:
                filename = f"crack_{current_second}.jpg"
                cv2.imwrite(filename, frame)
                last_saved_second = current_second

        cv2.imshow("Crack Detection", frame)

        # Exit on ESC.
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        run_camera_detection()
    else:
        if sys.argv[1] == "--camera":
            run_camera_detection()
        else:
            run_inference(sys.argv[1])
