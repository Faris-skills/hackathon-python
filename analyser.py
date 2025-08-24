import os
import tempfile
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from urllib.request import urlretrieve
from ultralytics import YOLO
from PIL import Image
import supervision as sv
import json
import argparse
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from google.cloud import vision
import base64
from openai import OpenAI
import re
from dotenv import load_dotenv
from collections import defaultdict
from scipy.spatial.distance import euclidean

# Load environment variables from a .env file if it exists.
# This is a best practice for managing API keys.
load_dotenv()

def download_video(video_source):
    """Downloads video if URL, or returns path if local."""
    if video_source.startswith("http"):
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "input_video.mp4")
        print(f"[INFO] Downloading video to {video_path}")
        urlretrieve(video_source, video_path)
        return video_path
    print(f"[INFO] Using local video: {video_source}")
    return video_source

def extract_keyframes_pyscenedetect(video_path, output_folder="keyframes_pyscenedetect"):
    """
    Uses PySceneDetect to find scene cuts and extracts the middle frame of each scene.
    Returns the output folder path and the number of saved frames.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())

    print(f"[Stage 1] Detecting scenes in video...")
    try:
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()

        if not scene_list:
            print("[INFO] No scenes detected. Video may be a single continuous shot.")
            return output_folder, 0

        saved = 0
        pbar = tqdm(scene_list, desc="[Stage 1] Extracting keyframes")
        for scene in pbar:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            key_frame_num = int((start_frame + end_frame) / 2)
            
            video_manager.seek(key_frame_num)
            ret, frame = video_manager.read()
            if ret:
                output_path = os.path.join(output_folder, f"keyframe_{saved:04d}.jpg")
                cv2.imwrite(output_path, frame)
                saved += 1
                pbar.set_postfix(saved_frames=saved)

    finally:
        video_manager.release()
    
    print(f"[INFO] ✅ Saved {saved} keyframes to {output_folder}")
    return output_folder, saved

def extract_keyframes_by_interval(video_path, output_folder="keyframes_interval", interval_seconds=1):
    """
    Extracts frames at a fixed time interval.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("[ERROR] Could not get FPS. Using a default interval of 30 frames.")
        frame_interval = 30
    else:
        frame_interval = int(fps * interval_seconds)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    pbar = tqdm(total=total_frames, desc="[Stage 1] Extracting frames by interval")

    for frame_idx in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_folder, f"frame_{saved:04d}.jpg")
            cv2.imwrite(output_path, frame)
            saved += 1
        pbar.update(frame_interval)
    
    pbar.close()
    cap.release()
    print(f"[INFO] ✅ Saved {saved} frames to {output_folder}")
    return output_folder, saved

def analyze_keyframes_with_yolo(input_folder, yolo_model_name="yolov8l.pt", confidence_threshold=0.7, nms_iou_threshold=0.5):
    """
    Analyzes keyframes with YOLO for object detection. Applies NMS and confidence filtering.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_model = YOLO(yolo_model_name)
    image_paths = sorted(Path(input_folder).glob("*.jpg"))
    results = []
    pbar = tqdm(image_paths, desc="[Stage 2] Analyzing frames with YOLO")

    for img_path in pbar:
        yolo_results = yolo_model(str(img_path), verbose=False)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        detections = detections[detections.confidence > confidence_threshold]
        
        if len(detections) > 0:
            detections = detections.with_nms(threshold=nms_iou_threshold)
        
        filtered_detections = []
        if len(detections) > 0:
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                label = yolo_results.names[detections.class_id[i]]
                confidence = detections.confidence[i]
                
                filtered_detections.append({
                    "label": label,
                    "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "count": 1 # Added count for consistency
                })

        frame_data = {
            "image_path": str(img_path),
            "detections": filtered_detections,
        }
        results.append(frame_data)

    print(f"[INFO] ✅ YOLO analysis complete for {len(results)} frames.")
    return results

def analyze_keyframes_with_google_cloud(input_folder, confidence_threshold=0.7):
    """
    Analyzes keyframes with Google Cloud Vision API for object localization.
    Requires GOOGLE_APPLICATION_CREDENTIALS env var to be set.
    """
    client = vision.ImageAnnotatorClient()
    image_paths = sorted(Path(input_folder).glob("*.jpg"))
    results = []
    pbar = tqdm(image_paths, desc="[Stage 2] Analyzing frames with Google Cloud Vision")

    for img_path in pbar:
        with open(img_path, "rb") as image_file:
            content = image_file.read()
        image = vision.Image(content=content)
        
        response = client.annotate_image({
            'image': image,
            'features': [{'type_': vision.Feature.Type.OBJECT_LOCALIZATION}],
        })
        
        detections = []
        img = cv2.imread(str(img_path))
        height, width, _ = img.shape

        for obj in response.localized_object_annotations:
            if obj.score > confidence_threshold:
                normalized_vertices = obj.bounding_poly.normalized_vertices
                
                x1 = int(normalized_vertices[0].x * width)
                y1 = int(normalized_vertices[0].y * height)
                x2 = int(normalized_vertices[2].x * width)
                y2 = int(normalized_vertices[2].y * height)

                detections.append({
                    "label": obj.name,
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence": float(obj.score),
                    "count": 1 # Added count for consistency
                })
            
        frame_data = {
            "image_path": str(img_path),
            "detections": detections,
        }
        results.append(frame_data)

    print(f"[INFO] ✅ Google Cloud Vision analysis complete for {len(results)} frames.")
    return results

def analyze_keyframes_with_openai(input_folder):
    """
    Analyzes keyframes using OpenAI's GPT-4o model for object detection and bounding box extraction.
    Note: OpenAI's vision models are not designed for precise object localization like traditional models. 
    We use a prompt to "trick" it into returning bounding boxes, which may not be as accurate as dedicated models.
    """
    client = OpenAI()
    image_paths = sorted(Path(input_folder).glob("*.jpg"))
    results = []
    pbar = tqdm(image_paths, desc="[Stage 2] Analyzing frames with OpenAI Vision")

    for img_path in pbar:
        with open(img_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt_text = """
        You are an expert image analyst. Your task is to identify all distinct objects in the image.
        For each object, provide a label and its bounding box coordinates in a list format [x1, y1, x2, y2].
        The coordinates should be in absolute pixel values, where x1, y1 are the top-left and x2, y2 are the bottom-right corners.
        Return the response as a JSON array of objects, with each object having a 'label' and 'bounding_box' key.
        The JSON should be the only content in your response. Do not include any other text, markdown formatting, or explanation.
        Example format:
        [{"label": "dog", "bounding_box": [100, 200, 300, 400]}, {"label": "cat", "bounding_box": [500, 600, 700, 800]}]
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=2048,
            )
            
            content = response.choices[0].message.content
            
            # Pre-process the content to handle common formatting issues
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Use a regular expression to find the JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            
            detections = []
            if json_match:
                json_string = json_match.group(0)
                try:
                    detections = json.loads(json_string)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Could not parse the extracted JSON for {img_path}: {e}")
            else:
                print(f"[WARNING] No valid JSON array found in OpenAI response for {img_path}. Raw response: '{content[:100]}...'")

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed for {img_path}: {e}")
            detections = []
        
        # We need to add a dummy confidence and count to match the output format.
        if isinstance(detections, list):
            for det in detections:
                det["confidence"] = 1.0
                det["count"] = 1 # Added count for consistency
        else:
            detections = []

        frame_data = {
            "image_path": str(img_path),
            "detections": detections,
        }
        results.append(frame_data)

    print(f"[INFO] ✅ OpenAI Vision analysis complete for {len(results)} frames.")
    return results


def get_openai_detailed_label(client, cropped_image):
    """
    Helper function to get a detailed label for a cropped image using OpenAI Vision.
    It returns a dictionary with 'label' and 'count' keys.
    """
    try:
        # Save the cropped image to a buffer and encode it
        img_buffer = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cropped_image.save(img_buffer.name, format="JPEG")
        with open(img_buffer.name, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove(img_buffer.name)

        # The new prompt asks for more detail and a count
        prompt_text = """
        You are an expert image analyst. Your task is to identify the object in this image.
        Provide a concise, detailed label for the object. If the object has a brand name,
        include it. If there are multiple identical items, provide the count.

        Return the response as a JSON object with 'label' and 'count' keys.
        'label' should be a short descriptive phrase.
        'count' should be an integer, default to 1 if not specified.
        The JSON should be the only content in your response. Do not include any other text or markdown.
        Example format: {"label": "Stack of Coca-Cola cans", "count": 5}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=100,
        )
        
        content = response.choices[0].message.content.strip()
        # Clean up and parse the JSON
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        
        try:
            result = json.loads(content)
            label = result.get('label', 'unknown')
            count = result.get('count', 1)
            return label, count
        except json.JSONDecodeError as e:
            print(f"[WARNING] Could not parse detailed JSON from OpenAI: {e}. Raw: '{content}'")
            return "unknown", 1
            
    except Exception as e:
        print(f"[WARNING] OpenAI relabeling failed for a cropped image: {e}")
        return "unknown", 1


def analyze_keyframes_with_hybrid_yolo_openai(input_folder, yolo_model_name="yolov8l.pt"):
    """
    Performs a hybrid analysis:
    1. Runs YOLO with a low confidence threshold to get bounding boxes for all potential objects.
    2. Crops each detected object from the original image.
    3. Sends each cropped image to OpenAI Vision API for accurate relabeling.
    4. Combines the precise YOLO bounding box with the accurate OpenAI label.
    """
    yolo_model = YOLO(yolo_model_name)
    openai_client = OpenAI()
    image_paths = sorted(Path(input_folder).glob("*.jpg"))
    hybrid_results = []
    
    pbar = tqdm(image_paths, desc="[Stage 2] Hybrid YOLO-OpenAI Analysis")

    for img_path in pbar:
        # Stage 1: Run YOLO with a very low confidence to catch everything
        # We also raise NMS to avoid filtering overlapping, but different, objects
        yolo_results = yolo_model(str(img_path), verbose=False, conf=0.1, iou=0.8)[0]
        detections = sv.Detections.from_ultralytics(yolo_results)
        
        final_detections = []
        original_image = Image.open(img_path)
        
        # Stage 2: Iterate through YOLO detections and relabel with OpenAI
        for i in range(len(detections)):
            box = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = box
            
            # Crop the image using the YOLO bounding box
            cropped_image = original_image.crop((x1, y1, x2, y2))
            
            # Stage 3: Send cropped image to OpenAI for relabeling with count
            openai_label, openai_count = get_openai_detailed_label(openai_client, cropped_image)
            
            # Stage 4: Add to final results
            final_detections.append({
                "label": openai_label,
                "count": openai_count,
                "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": float(detections.confidence[i]), # We keep YOLO's confidence score
            })

        hybrid_results.append({
            "image_path": str(img_path),
            "detections": final_detections,
        })
        
    print(f"[INFO] ✅ Hybrid YOLO-OpenAI analysis complete for {len(hybrid_results)} frames.")
    return hybrid_results

def track_objects_across_frames(detections_per_frame, max_distance=100):
    """
    Tracks objects across multiple frames using a simple centroid tracker.
    Assigns unique IDs to persistent objects.
    """
    tracked_objects = {}  # Stores last seen position and ID for each tracked object
    next_object_id = 0
    final_tracked_detections = defaultdict(list)

    for frame_data in detections_per_frame:
        current_centroids = []
        for det in frame_data['detections']:
            x1, y1, x2, y2 = det['bounding_box']
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            current_centroids.append({'label': det['label'], 'count': det['count'], 'centroid': centroid})

        current_frame_ids = {}

        # Try to match current detections with tracked objects from previous frame
        for i, current_det in enumerate(current_centroids):
            matched_id = -1
            min_dist = float('inf')

            for obj_id, obj_data in tracked_objects.items():
                distance = euclidean(current_det['centroid'], obj_data['centroid'])
                if distance < min_dist and distance < max_distance:
                    min_dist = distance
                    matched_id = obj_id
            
            if matched_id != -1:
                # Match found, update the tracked object
                current_frame_ids[i] = matched_id
                tracked_objects[matched_id] = current_det
            else:
                # No match found, this is a new object
                current_frame_ids[i] = next_object_id
                tracked_objects[next_object_id] = current_det
                next_object_id += 1

        # Add the object_id to the detections
        for i, det in enumerate(frame_data['detections']):
            det['object_id'] = current_frame_ids[i]
            final_tracked_detections[det['object_id']].append(det)

    return final_tracked_detections


def get_final_object_summary(tracked_detections):
    """
    Aggregates the tracked objects and generates a final list of unique items with counts.
    """
    final_summary = defaultdict(lambda: {'count': 0, 'labels': defaultdict(int)})

    for object_id, detections in tracked_detections.items():
        # Get the most common label for this object_id
        label_counts = defaultdict(int)
        total_count_for_id = 0
        for det in detections:
            label_counts[det['label']] += 1
            total_count_for_id = max(total_count_for_id, det['count'])
        
        most_common_label = max(label_counts, key=label_counts.get)
        
        final_summary[object_id]['count'] = total_count_for_id
        final_summary[object_id]['label'] = most_common_label

    # Format the final summary for output
    final_list = []
    for obj_id, data in final_summary.items():
        final_list.append({
            "id": obj_id,
            "label": data['label'],
            "count": data['count']
        })

    return final_list


def save_results_as_json(data, output_path):
    """Saves the analysis results list to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"[INFO] ✅ Analysis results saved to {output_path}")

def save_visualizations(analysis_results, base_output_folder="visualizations"):
    """
    Generates and saves images with bounding boxes.
    """
    Path(base_output_folder).mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(analysis_results, desc="[Stage 3] Saving visualizations")
    for frame_idx, result in enumerate(pbar):
        img_path = result['image_path']
        detections = result['detections']
        
        if not detections:
            continue

        img = cv2.imread(img_path)
        
        for detection in detections:
            if not isinstance(detection.get('bounding_box'), list) or len(detection['bounding_box']) != 4:
                print(f"[WARNING] Skipping malformed bounding box for {img_path}")
                continue

            x1, y1, x2, y2 = detection['bounding_box']
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)
            count = detection.get('count', 1)
            object_id = detection.get('object_id', 'N/A')
            
            display_text = f'ID:{object_id} {label} (x{count})' if count > 1 else f'ID:{object_id} {label}'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        output_path_boxes = Path(base_output_folder) / f"frame_{frame_idx:04d}_boxes.jpg"
        cv2.imwrite(str(output_path_boxes), img)
    
    print(f"[INFO] ✅ Visualizations saved to '{base_output_folder}'")

def process_video_pipeline(video_source, api='yolo', keyframe_strategy='hybrid', frame_interval_seconds=1, confidence_threshold=0.7, nms_iou_threshold=0.5):
    print(f"[INFO] ▶ Starting pipeline for: {video_source}")
    print(f"[INFO] ▶ Using keyframe strategy: {keyframe_strategy}")
    print(f"[INFO] ▶ Analysis API: {api}")
    
    video_path = download_video(video_source)
    
    keyframes_folder = "keyframes"
    Path(keyframes_folder).mkdir(exist_ok=True)
    
    num_frames_extracted = 0
    if keyframe_strategy == 'scene_detect':
        keyframes_folder, num_frames_extracted = extract_keyframes_pyscenedetect(video_path, keyframes_folder)
    elif keyframe_strategy == 'interval':
        keyframes_folder, num_frames_extracted = extract_keyframes_by_interval(video_path, keyframes_folder, interval_seconds=frame_interval_seconds)
    elif keyframe_strategy == 'hybrid':
        keyframes_folder, num_frames_extracted = extract_keyframes_pyscenedetect(video_path, keyframes_folder)
        if num_frames_extracted == 0:
            print("[INFO] Scene detection failed. Falling back to interval extraction.")
            keyframes_folder, num_frames_extracted = extract_keyframes_by_interval(video_path, keyframes_folder, interval_seconds=frame_interval_seconds)
    
    if num_frames_extracted > 0:
        print(f"\n[Stage 2] Starting analysis with {api}...")
        
        analysis_results = []
        if api == 'yolo':
            analysis_results = analyze_keyframes_with_yolo(
                keyframes_folder,
                confidence_threshold=confidence_threshold,
                nms_iou_threshold=nms_iou_threshold
            )
        elif api == 'google_cloud':
            analysis_results = analyze_keyframes_with_google_cloud(
                keyframes_folder,
                confidence_threshold=confidence_threshold
            )
        elif api == 'openai':
            analysis_results = analyze_keyframes_with_openai(keyframes_folder)
        elif api == 'hybrid':
            analysis_results = analyze_keyframes_with_hybrid_yolo_openai(keyframes_folder)
        else:
            raise ValueError("Invalid API specified. Use 'yolo', 'google_cloud', 'openai', or 'hybrid'.")
        
        # Track objects and get final summary
        print("\n[Stage 3] Tracking objects and generating final summary...")
        tracked_objects = track_objects_across_frames(analysis_results)
        final_summary = get_final_object_summary(tracked_objects)

        # Update the analysis_results with the object_id for visualizations
        for frame_data in analysis_results:
            for det in frame_data['detections']:
                for obj_id, detections in tracked_objects.items():
                    # Check if this detection belongs to this tracked object
                    if any(d['label'] == det['label'] and d['bounding_box'] == det['bounding_box'] for d in detections):
                        det['object_id'] = obj_id
                        break

        # Save all results
        save_results_as_json(analysis_results, Path("analysis_results.json"))
        save_visualizations(analysis_results)
        save_results_as_json(final_summary, Path("final_object_summary.json"))
        
        print("\n[INFO] Final Object Summary:")
        for item in final_summary:
            print(f"  - ID: {item['id']}, Label: {item['label']}, Count: {item['count']}")
            
    else:
        print("[INFO] No keyframes were extracted. Cannot proceed with analysis.")

    print(f"\n✅ Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full video -> Keyframe extraction -> Analysis pipeline.")
    parser.add_argument("video_source", help="URL or path to video file")
    parser.add_argument("--api", type=str, choices=['yolo', 'google_cloud', 'openai', 'hybrid'], default='yolo',
                        help="Choose the analysis API: 'yolo' (local), 'google_cloud' (cloud-based), 'openai' (cloud-based), or 'hybrid' (YOLO+OpenAI).")
    parser.add_argument("--keyframe_strategy", type=str, choices=['hybrid', 'scene_detect', 'interval'], default='hybrid',
                        help="Choose how to extract keyframes.")
    parser.add_argument("--frame_interval_seconds", type=float, default=1.0,
                        help="Interval in seconds for 'interval' keyframe extraction strategy.")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                        help="Confidence threshold for filtering detections. Used for YOLO and Google Cloud only.")
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5,
                        help="IoU (Intersection over Union) threshold for Non-Maximum Suppression (YOLO only).")
    args = parser.parse_args()
    process_video_pipeline(args.video_source, args.api, args.keyframe_strategy, args.frame_interval_seconds, args.confidence_threshold, args.nms_iou_threshold)