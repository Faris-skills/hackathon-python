import os
import tempfile
from pathlib import Path
import cv2
import json
import argparse
from tqdm import tqdm
from urllib.request import urlretrieve
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import base64
from openai import OpenAI
import re
from dotenv import load_dotenv
from collections import defaultdict

# Imports for Feature Extraction and Clustering
import torch
from torchvision import models, transforms
from PIL import Image
from hdbscan import HDBSCAN
import numpy as np

# New import for the hybrid solution
from thefuzz import fuzz

# Load environment variables from a .env file.
load_dotenv()

# --- HELPER FUNCTIONS ---

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

def analyze_keyframes_with_openai_vision(input_folder):
    """
    Analyzes keyframes using OpenAI's GPT-4o model.
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
        For each object, provide a detailed label (including brand names if visible), its bounding box coordinates [x1, y1, x2, y2], and the count of identical items.
        The coordinates must be in absolute pixel values, where x1, y1 are the top-left and x2, y2 are the bottom-right corners.

        Return the response as a JSON array of objects. Each object should have 'label', 'bounding_box', and 'count' keys.
        The JSON should be the only content in your response. Do not include any other text, markdown formatting, or explanation.
        Example format:
        [
          {"label": "red Solo cup", "bounding_box": [100, 200, 300, 400], "count": 1},
          {"label": "Stack of Coca-Cola cans", "bounding_box": [500, 600, 700, 800], "count": 5}
        ]
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
            
            content = response.choices[0].message.content.strip()
            
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            detections = []
            try:
                detections = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Could not parse JSON for {img_path}: {e}. Raw response: '{content[:200]}...'")
                
            frame_data = {
                "image_path": str(img_path),
                "detections": detections,
            }
            results.append(frame_data)

        except Exception as e:
            print(f"[ERROR] OpenAI API call failed for {img_path}: {e}")
            frame_data = {
                "image_path": str(img_path),
                "detections": [],
            }
            results.append(frame_data)

    print(f"[INFO] ✅ OpenAI Vision analysis complete for {len(results)} frames.")
    return results

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
    
    # Group detections by image path
    detections_by_image = defaultdict(list)
    for result in analysis_results:
        detections_by_image[result['image_path']].append(result)

    pbar = tqdm(detections_by_image.items(), desc="[Stage 5] Saving visualizations")
    for img_path, detections in pbar:
        if not detections:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        for detection in detections:
            if not isinstance(detection.get('bounding_box'), list) or len(detection['bounding_box']) != 4:
                continue

            x1, y1, x2, y2 = detection['bounding_box']
            label = detection.get('label', 'unknown')
            count = detection.get('count', 1)
            object_id = detection.get('object_id', 'N/A')
            
            display_text = f'ID:{object_id} {label} (x{count})' if count > 1 else f'ID:{object_id} {label}'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        output_path_boxes = Path(base_output_folder) / Path(img_path).name.replace('.jpg', '_boxes.jpg')
        cv2.imwrite(str(output_path_boxes), img)
    
    print(f"[INFO] ✅ Visualizations saved to '{base_output_folder}'")

class FeatureExtractor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device} for feature extraction.")
        
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def get_embedding(self, image_pil):
        """Extracts a feature vector from a PIL Image."""
        try:
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(image_tensor).squeeze()
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Failed to extract embedding: {e}")
            return None

def extract_features_from_detections(analysis_results, keyframes_folder):
    """
    Iterates through all detections, crops the objects, and extracts features.
    """
    feature_extractor = FeatureExtractor()
    
    for frame_data in tqdm(analysis_results, desc="[Stage 3] Extracting features"):
        img = cv2.imread(frame_data['image_path'])
        if img is None:
            print(f"[WARNING] Image not found: {frame_data['image_path']}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        for detection in frame_data['detections']:
            x1, y1, x2, y2 = detection['bounding_box']
            try:
                cropped_img = pil_img.crop((x1, y1, x2, y2))
                embedding = feature_extractor.get_embedding(cropped_img)
                if embedding is not None:
                    detection['embedding'] = embedding.tolist()
                    detection['image_path'] = frame_data['image_path']
            except Exception as e:
                print(f"[WARNING] Could not crop or embed object: {e}")
                detection['embedding'] = None
    
    return analysis_results

def cluster_embeddings_hdbscan(analysis_results):
    """
    Clusters all object embeddings using the HDBSCAN algorithm to de-duplicate items.
    """
    all_detections = []
    for frame_data in analysis_results:
        for det in frame_data['detections']:
            if 'embedding' in det and det['embedding'] is not None:
                all_detections.append(det)

    if not all_detections:
        print("[INFO] No objects with embeddings found for clustering.")
        return [], []
    
    embeddings = np.array([det['embedding'] for det in all_detections])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    clusterer = HDBSCAN(min_cluster_size=3, cluster_selection_epsilon=0.5)
    labels = clusterer.fit_predict(embeddings)
    
    for i, det in enumerate(all_detections):
        cluster_id = labels[i]
        det['object_id'] = f"cluster_{cluster_id}" if cluster_id != -1 else "noise"
    
    return analysis_results, all_detections


def post_process_and_summarize(all_detections, fuzzy_threshold=85):
    """
    Performs a final, text-based aggregation to merge similar objects.
    
    This function processes the output of the clustering step. It groups
    detections with similar labels (using fuzzy matching) to correct for
    duplications missed by the visual-only clustering.
    """
    final_summary_dict = {}
    
    # First, process the clustered detections (non-noise)
    for det in all_detections:
        if det.get('object_id') and det['object_id'] != "noise":
            obj_id = det['object_id']
            if obj_id not in final_summary_dict:
                final_summary_dict[obj_id] = {
                    "id": obj_id,
                    "label": det['label'],
                    "count": det['count'],
                    "detections": []
                }
            final_summary_dict[obj_id]['detections'].append(det)
            final_summary_dict[obj_id]['count'] = max(final_summary_dict[obj_id]['count'], det['count'])

    # Then, process the "noise" detections (unclustered)
    next_unique_id = 0
    noise_detections = [d for d in all_detections if d.get('object_id') == "noise"]
    for det in noise_detections:
        found_match = False
        
        # Check if the noise item's label is similar to an already-found unique object
        for obj_id, obj_data in final_summary_dict.items():
            label1 = det['label']
            label2 = obj_data['label']
            
            # Use fuzzy matching to check for high similarity
            similarity_score = fuzz.token_sort_ratio(label1, label2)
            if similarity_score >= fuzzy_threshold:
                # Merge this noise item into the existing group
                obj_data['count'] = max(obj_data['count'], det['count'])
                obj_data['detections'].append(det)
                det['object_id'] = obj_id  # Reassign the object_id
                found_match = True
                break
        
        # If no similar group was found, create a new one
        if not found_match:
            new_id = f"unique_{next_unique_id}"
            next_unique_id += 1
            det['object_id'] = new_id
            final_summary_dict[new_id] = {
                "id": new_id,
                "label": det['label'],
                "count": det['count'],
                "detections": [det]
            }

    # Generate the final summary list from the processed objects
    final_summary = []
    for obj_data in final_summary_dict.values():
        final_summary.append({
            "id": obj_data['id'],
            "label": obj_data['label'],
            "count": obj_data['count'],
        })

    return all_detections, final_summary

# --- MAIN PIPELINE EXECUTION ---

def process_video_pipeline(video_source, keyframe_strategy='hybrid', frame_interval_seconds=1):
    print(f"[INFO] ▶ Starting pipeline for: {video_source}")
    print(f"[INFO] ▶ Using keyframe strategy: {keyframe_strategy}")
    print(f"[INFO] ▶ Analysis API: OpenAI Vision")
    
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
        print(f"\n[Stage 2] Starting analysis with OpenAI Vision...")
        analysis_results = analyze_keyframes_with_openai_vision(keyframes_folder)
        
        print("\n[Stage 3] Extracting visual features for de-duplication...")
        analysis_results_with_features = extract_features_from_detections(analysis_results, keyframes_folder)
        
        print("\n[Stage 4] Clustering embeddings to de-duplicate objects...")
        # Get the flattened list of detections from the clustering step
        _, all_detections_flat = cluster_embeddings_hdbscan(analysis_results_with_features)
        
        print("\n[Stage 5] Post-processing and summarizing results...")
        final_results_flat, final_summary = post_process_and_summarize(all_detections_flat)
        
        # Save all results
        save_results_as_json(final_results_flat, Path("analysis_results.json"))
        save_visualizations(final_results_flat)
        save_results_as_json(final_summary, Path("final_object_summary.json"))
        
        print("\n[INFO] Final Object Summary:")
        for item in final_summary:
            print(f"  - ID: {item['id']}, Label: {item['label']}, Count: {item['count']}")
            
    else:
        print("[INFO] No keyframes were extracted. Cannot proceed with analysis.")

    print(f"\n✅ Pipeline complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full video -> Keyframe extraction -> Analysis pipeline using OpenAI Vision.")
    parser.add_argument("video_source", help="URL or path to video file")
    parser.add_argument("--keyframe_strategy", type=str, choices=['hybrid', 'scene_detect', 'interval'], default='hybrid',
                        help="Choose how to extract keyframes.")
    parser.add_argument("--frame_interval_seconds", type=float, default=1.0,
                        help="Interval in seconds for 'interval' keyframe extraction strategy.")
    args = parser.parse_args()
    process_video_pipeline(args.video_source, args.keyframe_strategy, args.frame_interval_seconds)