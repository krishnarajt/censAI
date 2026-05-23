import cv2
from PIL import Image
import imagehash
import os

# Parameters
video_path = "2024-01-17 21-19-54-done.mp4"  # Change this to your video file
output_dir = "scene_changes"  # Directory to save scene change images
frame_interval = 60  # Check every nth frame
similarity_threshold = 0.90  # Scene change threshold higher means more similar 1 is same, lower is different which is what we want

# Hash function weights (adjust if needed)
weights = {"aHash": 0.2, "dHash": 0.3, "pHash": 0.4, "wHash": 0.1}

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Open video file
cap = cv2.VideoCapture(video_path)

scene_count = 0
last_scene_image = None
last_hashes = None

frame_index = 0

# Define hash functions
hash_funcs = {
    "aHash": imagehash.average_hash,
    "dHash": imagehash.dhash,
    "pHash": imagehash.phash,
    "wHash": imagehash.whash
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process only every 5th frame
    if frame_index % frame_interval == 0:
        
        # Convert frame to PIL image
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Compute all hash values
        current_hashes = {name: func(pil_img) for name, func in hash_funcs.items()}

        # Compare with previous frame's hash
        if last_hashes is not None:
            similarities = {
                name: 1 - (last_hashes[name] - current_hashes[name]) / len(last_hashes[name].hash) ** 2
                for name in hash_funcs
            }

            # Compute weighted similarity score
            final_similarity = sum(similarities[name] * weights[name] for name in hash_funcs)
            print(f"Frame {frame_index}: Similarity scores: {similarities}, Final similarity: {final_similarity:.2f}")
            if final_similarity < similarity_threshold:
                # Scene change detected, save last frame
                if last_scene_image is not None:
                    scene_filename = os.path.join(output_dir, f"scene_{scene_count}.jpg")
                    cv2.imwrite(scene_filename, last_scene_image)
                    print(f"Scene {scene_count} saved: {scene_filename}")

                # Increase scene count
                scene_count += 1

        # Update last scene image and hashes
        last_scene_image = frame.copy()
        last_hashes = current_hashes

    frame_index += 1

cap.release()
print(f"Total scenes detected: {scene_count}")
