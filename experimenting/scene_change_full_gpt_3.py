# check if we can get individual progress bars
# check if time is indeed saved by multithreading
# ignore pdf videos count in main progress bar
# check if cpu usage increases with more threads thereby reducing time
# provide resume support by saving frame number as scene number, pulling the latest frame number if the folder exists.

import cv2
from PIL import Image
import imagehash
import os
import shutil
import img2pdf

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore

# Configuration
frame_interval = 60
weights = {"aHash": 0.2, "dHash": 0.3, "pHash": 0.4, "wHash": 0.1}
video_extensions = (".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm")

hash_funcs = {
    "aHash": imagehash.average_hash,
    "dHash": imagehash.dhash,
    "pHash": imagehash.phash,
    "wHash": imagehash.whash
}

def process_video(video_path: Path, similarity_threshold: float, semaphore: Semaphore):
    with semaphore:
        try:
            output_pdf = video_path.with_suffix(".pdf")
            if output_pdf.exists():
                return f"⏩ {video_path.name}: Skipped (PDF already exists)."

            cap = cv2.VideoCapture(str(video_path))
            scene_count = 0
            last_scene_image = None
            last_hashes = None
            frame_index = 0

            temp_dir = video_path.parent / (video_path.stem + "_scenes_temp")
            os.makedirs(temp_dir, exist_ok=True)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    current_hashes = {name: func(pil_img) for name, func in hash_funcs.items()}

                    if last_hashes is not None:
                        similarities = {
                            name: 1 - (last_hashes[name] - current_hashes[name]) / len(last_hashes[name].hash) ** 2
                            for name in hash_funcs
                        }
                        final_similarity = sum(similarities[name] * weights[name] for name in hash_funcs)

                        if final_similarity < similarity_threshold:
                            if last_scene_image is not None:
                                scene_filename = temp_dir / f"scene_{scene_count:04d}.jpg"
                                cv2.imwrite(str(scene_filename), last_scene_image)
                            scene_count += 1

                    last_scene_image = frame.copy()
                    last_hashes = current_hashes

                frame_index += 1

            cap.release()

            image_files = sorted(temp_dir.glob("*.jpg"), key=os.path.getmtime)
            if image_files:
                with open(output_pdf, "wb") as f:
                    f.write(img2pdf.convert([str(img) for img in image_files]))
            shutil.rmtree(temp_dir)

            return f"✅ {video_path.name}: {scene_count} scenes → PDF created."

        except Exception as e:
            return f"❌ {video_path.name}: Error - {str(e)}"

def find_videos_recursively(root_dir):
    return [p for p in Path(root_dir).rglob("*") if p.suffix.lower() in video_extensions]

def main():
    root_dir = input("Enter the root folder path containing videos: ").strip()
    if not os.path.isdir(root_dir):
        print("Folder does not exist. Exiting.")
        return

    try:
        thread_limit = int(input("Max concurrent videos to process [default 4]: ").strip() or "4")
    except ValueError:
        thread_limit = 4

    try:
        similarity_threshold = float(input("Enter similarity threshold [default 0.90]: ").strip() or "0.90")
    except ValueError:
        similarity_threshold = 0.90

    videos = find_videos_recursively(root_dir)
    if not videos:
        print("No video files found.")
        return

    print(f"\nFound {len(videos)} video(s). Starting processing with {thread_limit} worker(s)...\n")

    results = []
    semaphore = Semaphore(thread_limit)

    with ThreadPoolExecutor(max_workers=thread_limit) as executor:
        futures = {executor.submit(process_video, video, similarity_threshold, semaphore): video for video in videos}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            results.append(result)

    print("\nSummary:")
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
