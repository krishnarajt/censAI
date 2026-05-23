from tqdm import tqdm
import time

progress = tqdm(total=100, desc="Processing", dynamic_ncols=True)

for i in range(10):
    time.sleep(0.5)  # Simulate work
    tqdm.write(f"Step {i+1}/10: Processing chunk {i}")  # Print log above progress bar
    progress.update(10)

progress.close()
