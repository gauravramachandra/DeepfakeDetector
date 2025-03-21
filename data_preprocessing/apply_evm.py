import os
import cv2
import numpy as np
import scipy.fftpack as fftpack

def apply_evm(video_path, output_path, alpha=50, fl=0.4, fh=3.0, sampling_rate=30, batch_size=100):
    """
    Apply Eulerian Video Magnification (EVM) on a video using batch processing.
    This reduces memory usage by processing frames in smaller chunks.
    
    Parameters:
      video_path (str): Path to the input video.
      output_path (str): Path to save the processed (EVM output) video.
      alpha (float): Amplification factor.
      fl (float): Low cutoff frequency.
      fh (float): High cutoff frequency.
      sampling_rate (int): Frame rate of the video.
      batch_size (int): Number of frames to process at once.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get video dimensions and initialize VideoWriter using mp4v codec
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, sampling_rate, (width, height))

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_count += 1

        # Process in batches
        if len(frames) >= batch_size:
            process_batch(frames, out, alpha, fl, fh, sampling_rate)
            frames.clear()  # Free memory

    # Process remaining frames, if any
    if frames:
        process_batch(frames, out, alpha, fl, fh, sampling_rate)

    cap.release()
    out.release()
    print(f"EVM applied for {video_path} and saved to {output_path}")

def process_batch(frames, out, alpha, fl, fh, sampling_rate):
    """
    Process a batch of frames using FFT-based filtering.
    """
    frames = np.array(frames, dtype=np.float32)

    # Compute FFT along the time axis (axis=0 is time)
    fft_frames = fftpack.fft(frames, axis=0)
    frequencies = fftpack.fftfreq(frames.shape[0], d=1.0/sampling_rate)
    
    # Create a bandpass filter mask for frequencies between fl and fh
    mask = (frequencies >= fl) & (frequencies <= fh)
    fft_frames[~mask] = 0

    # Inverse FFT to get filtered frames and amplify
    filtered_frames = fftpack.ifft(fft_frames, axis=0).real
    amplified_frames = frames + alpha * filtered_frames
    amplified_frames = np.clip(amplified_frames, 0, 255).astype(np.uint8)

    # Write processed frames to output video
    for frame in amplified_frames:
        out.write(frame)

def is_video_valid(video_path):
    """
    Check if a video file is valid (not corrupted or incomplete).
    Returns True if the file exists, its size is >100KB,
    and it contains more than 10 frames.
    """
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 100 * 1024:
        return False
    
    cap = cv2.VideoCapture(video_path)
    valid = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 10
    cap.release()
    return valid

def process_all_videos(input_folder, output_folder, alpha=50, fl=0.4, fh=3.0, sampling_rate=30, batch_size=100):
    """
    Process all videos in the input folder using EVM.
    Skips already processed videos (if output exists and is valid).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".mp4"):
            output_filename = f"evm_{filename}"
            output_path = os.path.join(output_folder, output_filename)

            # Skip if the output video exists and is valid
            if os.path.exists(output_path) and is_video_valid(output_path):
                print(f"Skipping {filename} (already processed and valid).")
                continue

            video_path = os.path.join(input_folder, filename)
            print(f"Processing {filename}...")
            apply_evm(video_path, output_path, alpha, fl, fh, sampling_rate, batch_size)

if __name__ == "__main__":
    # Process Real Videos
    real_input = r"C:\Users\gau68\Downloads\FF++\real"
    real_output = r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\evm_output\real"
    process_all_videos(real_input, real_output)

    # Process Fake Videos
    fake_input = r"C:\Users\gau68\Downloads\FF++\fake"
    fake_output = r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\evm_output\fake"
    process_all_videos(fake_input, fake_output)
