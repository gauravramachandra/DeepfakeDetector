import os
import cv2

def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extract frames from a video at a specified frame rate.
    Skips already extracted frames to resume from where it stopped.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Calculate expected number of extracted frames (approx.)
    expected_extracted = total_frames // frame_rate + (1 if total_frames % frame_rate != 0 else 0)

    # Get last extracted frame number
    existing_frames = sorted([int(f.split('_')[1].split('.')[0]) 
                              for f in os.listdir(output_folder) 
                              if f.startswith("frame_") and f.endswith(".jpg")])
    last_extracted_frame = existing_frames[-1] if existing_frames else -1  # -1 means no frames exist
    current_count = len(existing_frames)

    # Check if extraction is complete (using 90% threshold)
    if current_count >= 0.9 * expected_extracted:
        print(f"Skipping {os.path.basename(video_path)}, already processed.")
        cap.release()
        return

    print(f"Resuming extraction from frame {last_extracted_frame + 1} for {os.path.basename(video_path)}")
    frame_count = 0
    success, image = cap.read()

    while success:
        if frame_count > last_extracted_frame and frame_count % frame_rate == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(filename, image)

        success, image = cap.read()
        frame_count += 1

    cap.release()
    print(f"Frames extracted to {output_folder}")

def process_all_videos(input_folder, output_base_folder, frame_rate=5):
    """
    Process all videos in the input folder and extract frames.
    Skips fully processed videos and resumes partially processed ones.
    """
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    for video_file in os.listdir(input_folder):
        if video_file.endswith((".mp4", ".avi", ".mov")):  # Add other formats if needed
            video_path = os.path.join(input_folder, video_file)
            video_name = os.path.splitext(video_file)[0]  # Remove extension
            output_folder = os.path.join(output_base_folder, video_name)

            print(f"Processing {video_file}...")
            extract_frames(video_path, output_folder, frame_rate)

if __name__ == "__main__":
    # Process Real Videos
    process_all_videos(
        r"C:\Users\gau68\Downloads\FF++\real",
        r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\extracted_frames\real"
    )

    # Process Fake Videos
    process_all_videos(
        r"C:\Users\gau68\Downloads\FF++\fake",
        r"C:\Users\gau68\DeepfakeDetector\data_preprocessing\extracted_frames\fake"
    )
