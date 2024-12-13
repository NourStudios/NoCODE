import cv2
import os

def create_video(image_folder, output_file, frame_duration=0.05):
    """
    Create a video from images sorted by numerical order.

    :param image_folder: Folder containing numbered image files.
    :param output_file: Path to save the output video file.
    :param frame_duration: Duration each image is displayed in seconds (default 0.2s).
    """
    # List and sort image files numerically
    images = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(os.path.splitext(x)[0])
    )
    
    if not images:
        print("No images found in the folder.")
        return

    # Read the first image to get the frame dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Initialize the video writer
    fps = int(1 / frame_duration)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output codec
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Add each image to the video
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Warning: Could not read image {image_name}, skipping.")
            continue
        video.write(frame)

    # Release the video writer
    video.release()
    print(f"Video saved as {output_file}")

# Example usage
create_video(
    image_folder="generated",  # Replace with your folder path
    output_file="falltest.mp4"  # Replace with desired output file name
)
