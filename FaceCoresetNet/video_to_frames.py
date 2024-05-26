

#input: input directory containing video files.
#output: output directgory containing each video frames in a different directory



import cv2
import os
import argparse

def extract_frames(input_dir, output_dir):
    # List all video files in the input directory
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]

    # Iterate over each video file
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]

        # Create a subdirectory with the same name as the video file
        output_subdir = os.path.join(output_dir, video_name)
        os.makedirs(output_subdir, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Initialize a frame counter
        frame_count = 0

        while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break

            # Construct the output file name
            frame_filename = os.path.join(output_subdir, f'frame_{frame_count:04d}.png')

            # Save the frame as an image file
            cv2.imwrite(frame_filename, frame)

            frame_count += 1

        # Release the video capture object
        cap.release()

        print(f'Extracted {frame_count} frames from {video_file} to {output_subdir}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Define the two arguments
    parser.add_argument('--input_directory', help="The directory containing videos")
    parser.add_argument('--output_directory', help="Where to place the output")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the values of the arguments
    input_directory = args.input_directory
    output_directory = args.output_directory

    extract_frames(input_directory, output_directory)
