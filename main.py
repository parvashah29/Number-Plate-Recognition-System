import os
import argparse
from license_plate_detection import LicensePlateDetector

def main():
    parser = argparse.ArgumentParser(description='Number Plate Recognition System')
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam or path to video file')
    args = parser.parse_args()

    # Initialize detector with the default model path
    detector = LicensePlateDetector()

    # List available test videos
    video_dir = "Testing Videos"
    if os.path.exists(video_dir):
        # Include images and videos
        videos = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.jpg', '.jpeg', '.png'))]
        if videos:
            print("\nAvailable test files:")
            for i, video in enumerate(videos, 1):
                print(f"{i}. {video}")
            print(f"{len(videos)+1}. Webcam (0)")
            
            choice = input("\nEnter file number: ").strip()
            
            if choice and choice.isdigit():
                choice_idx = int(choice)
                if 1 <= choice_idx <= len(videos):
                    video_path = os.path.join(video_dir, videos[choice_idx-1])
                    print(f"\nProcessing file: {videos[choice_idx-1]}")
                    detector.process_video(video_path)
                elif choice_idx == len(videos) + 1:
                    print("\nStarting webcam feed...")
                    detector.process_video('0')
                else:
                    print("\nInvalid choice. Exiting.")
            else:
                print("\nInvalid input. Exiting.")
        else:
            print("\nNo test files found in 'Testing Videos'. Starting webcam feed...")
            detector.process_video('0')
    else:
        print("\n'Testing Videos' directory not found. Starting webcam feed...")
        detector.process_video('0')

if __name__ == "__main__":
    main()
