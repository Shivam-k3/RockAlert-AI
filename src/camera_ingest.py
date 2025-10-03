"""
Camera Data Ingestion Module
Extracts frames from video or webcam at specified intervals and saves with timestamps.
"""

import cv2
import os
import time
from datetime import datetime
import argparse
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraIngest:
    def __init__(self, data_dir="data/raw", frame_interval=5):
        """
        Initialize camera ingestion system.
        
        Args:
            data_dir (str): Directory to save extracted frames
            frame_interval (int): Interval in seconds between frame captures
        """
        self.data_dir = data_dir
        self.frame_interval = frame_interval
        self.frames_dir = os.path.join(data_dir, "frames")
        self.metadata_file = os.path.join(data_dir, "frame_metadata.json")
        
        # Create directories if they don't exist
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Initialize metadata storage
        self.metadata = []
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
    
    def save_metadata(self):
        """Save frame metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def extract_from_webcam(self, camera_id=0, duration_minutes=10):
        """
        Extract frames from webcam for specified duration.
        
        Args:
            camera_id (int): Camera device ID (0 for default)
            duration_minutes (int): Duration to capture in minutes
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Cannot open camera {camera_id}")
            return False
        
        logger.info(f"Starting webcam capture for {duration_minutes} minutes")
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        last_capture = 0
        
        try:
            while time.time() < end_time:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                current_time = time.time()
                
                # Check if it's time to save a frame
                if current_time - last_capture >= self.frame_interval:
                    timestamp = datetime.now()
                    filename = f"frame_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    filepath = os.path.join(self.frames_dir, filename)
                    
                    # Save frame
                    cv2.imwrite(filepath, frame)
                    
                    # Save metadata
                    frame_metadata = {
                        "filename": filename,
                        "timestamp": timestamp.isoformat(),
                        "source": "webcam",
                        "camera_id": camera_id,
                        "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0]
                    }
                    self.metadata.append(frame_metadata)
                    
                    logger.info(f"Captured frame: {filename}")
                    last_capture = current_time
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
        
        finally:
            cap.release()
            self.save_metadata()
            logger.info("Webcam capture completed")
        
        return True
    
    def extract_from_video(self, video_path):
        """
        Extract frames from video file at specified intervals.
        
        Args:
            video_path (str): Path to video file
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(fps * self.frame_interval)
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}, Frame step: {frame_step}")
        
        frame_count = 0
        saved_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame at specified intervals
                if frame_count % frame_step == 0:
                    timestamp = datetime.now()
                    filename = f"video_frame_{saved_count:06d}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    filepath = os.path.join(self.frames_dir, filename)
                    
                    # Save frame
                    cv2.imwrite(filepath, frame)
                    
                    # Calculate video timestamp
                    video_timestamp = frame_count / fps
                    
                    # Save metadata
                    frame_metadata = {
                        "filename": filename,
                        "timestamp": timestamp.isoformat(),
                        "source": "video",
                        "video_path": video_path,
                        "video_timestamp": video_timestamp,
                        "frame_number": frame_count,
                        "frame_width": frame.shape[1],
                        "frame_height": frame.shape[0]
                    }
                    self.metadata.append(frame_metadata)
                    
                    logger.info(f"Extracted frame {saved_count}: {filename}")
                    saved_count += 1
                
                frame_count += 1
        
        finally:
            cap.release()
            self.save_metadata()
            logger.info(f"Video processing completed. Extracted {saved_count} frames")
        
        return True
    
    def get_latest_frames(self, count=10):
        """
        Get metadata for the latest captured frames.
        
        Args:
            count (int): Number of latest frames to return
        
        Returns:
            list: Latest frame metadata
        """
        return self.metadata[-count:] if len(self.metadata) >= count else self.metadata


def main():
    parser = argparse.ArgumentParser(description="Camera Data Ingestion")
    parser.add_argument("--source", choices=["webcam", "video"], required=True,
                       help="Data source: webcam or video file")
    parser.add_argument("--video-path", type=str, help="Path to video file (if source=video)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID (if source=webcam)")
    parser.add_argument("--duration", type=int, default=10, help="Capture duration in minutes (webcam only)")
    parser.add_argument("--interval", type=int, default=5, help="Frame capture interval in seconds")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    
    args = parser.parse_args()
    
    # Create camera ingestion instance
    camera_ingest = CameraIngest(data_dir=args.data_dir, frame_interval=args.interval)
    
    if args.source == "webcam":
        success = camera_ingest.extract_from_webcam(
            camera_id=args.camera_id,
            duration_minutes=args.duration
        )
    elif args.source == "video":
        if not args.video_path:
            logger.error("Video path is required when source=video")
            return
        success = camera_ingest.extract_from_video(args.video_path)
    
    if success:
        logger.info("Camera ingestion completed successfully")
    else:
        logger.error("Camera ingestion failed")


if __name__ == "__main__":
    main()