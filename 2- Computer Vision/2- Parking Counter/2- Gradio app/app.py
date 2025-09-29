import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import os
import tempfile


class ParkingDetector:
    def __init__(self, model_path="/yolo8s_parklot.pt"):
        """
        Initialize parking detector with model path
        """
        try:
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
            print(f"Model classes: {self.model.names}")
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def process_frame(self, frame):
        """
        Process a single frame to detect parking occupancy
        For a model that detects 'car' and 'free' spaces directly
        """
        # Run YOLO inference
        results = self.model(frame, verbose=False)[0]

        if results.boxes is None or len(results.boxes) == 0:
            return frame, 0, 0

        boxes = results.boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confidences = boxes.conf.cpu().numpy()

        # Count detections by class
        car_count = 0
        free_count = 0
        car_boxes = []
        free_boxes = []

        for i, (box, cls, conf) in enumerate(zip(coords, classes, confidences)):
            if conf < 0.5:  # Filter low confidence detections
                continue

            class_name = results.names[cls]
            x1, y1, x2, y2 = map(int, box)

            if class_name == "car":
                car_count += 1
                car_boxes.append((x1, y1, x2, y2))
            elif class_name == "free":
                free_count += 1
                free_boxes.append((x1, y1, x2, y2))

        return self.visualize_detections(frame, car_boxes, free_boxes, car_count, free_count)

    def visualize_detections(self, frame, car_boxes, free_boxes, car_count, free_count):
        """
        Draw detections and calculate statistics
        """
        # Draw free spaces in green
        for (x1, y1, x2, y2) in free_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Free", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw Vehicle in blue
        for (x1, y1, x2, y2) in car_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Vehicle", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Calculate statistics
        # Method 1: Assume total spaces = Vehicle in parking lot + free spaces
        total_spaces = car_count + free_count
        occupied_spaces = car_count  # All detected Vehicle are in parking spaces
        free_spaces = free_count

        return frame, occupied_spaces, total_spaces

    def add_info_overlay(self, frame, occupied, total, frame_num=None):
        """
        Add information overlay to frame
        """
        # Create semi-transparent overlay for better text visibility
        overlay = frame.copy()

        # Background rectangle for stats
        cv2.rectangle(overlay, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Statistics text
        free = total - occupied
        occupancy_rate = (occupied / total * 100) if total > 0 else 0

        stats = [
            f"Total Spaces: {total}",
            f"Occupied: {occupied} (Vehicle detected)",
            f"Free: {free} (free spaces detected)",
            f"Occupancy Rate: {occupancy_rate:.1f}%"
        ]

        if frame_num is not None:
            stats.append(f"Frame: {frame_num}")

        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (20, 35 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 135, 66), 2)

        return frame


def process_video(video_path):
    """
    Process video file for parking detection
    """
    if video_path is None:
        return None, "Please upload a video file"

    try:
        # Initialize detector
        detector = ParkingDetector()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error: Could not open video file"

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create temporary output file
        temp_dir = tempfile.gettempdir()
        out_path = os.path.join(temp_dir, "parking_detection_output.mp4")

        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_num = 0
        occupancy_history = []

        # Process video frame by frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame - returns occupied and total counts
            processed_frame, occupied, total = detector.process_frame(frame)

            # Add info overlay
            processed_frame = detector.add_info_overlay(
                processed_frame, occupied, total, frame_num)

            # Track statistics
            if total > 0:
                occupancy_rate = occupied / total
                occupancy_history.append(occupancy_rate)

            out.write(processed_frame)
            frame_num += 1

            # Progress indicator
            if frame_num % 30 == 0:
                print(f"Processing frame {frame_num}/{total_frames}")

        # Calculate final statistics
        avg_occupancy = np.mean(occupancy_history) * \
            100 if occupancy_history else 0
        max_occupancy = max(occupancy_history) * \
            100 if occupancy_history else 0

        # Cleanup
        cap.release()
        out.release()

        # Create summary message
        summary = f"""
        Processing Complete!
        
        Detection Logic:
        - Vehicle detected = Occupied spaces (parked vehicle)
        - Free spaces = Empty spaces detected by model
        - Total spaces = Vehicle + Free spaces
        
        Video Statistics:
        - Total Frames Processed: {frame_num}
        - Average Occupancy: {avg_occupancy:.1f}%
        - Peak Occupancy: {max_occupancy:.1f}%
        
        Note: This assumes all detected vehicle are in parking spaces.
        """

        return out_path, summary

    except Exception as e:
        return None, f"Error processing video: {str(e)}"

# Enhanced Gradio interface with both approaches


def create_interface():
    """Create Gradio interface with both detection approaches"""

    with gr.Blocks(title="Parking Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 Parking Space Detection")
        gr.Markdown("""
        **Model Classes:** `car` (parked vehicles) and `free` (empty spaces)
        
        **Logic:** 
        - Occupied = vehicles detected (assuming all vehicles are parked)
        - Free = free spaces detected by model
        - Total = occupied + free
        """)

        with gr.Tabs():
            with gr.TabItem("Dynamic Total (Cars + Free)"):
                with gr.Row():
                    with gr.Column():
                        video_input1 = gr.Video(label="Upload Video")
                        process_btn1 = gr.Button(
                            "🔍 Process", variant="primary")
                    with gr.Column():
                        video_output1 = gr.Video(label="Processed Video")
                        status_output1 = gr.Textbox(label="Status", lines=10)

                process_btn1.click(
                    fn=process_video,
                    inputs=[video_input1],
                    outputs=[video_output1, status_output1]
                )

        gr.Markdown("""
        
        **Dynamic Total (Vehicles + Free):**
        - Good for varying occupancy scenarios
        - Total spaces change based on what's detected
        - May undercount if some vehicles aren't detected
        
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
