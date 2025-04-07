import cv2
import numpy as np
from collections import deque

class LaneDetector:
    def __init__(self):
        self.left_history = deque(maxlen=10)  
        self.right_history = deque(maxlen=10) 
        self.width = True
        self.height = None

    def region_of_interest(self, img):
        """Define a region of interest (ROI) to focus on the road lanes."""
        height, width = img.shape
        mask = np.zeros_like(img)
        vertices = np.array([[
            (width * 0.2, height),         # Bottom-left
            (width * 0.45, height * 0.6),  # Top-left
            (width * 0.55, height * 0.6),  # Top-right
            (width * 0.9, height)          # Bottom-right
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        masked_img = cv2.bitwise_and(img, mask)
        print(f"region_of_interest: Created trapezoidal mask with vertices {vertices.tolist()} and applied it. Shape: {masked_img.shape}")
        return masked_img

    def smooth_lines(self, lines):
        """Smooth lines over time using historical data."""
        if not lines:
            print("smooth_lines: No lines to smooth.")
            return []
        
        current_left = []
        current_right = []
        
        for x1, y1, x2, y2 in lines:
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            if slope < -0.5:  # Left lane
                current_left.append((x1, y1, x2, y2))
            elif slope > 0.5:  # Right lane
                current_right.append((x1, y1, x2, y2))
        
        # Update history with the longest line per side
        if current_left:
            self.left_history.append(max(current_left, key=lambda l: np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)))
        if current_right:
            self.right_history.append(max(current_right, key=lambda l: np.sqrt((l[2] - l[0])**2 + (l[3] - l[1])**2)))
        
        # Average lines from history
        def average_line(history):
            if not history:
                return None
            avg = np.mean(history, axis=0).astype(int)
            x1, y1, x2, y2 = avg
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            intercept = y1 - slope * x1 if slope != float('inf') else None
            if intercept is not None:
                y_bottom = self.height
                y_top = int(self.height * 0.6)
                x_bottom = int((y_bottom - intercept) / slope)
                x_top = int((y_top - intercept) / slope)
                x_bottom = max(0, min(x_bottom, self.width - 1))
                x_top = max(0, min(x_top, self.width - 1))
                return (x_bottom, y_bottom, x_top, y_top)
            return (x1, y1, x2, y2)
        
        smoothed_left = average_line(self.left_history)
        smoothed_right = average_line(self.right_history)
        smoothed = [l for l in [smoothed_left, smoothed_right] if l is not None]
        print(f"smooth_lines: Smoothed {len(lines)} lines into {len(smoothed)} stable lines. Left history: {len(self.left_history)}, Right history: {len(self.right_history)}")
        return smoothed

    def process_frame(self, frame):
        """Process a single frame to detect and draw stable lane lines, showing output at each step."""
        self.height, self.width = frame.shape[:2]
        
        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Step 2: Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Edge detection using Canny
        edges = cv2.Canny(blur, 50, 150)
        print(f"process_frame: Detected edges using Canny (low=50, high=150). Shape: {edges.shape}")
        cv2.imshow('Edges', edges)
        
        # Step 4: Apply region of interest
        masked_edges = self.region_of_interest(edges)
        print(f"process_frame: Applied ROI to edges.")
        cv2.imshow('Masked Edges', masked_edges)
        
        # Step 5: Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=30,
            minLineLength=50,
            maxLineGap=50
        )
        print(f"process_frame: Ran Hough Transform. Detected {len(lines) if lines is not None else 0} lines.")
        
        # Step 6: Draw raw Hough lines (for reference)
        raw_line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(raw_line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for raw lines
        raw_result = cv2.addWeighted(frame, 0.8, raw_line_image, 1, 0)
        cv2.imshow('Raw Hough Lines', raw_result)
        print(f"process_frame: Drew {len(lines) if lines is not None else 0} raw Hough lines in red.")
        
        # Step 7: Smooth and filter lines
        line_list = [line[0] for line in lines] if lines is not None else []
        stable_lines = self.smooth_lines(line_list)
        
        # Step 8: Draw stable lines
        line_image = np.zeros_like(frame)
        for i, line in enumerate(stable_lines):
            x1, y1, x2, y2 = line
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
            print(f"process_frame: Drew stable line {i+1}: from ({x1}, {y1}) to ({x2}, {y2})")
        
        # Step 9: Overlay stable lines on original frame
        result = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
        print(f"process_frame: Overlaid {len(stable_lines)} stable lines on the original frame.")
        return result

def main():
    """Handle video capture and display."""
    detector = LaneDetector()
    video_path = 'videos/test_video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"main: Error - Could not open video file at {video_path}. Check the file path.")
        return
    
    print("main: Video opened successfully. Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("main: End of video reached.")
            break
        
        # Process the frame and detect lanes
        processed_frame = detector.process_frame(frame)
        print(f"main: Processed frame with shape {processed_frame.shape}")
        
        # Display the final processed frame
        cv2.imshow('Final Lane Detection', processed_frame)
        print("main: Displayed final processed frame in window 'Final Lane Detection'")
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("main: Stopped by user pressing 'q'.")
            break
    
    cap.release()
    print("main: Released video capture object.")
    cv2.destroyAllWindows()
    print("main: Closed all OpenCV windows.")

if __name__ == "__main__":
    main()