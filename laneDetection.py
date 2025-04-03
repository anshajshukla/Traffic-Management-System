import cv2 # Import OpenCV library for image and video processing
import numpy as np

# # Open a video file or a camera stream
# cap  = cv2.VideoCapture('videos\Cars, Busy Streets, City Traffic - No Copyright Royalty Free Stock Videos.mp4') 

# # Opening a video 
# if not cap.isOpened():
#     print("Error: Could not open video.")
#     exit()
# while True:
#     # Capture the video frame by frame
#     '''
#     ret: boolean value, True if the frame is captured correctly, False otherwise.
#     frame: the captured frame from the video stream stored as a numpy array.
#     The frame is a 3D array representing the image in BGR format (Blue, Green, Red).
#     '''
#     ret, frame = cap.read()
#     # if no frame is read, end of the video processing 
#     if not ret:
#         print("End of video or error reading frame.")
#         break

#     # Display the frame in a window named 'Video'
#     cv2.imshow('Video Frame', frame)

#     # Wait for 1 ms and check if the 'q' key is pressed to exit the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):#ASCII of q is 113
#         break

#     '''
#     cv2.waitKey(1): This function waits for a key event for 1 millisecond.
#     If a key is pressed during this time, it returns the ASCII value of the key pressed.
#     If no key is pressed, it returns -1.
#     The bitwise AND operation with 0xFF ensures that only the last 8 bits of the key code are considered.
#     ord('q'): This function returns the ASCII value of the character 'q'.
#     If the 'q' key is pressed, the loop will break and the program will exit.
#     '''

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
# # The code captures video frames from a file or camera stream and displays them in a window.    


# Classical Lane detection 

def process_frame(frame):
    #1. Convert to Grayscale as it simplifies processing 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #2. Apply Gaussian Blur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)[1] # Thresholding to create a binary image
    ''' 
    1) Why gray? --> gray variable represents the grayscale version of the input frame.
    Many image processing tasks (like edge detection) work better on single-channel grayscale images instead of color images (which have three channels: BGR).
    2) Why Gaussian Blur? --> The Gaussian blur is applied to the grayscale image to reduce noise and detail, which helps in better edge detection.
    3) Why (5, 5)? --> The (5, 5) kernel size determines the extent of the blur. A 5x5 kernel means each pixel in the output image is influenced by a 5x5 neighborhood of pixels in the input image.
    4) Why 5x5 specifically? --> Larger kernel sizes produce stronger blurring because they take more neighboring pixels into account.
    Smaller kernels (e.g., 3x3) reduce less noise, while larger ones (e.g., 7x7) may remove details. A 5x5 kernel provides a balance between noise reduction and preserving important edges.
    5) Why 0? (sigmaX) --> The third parameter 0 is the standard deviation in the X and Y directions. If 0 is given, OpenCV automatically calculates an optimal sigma based on the kernel size. A higher sigma results in more blur.
    '''
    # 3) Canny Edge Detection to find edges in the image
    med_val = np.median(blur)
    low = int(max(0, 0.7 * med_val))
    high = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(blur, low, high)

    '''
    1) Why Canny? --> Canny edge detection is a multi-stage algorithm that detects a wide range of edges in images. It's effective for detecting edges in noisy images.
    2) Why low,high? --> These are the lower and upper thresholds for the hysteresis procedure in Canny. The lower threshold is used to identify strong edges, while the upper threshold is used to identify weak edges. The algorithm will only consider weak edges as part of an edge if they are connected to strong edges.
    3) Why 50? --> A lower threshold of 50 means that any gradient value above 50 will be considered a strong edge. This helps in detecting more edges.
    4) Why 150? --> An upper threshold of 150 means that any gradient value above 150 will be considered a strong edge. This helps in filtering out noise and weak edges.
    5) Why hysteresis? --> Hysteresis is a technique used in edge detection to track edges by suppressing noise and spurious responses. It helps in connecting weak edges to strong edges, ensuring that only significant edges are retained.
    '''
# 4) Define Region of Interest (ROI) --> This is the area of the image where we expect to find lanes. A trapezoidal shape is often used to focus on the road area.
    # Define the vertices of the polygon for the ROI 
    
    
    mask = np.zeros_like(edges)  # Create a mask with the same dimensions as the edges image  
    height, width = edges.shape[:2]

    # Define trapezoidal region of interest
    roi_corners = np.array([[
        (int(0.2 * width), height),              # Bottom-left
        (int(0.4 * width), int(0.6 * height)),  # Top-left
        (int(0.5 * width), int(0.6 * height)),  # Top-right
        (int(0.9 * width), height)               # Bottom-right
    ]], dtype=np.int32)

    cv2.fillPoly(mask, roi_corners, 255)  # Fill the polygon with white color (255)
    masked_edges = cv2.bitwise_and(edges, mask)  # Apply the mask to the edges image
    ''' 
    1) Why mask? --> The mask is used to isolate the region of interest (ROI) in the image. By applying the mask, we focus on the area where we expect to find lanes, ignoring other parts of the image.
    2) Why fillPoly? --> The fillPoly function fills the defined polygon (ROI) with a specified color (255 in this case, which is white). This creates a mask that highlights the area of interest.
    3) Why bitwise_and? --> The bitwise_and operation combines the edges image with the mask. It retains only the edges that fall within the defined ROI, effectively filtering out edges outside the ROI.
    '''
    
# 5. Apply Hough Line Transform to detect lines in the masked edges image
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=40, minLineLength=100, maxLineGap=60)
    '''
    1) Why Hough Transform? --> The Hough Transform is a technique used to detect lines in images. It works by transforming points in the image space to a parameter space, where lines can be represented as points. 
    2) Why HoughLinesP? --> HoughLinesP is a probabilistic version of the Hough Transform that detects line segments rather than full lines. It is more efficient and can handle noise better.
    3) Why 1? --> The first parameter (1) is the distance resolution of the accumulator in pixels. It determines how finely the Hough Transform samples the parameter space.
    4) Why np.pi / 180? --> This is the angle resolution of the accumulator in radians. It converts degrees to radians, allowing the function to work with angles in a more standard way.
    5) Why threshold=50? --> This is the minimum number of votes (intersections in the Hough space) required to detect a line. A higher threshold means fewer lines will be detected, while a lower threshold may detect more lines, including noise.
    6) Why minLineLength=100? --> This is the minimum length of a line segment to be considered valid. Shorter segments will be discarded. It helps in filtering out noise and short lines.
    7) Why maxLineGap=50? --> This is the maximum gap between segments to be considered as a single line. If the gap between two segments is larger than this value, they will be treated as separate lines.

    '''
    # Draw the detected lines on the original frame
    line_image = np.zeros_like(frame)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw lines in green color with thickness 5
# 6) Combine the original frame with the line image to visualize the detected lanes
    lane_overlay = cv2.addWeighted(frame, 0.8, line_image, 1, 1)  # Blend the images
    '''
    1) Why addWeighted? --> The addWeighted function is used to blend two images together. It allows for adjusting the transparency of each image in the blend.
    2) Why frame, 0.8? --> The first image (frame) is the original video frame, and 0.8 is its weight in the blend. This means the original frame will be slightly visible. 
    3) Why line_image, 1? --> The second image (line_image) is the image with detected lines, and 1 is its weight in the blend. This means the lines will be more prominent in the final output.
    4) Why 1? --> The last parameter is a scalar added to each sum. It can be used to adjust brightness, but in this case, it's set to 1 for simplicity.
    '''
    return lane_overlay


# Open a video file or a camera stream

cap  = cv2.VideoCapture('videos\Lane Detection Test Video 01.mp4') 

# Opening a video 
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
while True:
    # Capture the video frame by frame
    '''
    ret: boolean value, True if the frame is captured correctly, False otherwise.
    frame: the captured frame from the video stream stored as a numpy array.
    The frame is a 3D array representing the image in BGR format (Blue, Green, Red).
    '''
    ret, frame = cap.read()
    # if no frame is read, end of the video processing 
    if not ret:
        print("End of video or error reading frame.")
        break

    # Process the frame to detect lanes
    processed_frame = process_frame(frame)
    
    # Display the processed frame in a window named 'Lane Detection'
    cv2.imshow('Lane Detection', processed_frame)

    # Wait for 1 ms and check if the 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):#ASCII of q is 113
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
# The code captures video frames from a file or camera stream, processes each frame to detect lanes using image processing techniques, and displays the processed frames in a window.

    

