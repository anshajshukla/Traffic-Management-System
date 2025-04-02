''' packages required 
1) pip install opencv-contrib-python it contains extra modules like SIFT, SURF, etc. 
2) pip install caer  # for helper function that speed up the workflow

'''

# Reading images in OpenCV
import cv2 as cv

# Read the image
img = cv.imread("photos/cat.jpg")

# Check if image was loaded successfully
if img is None:
    print("Error: Could not read the image. Please check if 'photos/cat.jpg' exists.")
else:
    # Display image in a window
    cv.imshow('cat', img)
    cv.waitKey(0)         # wait for key press
    cv.destroyAllWindows()  # properly close windows
