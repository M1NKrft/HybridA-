import cv2
import numpy as np


# Load the image in grayscale
# Apply Gaussian Blur to reduce noise and improve contour detection
image = cv2.imread(img_path, 0)
# Apply Gaussian Blur to reduce noise and improve contour detection
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# Threshold the image to create a binary image (black and white)
_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours in the thresholded image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank mask to draw the filled area between contours
mask = np.zeros_like(image)

# Check if at least two contours are found
if len(contours) > 1:
    # Sort the contours by area (largest to smallest)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assume the largest contour is the outer curve and the second largest is the inner curve
    outer_contour = contours[0]
    inner_contour = contours[1]

    # Draw the outer contour on the mask (filled)
    cv2.drawContours(mask, [outer_contour], -1, 255, thickness=cv2.FILLED)

    # Draw the inner contour on the mask (subtracting it)
    cv2.drawContours(mask, [inner_contour], -1, 0, thickness=cv2.FILLED)

    # Now the mask has the region between the two curves filled with white
    output = mask

    # Save or display the output image
    cv2.imwrite('Spielberg_map_filled_path.png', output)
    cv2.imshow('Filled Path Region', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Could not find both inner and outer contours.")
    
    # Optional: visualize detected contours on the original image for debugging
    contour_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Detected Contours', contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()