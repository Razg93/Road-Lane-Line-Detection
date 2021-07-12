import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



figsize = (10, 10)
image = cv2.imread("road.png")
LINES = []

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
     
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
   
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), 
              minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    
    draw_lines(line_img, lines)
    return line_img , lines

def draw_lines(img, lines, color=[0, 255, 0], thickness=8):

    imshape = img.shape
    
    # these variables represent the y-axis coordinates to which 
    # the line will be extrapolated to
    ymin_global = img.shape[0]
    ymax_global = img.shape[0]
    
    # left lane line variables
    all_left_grad = []
    all_left_y = []
    all_left_x = []
    
    # right lane line variables
    all_right_grad = []
    all_right_y = []
    all_right_x = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            gradient, intercept = np.polyfit((x1,x2), (y1,y2), 1)
            ymin_global = min(min(y1, y2), ymin_global)
            
            if (gradient > 0):
                all_left_grad += [gradient]
                all_left_y += [y1, y2]
                all_left_x += [x1, x2]
            else:
                all_right_grad += [gradient]
                all_right_y += [y1, y2]
                all_right_x += [x1, x2]
    
    left_mean_grad = np.mean(all_left_grad)
    left_y_mean = np.mean(all_left_y)
    left_x_mean = np.mean(all_left_x)
    left_intercept = left_y_mean - (left_mean_grad * left_x_mean)
    
    right_mean_grad = np.mean(all_right_grad)
    right_y_mean = np.mean(all_right_y)
    right_x_mean = np.mean(all_right_x)
    right_intercept = right_y_mean - (right_mean_grad * right_x_mean)
    
    # Make sure we have some points in each lane line category
    if ((len(all_left_grad) > 0) and (len(all_right_grad) > 0)):
        upper_left_x = int((ymin_global - left_intercept) / left_mean_grad)
        lower_left_x = int((ymax_global - left_intercept) / left_mean_grad)
        upper_right_x = int((ymin_global - right_intercept) / right_mean_grad)
        lower_right_x = int((ymax_global - right_intercept) / right_mean_grad)

        cv2.line(img, (upper_left_x, ymin_global), 
                      (lower_left_x, ymax_global), color, thickness)
        cv2.line(img, (upper_right_x, ymin_global), 
                      (lower_right_x, ymax_global), color, thickness)

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    # grayscale the image
    grayscaled = grayscale(image)

    # apply gaussian blur
    kernelSize = 5
    gaussianBlur = gaussian_blur(grayscaled, kernelSize)

    # canny
    minThreshold =200
    maxThreshold = 300
    edgeDetectedImage = canny(gaussianBlur, minThreshold, maxThreshold)

    # apply mask
    lowerLeftPoint = [59, 654]
    upperLeftPoint = [568, 412]
    upperRightPoint = [705, 412]
    lowerRightPoint = [1056, 644]

    pts = np.array([[lowerLeftPoint, upperLeftPoint, upperRightPoint, 
                    lowerRightPoint]], dtype=np.int32)
    masked_image = region_of_interest(edgeDetectedImage, pts)

    # hough lines
    rho = 1
    theta = np.pi/180
    threshold = 80
    min_line_len = 20 
    max_line_gap = 20

    houged, lines = hough_lines(masked_image, rho, theta, threshold, min_line_len, 
                         max_line_gap)

    # outline the input image
    colored_image = weighted_img(houged, image)
    return colored_image


cap = cv2.VideoCapture('road_drive.mp4')
fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('road_drive_line_detection.mp4',fourcc, 20.0, (1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        Road_Lines = process_image(frame)

        # write the flipped frame
        out.write(Road_Lines)

        cv2.imshow('Road_Lines',Road_Lines)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
