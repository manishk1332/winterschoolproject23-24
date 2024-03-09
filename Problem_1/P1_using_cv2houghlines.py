import cv2
import numpy as np
import math

def avg_slope_intercept(lines):
    pts = []
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt = [int(x0 + 1000*(-b)), int(y0 + 1000*(a)), int(x0 - 1000*(-b)), int(y0 - 1000*(a))]
        pts.append(pt)
    print(pts)

    neg_slope_lines = []
    pos_slope_lines = []

    # Calculate the slope and intercept of each line and 
    # segragate on basis of neg and pos slope
    for line in pts:
        print(line)
        x1, y1, x2, y2 = line
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - (slope * x1)
        if (slope<-3.1):
            neg_slope_lines.append((slope, intercept))
        elif(slope>2.5):
            pos_slope_lines.append((slope, intercept))

    # Take average of lines
    if(len(neg_slope_lines)>len(pos_slope_lines) and len(neg_slope_lines)>0):    
        sum = 0
        for i in neg_slope_lines:
            sum += i[0]
        slp = sum/len(neg_slope_lines)
        sum = 0
        for i in neg_slope_lines:
            sum+=i[1]
        incp = sum/len(neg_slope_lines)
        neg_slope_line_sni = [slp, incp]
        return neg_slope_line_sni

    elif(len(pos_slope_lines)>0):    
        sum = 0
        for i in pos_slope_lines:
            sum += i[0]
        slp = sum/len(pos_slope_lines)
        sum = 0
        for i in pos_slope_lines:
            sum += i[1]
        incp = sum/len(pos_slope_lines)
        pos_slope_line_sni = [slp, incp]
        return pos_slope_line_sni

    return None


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def reqd_lines(image, lines):
    """
    Create full lenght lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    line_sni = avg_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = 0
    line_points  = pixel_points(y1, y2, line_sni)
    return line_points


cap = cv2.VideoCapture(r"assets\Videos_ComputerVision_Project1\1.mp4")

fgbg = cv2.createBackgroundSubtractorKNN(dist2Threshold=800,detectShadows=False)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    fgmask = fgbg.apply(frame)
    opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = cv2.GaussianBlur(closing, (3, 3), 0)
    
    grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
      
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    lines = cv2.HoughLines(grad, 1, np.pi / 180, 200, None, 0, 0)

    l1 = None

    if(lines is not None):
        l1 = reqd_lines(grad, lines)
        print(l1)

    if l1 is not None: 
        cv2.line(frame, l1[0],l1[1],(255,255,0),5)

    '''for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1,y1),(x2,y2),(255,255,0),5)'''
        
    '''for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        print(pt1,pt2)
        cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)'''

    cv2.imshow('Frame',frame)
 
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  else: 
    break

cap.release()
cv2.destroyAllWindows()