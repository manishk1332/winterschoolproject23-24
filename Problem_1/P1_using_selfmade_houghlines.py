import cv2
import numpy as np

def hough_lines(img: np.ndarray, rho_res: float, theta_res: float, threshold: int, min_theta: float, max_theta: float) -> np.ndarray:
    max_rho = np.sqrt(img.shape[0]**2 + img.shape[1]**2) #Diagonal Size of the Image

    # Define the Hough space range
    theta_range = np.arange(min_theta, max_theta, theta_res)
    rho_range = np.arange(-max_rho, max_rho, rho_res)

    num_theta = len(theta_range)
    num_rho = len(rho_range)

    # Create the vote bank array to store the votes, initially initialize all votes to 0
    vote_bank = np.zeros([num_rho,num_theta])

    sin_theta_range = np.sin(theta_range)
    cos_theta_range = np.cos(theta_range)

    # Find edge points (non-zero values) in the image
    edge_points = np.where(img>0)
    
    # Voting starts
    for i in range(len(edge_points[0])):
        x = edge_points[0][i]
        y = edge_points[1][i]

        for current_theta in range(num_theta):
            current_rho = x*cos_theta_range[current_theta] + y*sin_theta_range[current_theta]

            rho_vote = np.where(rho_range < current_rho)[0][-1]
            vote_bank[rho_vote, current_theta]+=1

    # Identify lines that are above the threshold
    final_rho_index, final_theta_index = np.where(vote_bank > threshold)
    final_rho = rho_range[final_rho_index]
    final_theta = theta_range[final_theta_index]

    lines = np.vstack([final_rho, final_theta]).T
    return lines


def polar2cartesian(radius: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return radius * np.array([np.sin(angle), np.cos(angle)])


def avg_slope_intercept(lines):
    pts = []
    # Convert lines from rho, theta to two points
    for rho, theta in lines:
        x0 = polar2cartesian(rho, theta)
        direction = np.array([x0[1], -x0[0]])
        pt1 = np.round(x0 + 1000*direction).astype(int)
        pt2 = np.round(x0 - 1000*direction).astype(int)
        pt = [pt1[0],pt1[1],pt2[0],pt2[1]]
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


cap = cv2.VideoCapture(r"assets\Videos_ComputerVision_Project1\3.mp4")

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

        lines = hough_lines(grad,6,0.161,700,-np.pi/2,np.pi/2)

        l = None

        if(lines is not None):
            l = reqd_lines(grad, lines)


        if l is not None: 
            cv2.line(frame, l[0],l[1],(255,255,0),5)

        '''for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1,y1),(x2,y2),(255,255,0),5)'''
            
        '''for rho, theta in lines:
            x0 = polar2cartesian(rho, theta)
            direction = np.array([x0[1], -x0[0]])
            pt1 = np.round(x0 + 1000*direction).astype(int)
            pt2 = np.round(x0 - 1000*direction).astype(int)
            print(pt1,pt2)
            cv2.line(frame, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)'''

        cv2.imshow('Frame',frame)
    
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    else: 
        break

cap.release()
cv2.destroyAllWindows()