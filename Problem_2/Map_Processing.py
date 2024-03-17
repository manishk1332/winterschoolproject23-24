import cv2 as cv
import numpy as np
import math

map_init = cv.imread("assets/unlabelled_map.png", cv.IMREAD_COLOR)
cv.imshow("Unprocessed Map", map_init)
cv.waitKey(0)
cv.destroyAllWindows()

bin_map = np.full((476,890), 0, dtype=np.uint8)
for i in range(476):
    for j in range(890):
        if(tuple(map_init[i,j]) == (255, 255, 255)):
            bin_map[i,j] = 255

cv.imshow("Binary map", bin_map)
cv.waitKey(0)
cv.destroyAllWindows()

def distance(point1, point2):
    y1, x1 = np.float64(point1[0]), np.float64(point1[1])
    y2, x2 = np.float64(point2[0]), np.float64(point2[1])
    return math.sqrt((y2-y1)**2 + (x2-x1)**2)

def line_segment_is_unobstructed(point1, point2, bin_image):
    #checks whether line segment formed by Bresenham's Algorithm has value 255 in each pixel of bin_image
    y1, x1 = point1
    y2, x2 = point2
    dy = abs(y2-y1)
    dx = abs(x2-x1)
    if(dy>=dx):
        D = (2*dx-dy)
        y = min(y1, y2)
        if(y==y1): x = x1
        else: x = x2
        if(x<max(x1, x2)): step = 1
        else: step = -1
        while(y <= max(y1, y2)):
            if(bin_image[y][x] != 255):
                return False
            if (D>0):
                x += step
                D = D-2*dy
            D = D+2*dx
            y += 1
    elif(dx>dy):
        D = 2*dy-dx
        x = min(x1, x2)
        if(x==x1): y = y1
        else: y = y2
        if(y<max(y1, y2)): step = 1
        else: step = -1
        while(x <= max(x1, x2)):
            if(bin_image[y][x] != 255):
                return False
            if (D>0):
                y += step
                D = D-2*dx
            D = D+2*dy
            x += 1
    return True

class Graph:
    def __init__(self, map, destination_coordinates):
        self.Coordinates = []
        self.parent = []
        self.children_list = []
        self.dist = []
        self.size = 0
        self.blank_map = map

        #for checking whether destination is reached, and to backtrace the path to destination while rendering map for display
        self.dest_reached = False
        self.dest_index = None
        self.dest = destination_coordinates

    def add_node(self, coordinates, cur_parent):
        self.size += 1
        self.Coordinates.append(coordinates)
        try:
            if(cur_parent < 0):
                cur_parent = self.size + cur_parent
            self.children_list[cur_parent].append(self.size-1)
            self.dist.append(self.dist[cur_parent]+distance(self.Coordinates[cur_parent], coordinates))
        except TypeError:
            self.dist.append(0)
        self.parent.append(cur_parent)
        self.children_list.append([])
    
    def ch_parent(self, index, new_parent):
        if(new_parent<0):
            new_parent = self.size + new_parent
        self.children_list[self.parent[index]].remove(index)
        self.parent[index] = new_parent
        self.children_list[new_parent].append(index)

    def add_dest(self, cur_parent):
        if(self.dest_reached):
            self.ch_parent(self.dest_index, cur_parent)
        else:
            self.dest_reached = True
            self.add_node(self.dest, cur_parent)
            self.dest_index = self.size - 1

    def sample_point(self):
        x, y = np.random.randint(890), np.random.randint(476)
        return (y, x)

    def find_Nearest(self, coordinates):
        found_a_reachable_point = False
        min_dist = -1 #think of -1 as some random uninitialized value, found_a_reachable_point will take care of it
        y1, x1 = coordinates
        for i in range(len(self.Coordinates)):
            dist = distance(self.Coordinates[i], coordinates)+ self.dist[i]
            if(min_dist > dist or (not found_a_reachable_point)):
                y2, x2 = self.Coordinates[i]
                if(line_segment_is_unobstructed(coordinates, self.Coordinates[i], bin_map)):
                    min_dist = dist
                    nearest = i
                    found_a_reachable_point = True
        if(found_a_reachable_point == True):
            return nearest
        else:
            return None

    def Render(self):
        display = self.blank_map.copy()
        for i in range(self.size):
            try:
                y1, x1 = self.Coordinates[i]
                cv.circle(display, (x1, y1), 2, color=(255,0,0))
                if(self.parent[i] == None):
                    raise ValueError
                y2, x2 = self.Coordinates[self.parent[i]]
                cv.line(display, (x1, y1), (x2, y2), color=(0,255,0), thickness=1)
            except ValueError:
                pass
        #Highlighting path from source to destination
        try:
            if(not self.dest_reached):
                raise ValueError
            index = self.dest_index
            while (index != 0):
                y1, x1 = self.Coordinates[index] # type: ignore
                y2, x2 = self.Coordinates[index := self.parent[index]] # type: ignore
                cv.line(display, (x1, y1), (x2, y2), color=(0,0,255), thickness=1)
        except:
            pass
        cv.imshow("RRT Path Planning", display)
        
def movement_towards_unit_vector(point1, point2, move_distance):
    y1, x1 = point1
    y2, x2 = point2
    dist = distance(point1, point2)
    if(dist>move_distance):
        y = np.floor(y1+(y2-y1)*move_distance/dist + 0.5).astype(np.uint16)
        x = np.floor(x1+(x2-x1)*move_distance/dist + 0.5).astype(np.uint16)
        return y,x
    else:
        return point2

def RRT(bin_image, initial_pos, destination, dist_trim, goal_radius):
    temp_image = cv.cvtColor(bin_image, cv.COLOR_GRAY2BGR)
    G = Graph(temp_image, destination)
    G.add_node(initial_pos, None)
    while True:
        #This loops ends only when user Presses a Key
        G.Render()
        Key = cv.waitKey(20)
        if(Key != -1):
            print("Loop broken due to Key Press, press any key to exit...")
            break
        while True:
            #End this loop when the Tree has grown by one vertex
            point = G.sample_point()
            if(bin_image[point] == 255):
                nearest = G.find_Nearest(point)
                if(nearest == None):
                    continue
                else:
                    y, x = movement_towards_unit_vector(G.Coordinates[nearest], point, dist_trim)
                    G.add_node((y, x), nearest)
                    break
        dist = distance(destination, G.Coordinates[-1])
        if(dist <= goal_radius):
            if(not G.dest_reached):
                print("Goal Reached! further iterations are being performed")
                print("Press any key to stop further iterations and exit...")
            G.add_dest(G.size-1)
            G.Render()
    cv.waitKey(0)
    cv.destroyAllWindows()

RRT(bin_map, (188, 187), (223, 724), 100, 17)
