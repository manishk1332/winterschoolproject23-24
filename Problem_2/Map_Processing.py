import cv2 as cv
import numpy as np
import math

map_init = cv.imread(r"assets\unlabelled_map.png", cv.IMREAD_COLOR)
cv.imshow("Unprocessed Map", map_init)
cv.waitKey(0)
cv.destroyAllWindows()

bin_map_nodfs = np.full((476,890), 0, dtype=np.uint8)
for i in range(476):
    for j in range(890):
        if(tuple(map_init[i,j]) == (255, 255, 255)):
            bin_map_nodfs[i,j] = 255

cv.imshow("Binary unprocessed map, before DFS", bin_map_nodfs)
cv.waitKey(0)
cv.destroyAllWindows()

#Taking the Final Roadmap as the current binary Image
bin_map = bin_map_nodfs

class Graph:
    def __init__(self, display):
        self.__AdjList = []
        self.Coordinates = []
        self.image = display
        self.maxminx = None
        self.maxminy = None

    def add_node(self, coordinates):
        self.__AdjList.append([])
        self.Coordinates.append(coordinates)
        y, x = coordinates
        cv.circle(self.image, (x, y), 2, color=(255,0,0))
        if(self.maxminx == None):
            self.maxminx = (x, x)
            self.maxminy = (y, y)
        else:
            self.maxminx = (max(self.maxminx[0], x), min(self.maxminx[1], x))
            self.maxminy = (max(self.maxminy[0], y), min(self.maxminy[1], y))

    def add_edge(self, i, j):
        self.__AdjList[i].append(j)
        self.__AdjList[j].append(i)
        y1, x1 = self.Coordinates[i]
        y2, x2 = self.Coordinates[j]
        cv.line(self.image, (x1, y1), (x2, y2), color=(0,255,0), thickness=1)

    def sample_point(self):
        range = 10
        x, y = np.random.randint(890), np.random.randint(476)
        return (y, x)

    def find_Nearest(self, coordinates):
        found_a_reachable_point = False
        min_dist = math.sqrt(475*475 + 889*889)
        y1, x1 = coordinates
        for i in range(len(self.Coordinates)):
            y2, x2 = self.Coordinates[i]
            dist = math.sqrt((y2-y1)**2+(x2-x1)**2)
            if(min_dist > dist):
                empty_image = np.zeros((476, 890), dtype=np.uint8)
                cv.line(empty_image, (x1, y1), (x2, y2), color=255, thickness=1)
                should_be_just_a_line = cv.bitwise_and(empty_image, bin_map)
                if(np.array_equal(should_be_just_a_line, empty_image)):
                    min_dist = dist
                    nearest = i
                    found_a_reachable_point = True
        if(found_a_reachable_point == True):
            return nearest
        else:
            return None
        
def movement_towards_unit_vector(point1, point2, move_distance):
    y1, x1 = point1
    y2, x2 = point2
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    if(dist>move_distance):
        y = np.floor(y1+(y2-y1)*move_distance/dist + 0.5).astype(np.uint16)
        x = np.floor(x1+(x2-x1)*move_distance/dist + 0.5).astype(np.uint16)
        return y,x
    else:
        return point2

def RRT(bin_image, initial_pos, destination, dist_trim, goal_radius):
    temp_image = cv.cvtColor(bin_image, cv.COLOR_GRAY2BGR)
    G = Graph(temp_image)
    G.add_node(initial_pos)
    while True:
        #End this loop when nearest vertex to goal is within 'goal_radius'
        cv.imshow("RRT Path Planning", temp_image)
        Key = cv.waitKey(20)
        if(Key != -1):
            print("Loop broken due to Key Press")
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
                    G.add_node((y, x))
                    G.add_edge(-1, nearest)
                    break
        y1, x1 = destination
        y2, x2 = G.Coordinates[-1]
        dist = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        if(dist <= goal_radius):
            G.add_node(destination)
            G.add_edge(-2, -1)
            print("Goal reached! press any key to exit")
            cv.imshow("RRT Path Planning", temp_image)
            break
    cv.waitKey(0)
    cv.destroyAllWindows()

RRT(bin_map, (188, 187), (223, 724), 100, 17)
    
