import numpy as np
from enum import Enum
from queue import PriorityQueue
from bresenham import bresenham

class Action(Enum):
    """
    An action is represented by a 3 element tuple.
    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """

    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)
    
    NE = (-1, 1, 1)
    NW = (-1, -1, 1)
    SE = (1, 1, 1)
    SW = (1, -1, 1)
    	
    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])

def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil(north_max - north_min))
    east_size = int(np.ceil(east_max - east_min))
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)

def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    # check if the node is off the grid or
    # it's an obstacle

    if x - 1 < 0 or grid[x - 1, y] == 1:
        valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y] == 1:
        valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x, y - 1] == 1:
        valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x, y + 1] == 1:
        valid_actions.remove(Action.EAST)




    return valid_actions


def unreal_to_AirSim_local(points):
	
	Drone_start_X = -20
	Drone_start_Y = 0
        
	Player_start_component = [5690, -100, 202]
	airsim_local = []
	for i in range(0, len(points)):
		local_x = (points[i][0]-Player_start_component[0])/100
		local_y = (points[i][1]-Player_start_component[1])/100
		local_z = (points[i][2]+Player_start_component[2])/100	
		airsim_local.append([local_x , local_y , local_z])		
	return airsim_local
	

def a_star(grid, h, start, goal):

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False
    
    while not queue.empty():
        item = queue.get()
        current_node = item[1]
        if current_node == start:
            current_cost = 0.0
        else:              
            current_cost = branch[current_node][0]
            
        if current_node == goal:        
            #print('Found a path.')
            found = True
            break
        else:
            for action in valid_actions(grid, current_node):
                # get the tuple representation
                da = action.delta
                next_node = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)               
                    branch[next_node] = (branch_cost, current_node, action)
                    queue.put((queue_cost, next_node))
             
    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

def prune_path(path, grid):
    pruned_path = [p for p in path]
    #print ('pruneing path \n length of path {0}'.format (len(path)))
    #print (path)
    
    i = 0
    while i < (len(pruned_path) - 2):
        p1 = np.array([pruned_path[i][0], pruned_path[i][1]])
        p2 = np.array([pruned_path[i+1][0], pruned_path[i+1][1]])
        p3 = np.array([pruned_path[i+2][0], pruned_path[i+2][1]])
        
        #print ('checking bresenham for {0} \t {1} \t {2}'.format(p1, p2, p3))
        
        if bresenham_check(p1, p2, p3, grid):
            pruned_path.remove(pruned_path[i+1])
        else:
            i +=1
            
    return pruned_path

def point(p):
    #print (p[0], p[1])
    return np.array(p[0], p[1])
    
def bresenham_check(p1, p2, p3, grid, epsilon=1e-6):
    line = (p1[0], p1[1], p3[0], p3[1])
    cells = list(bresenham(line[0], line[1], line[2], line[3]))
    clear = True
    
    for q in cells:
        if grid[q[0], q[1]] == 1:
            clear = False
            return clear
    return clear
	
def airsim_to_unreal(p):	
	x = p[0]
	y = p[1]
	
	unreal_x = (x)*100+3690
	unreal_y = (y)*100+(-100)
	return (unreal_x, unreal_y)
	
def plan_path(start, goal):
	
        unreal_start= airsim_to_unreal(start) 
        unreal_goal = airsim_to_unreal(goal)
	
        TARGET_ALTITUDE = 600
        SAFETY_DISTANCE = 100
	
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
	
        # Define starting point on the grid (this is just grid center)
        grid_start = (int(unreal_start[0])-north_offset, int(unreal_start[1])-east_offset)
        # Set goal as some arbitrary position on the grid
        grid_goal = (int(unreal_goal[0])-north_offset, int(unreal_goal[1])-east_offset)

        # Run A* to find a path from start to goal
        # or move to a different search space such as a graph (not done here)
        #print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)
        pruned_path = prune_path(path, grid)

        # Convert path to waypoints
        waypoints = [[p[0] + north_offset , p[1] + east_offset, TARGET_ALTITUDE] for p in pruned_path]
        return unreal_to_AirSim_local(waypoints)
        
if __name__ ==  "__main__":
	
	airsim_way_points = plan_path((0,0),(0,0))
	print(airsim_way_points)        

