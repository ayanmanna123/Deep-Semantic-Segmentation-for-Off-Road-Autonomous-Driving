import numpy as np
import heapq
import cv2

class AStarPlanner:
    """
    Advanced path planner using A* algorithm on a traversability cost map.
    """
    
    # Costs (same as PathPlanner for consistency)
    TRAVERSABILITY_COSTS = {
        0: 100,  # Trees (Impassable)
        1: 20,   # Lush Bushes (Moderate resistance)
        2: 1,    # Dry Grass (Extremely safe) - Lowered to 1 for A* preference
        3: 10,   # Dry Bushes (Low resistance)
        4: 5,    # Ground Clutter (Very low resistance)
        5: 5,    # Flowers (Very low resistance)
        6: 100,  # Logs (Impassable/Obstacle)
        7: 100,  # Rocks (Impassable/Obstacle)
        8: 2,    # Landscape (Generally safe dirt/terrain)
        9: 255   # Sky (Not traversable)
    }

    def __init__(self, costs=None):
        self.costs = costs if costs else self.TRAVERSABILITY_COSTS

    def calculate_cost_map(self, mask):
        """
        Converts semantic mask to cost map with obstacle inflation.
        """
        h, w = mask.shape
        cost_map = np.zeros_like(mask, dtype=np.float32)
        for class_idx, cost in self.costs.items():
            cost_map[mask == class_idx] = cost
        
        # Ensure unmapped are high cost
        cost_map[~np.isin(mask, list(self.costs.keys()))] = 255

        # Inflate obstacles to keep path away from edges
        # Obstacles: 0 (Trees), 6 (Logs), 7 (Rocks), 9 (Sky)
        obstacles = np.isin(mask, [0, 6, 7, 9]).astype(np.uint8)
        
        # Kernel for inflation
        kernel = np.ones((11, 11), np.uint8)
        inflated_obstacles = cv2.dilate(obstacles, kernel, iterations=1)
        
        # Increase cost in inflated areas
        cost_map[inflated_obstacles == 1] += 50
        
        # Add a centering bias to prefer middle of the image
        center_x = w // 2
        y_coords, x_coords = np.indices((h, w))
        dist_from_center = np.abs(x_coords - center_x) / (w / 2)
        cost_map += dist_from_center * 5.0 # Centering strength

        return cost_map

    def find_path(self, mask, start=None, end=None):
        """
        A* algorithm to find the optimal path.
        """
        cost_map = self.calculate_cost_map(mask)
        h, w = cost_map.shape
        
        # If no start provided, pick bottom center
        if start is None:
            start = (h - 1, w // 2)
        
        # If no end provided, pick top center
        if end is None:
            end = (0, w // 2)

        # A* algorithm
        queue = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}
        
        # Neighbors: 8-connectivity
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        max_iterations = 10000 # Safety break
        iterations = 0

        while queue:
            iterations += 1
            if iterations > max_iterations:
                break

            current_priority, current = heapq.heappop(queue)

            if current == end:
                break
            
            # Optimization: if we reach top row, we can stop if goal is just "reach top"
            if current[0] == 0:
                end = current
                break

            for dx, dy in neighbors:
                next_node = (current[0] + dx, current[1] + dy)
                
                if 0 <= next_node[0] < h and 0 <= next_node[1] < w:
                    # Cost = traversability cost + movement cost (diagonal vs straight)
                    move_cost = np.sqrt(dx**2 + dy**2)
                    new_cost = cost_so_far[current] + (cost_map[next_node] * move_cost)
                    
                    if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                        cost_so_far[next_node] = new_cost
                        # Heuristic: simple L2 distance to top-center
                        priority = new_cost + np.sqrt((next_node[0] - end[0])**2 + (next_node[1] - end[1])**2)
                        heapq.heappush(queue, (priority, next_node))
                        came_from[next_node] = current

        # Reconstruct path
        path = []
        curr = end
        if curr not in came_from and curr != start:
            return [] # No path found
            
        while curr != start:
            path.append((curr[1], curr[0])) # Convert to (x, y)
            curr = came_from[curr]
        path.append((start[1], start[0]))
        path.reverse()
        
        return path
