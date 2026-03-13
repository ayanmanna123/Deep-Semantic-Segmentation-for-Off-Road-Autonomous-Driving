import numpy as np
import cv2
import heapq

class PathPlanner:
    """
    Advanced Path Planner using A* search with Obstacle Inflation (Safety Buffers).
    """
    
    # Class mapping for reference
    # 0: Trees, 1: Lush Bushes, 2: Dry Grass, 3: Dry Bushes, 4: Ground Clutter,
    # 5: Flowers, 6: Logs, 7: Rocks, 8: Landscape, 9: Sky

    TRAVERSABILITY_COSTS = {
        0: 255,  # Trees (Impassable)
        1: 40,   # Lush Bushes (High resistance)
        2: 2,    # Dry Grass (Ideal)
        3: 20,   # Dry Bushes (Moderate resistance)
        4: 10,   # Ground Clutter (Low resistance)
        5: 10,   # Flowers (Low resistance)
        6: 255,  # Logs (Impassable)
        7: 255,  # Rocks (Impassable)
        8: 2,    # Landscape (Ideal)
        9: 255   # Sky (Impassable)
    }

    def __init__(self, costs=None):
        self.costs = costs if costs else self.TRAVERSABILITY_COSTS

    def calculate_cost_map(self, mask, safety_margin=25):
        """
        Converts semantic mask to a cost map with safety buffers around obstacles.
        """
        h, w = mask.shape
        cost_map = np.zeros_like(mask, dtype=np.float32)
        
        # 1. Base terrain costs
        for class_idx, cost in self.costs.items():
            cost_map[mask == class_idx] = cost

        # 2. Obstacle Inflation (Safety Buffer)
        # Identify "Hard" obstacles: Trees, Logs, Rocks
        obstacles = np.isin(mask, [0, 6, 7]).astype(np.uint8)
        
        # Distance to nearest obstacle (pixels)
        # 0 at obstacle, high value far away
        dist_transform = cv2.distanceTransform(1 - obstacles, cv2.DIST_L2, 5)
        
        # Add 'Fear' penalty near obstacles: higher cost closer to objects
        buffer_mask = dist_transform < safety_margin
        # Exponential curve for smooth repulsion
        penalty = ((safety_margin - dist_transform[buffer_mask]) / safety_margin) * 50
        cost_map[buffer_mask] += penalty.astype(np.float32)
        
        return np.clip(cost_map, 0, 255)

    def heuristic(self, a, b):
        """Diagonal distance heuristic."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def find_safest_path(self, mask):
        """
        Finds a globally optimal path from bottom-center to the horizon using A*.
        """
        # Optimization: Downsample for real-time speed (4x speedup)
        scale = 4
        h_full, w_full = mask.shape
        small_mask = cv2.resize(mask, (w_full // scale, h_full // scale), interpolation=cv2.INTER_NEAREST)
        
        # Increased safety margin to account for vehicle width
        cost_map = self.calculate_cost_map(small_mask, safety_margin=25 // scale)
        h, w = cost_map.shape

        start = (h - 1, w // 2)
        # Define many goals at the top (horizon area)
        goal_row = 0
        
        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: h} # Heuristic to reach top
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        # Steering smoothing factors
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        while oheap:
            current = heapq.heappop(oheap)[1]

            # If we reached the top row
            if current[0] <= 1:
                path = []
                while current in came_from:
                    # Scale back to original resolution
                    path.append((current[1] * scale, current[0] * scale))
                    current = came_from[current]
                return path[::-1]

            close_set.add(current)
            
            for i, j in neighbors:
                neighbor = (current[0] + i, current[1] + j)
                
                # Boundary check
                if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                    continue
                
                # Check for walls
                move_cost = cost_map[neighbor[0], neighbor[1]]
                if move_cost >= 200:
                    continue

                # Distance factor (Diagonal moves are 1.4x further)
                dist = 1.41 if (i != 0 and j != 0) else 1.0
                tentative_g_score = gscore[current] + (move_cost * dist)

                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 1e9):
                    continue

                if tentative_g_score < gscore.get(neighbor, 1e9):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    # Heuristic: Progress toward top + bias to stay centered
                    fscore[neighbor] = tentative_g_score + neighbor[0] + abs(neighbor[1] - w//2)*0.2
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return []

    def visualize_on_image(self, image, path, color=(255, 144, 30), thickness=6):
        """Draws a premium glowing path 'ribbon'."""
        vis_image = image.copy()
        if len(path) < 2:
            return vis_image
            
        pts = np.array(path, np.int32).reshape((-1, 1, 2))
        
        # Outer Glow (Wide, very transparent)
        glow_outer = vis_image.copy()
        cv2.polylines(glow_outer, [pts], False, color, thickness * 8)
        vis_image = cv2.addWeighted(vis_image, 0.8, glow_outer, 0.2, 0)
        
        # Inner Ribbon (Medium)
        ribbon = vis_image.copy()
        cv2.polylines(ribbon, [pts], False, color, thickness * 2)
        vis_image = cv2.addWeighted(vis_image, 0.7, ribbon, 0.3, 0)
        
        # Core Line (White/Bright)
        cv2.polylines(vis_image, [pts], False, (255, 255, 255), thickness // 2)
        
        return vis_image
