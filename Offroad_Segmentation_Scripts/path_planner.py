import numpy as np
import cv2

class PathPlanner:
    """
    Utility to calculate the safest path from a segmentation mask.
    """
    
    # Class mapping from dataset_loader.py (for reference)
    # 0: Trees
    # 1: Lush Bushes
    # 2: Dry Grass
    # 3: Dry Bushes
    # 4: Ground Clutter
    # 5: Flowers
    # 6: Logs
    # 7: Rocks
    # 8: Landscape
    # 9: Sky

    # Traversability Costs (Lower is safer)
    TRAVERSABILITY_COSTS = {
        0: 100,  # Trees (Impassable)
        1: 20,   # Lush Bushes (Moderate resistance)
        2: 2,    # Dry Grass (Extremely safe)
        3: 10,   # Dry Bushes (Low resistance)
        4: 5,    # Ground Clutter (Very low resistance)
        5: 5,    # Flowers (Very low resistance)
        6: 100,  # Logs (Impassable/Obstacle)
        7: 100,  # Rocks (Impassable/Obstacle)
        8: 2,    # Landscape (Generally safe dirt/terrain)
        9: 255   # Sky (Not traversable)
    }

    # Path Centering Parameters
    CENTERING_STRENGTH = 10.0  # Higher value = path stays closer to center
    CORRIDOR_MARGIN = 0.2     # Keep path in middle 60% (ignore 20% on each side)

    def __init__(self, costs=None):
        self.costs = costs if costs else self.TRAVERSABILITY_COSTS

    def calculate_cost_map(self, mask):
        """
        Converts a semantic mask to a cost map with noise filtering.
        
        Args:
            mask (np.ndarray): Mask with class indices [0, 9]
        Returns:
            np.ndarray: Cost map with values representing traversability cost.
        """
        # 1. Base cost map calculation
        cost_map = np.zeros_like(mask, dtype=np.float32)
        for class_idx, cost in self.costs.items():
            cost_map[mask == class_idx] = cost
        
        # Ensure any unmapped values are high cost
        cost_map[~np.isin(mask, list(self.costs.keys()))] = 255

        # 2. Noise Filtering: Remove small high-cost artifacts
        # Identify "Hard" obstacles: Trees (0), Logs (6), Rocks (7)
        hard_obstacles = np.isin(mask, [0, 6, 7]).astype(np.uint8)
        
        # Use connected components to filter by area
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hard_obstacles, connectivity=8)
        
        # Create a mask for refined obstacles (starting empty)
        refined_obstacles_mask = np.zeros_like(hard_obstacles)
        
        # Minimum area (in pixels) to be considered a real obstacle at 320x320 resolution
        MIN_OBSTACLE_AREA = 10 
        
        for i in range(1, num_labels): # Skip background (index 0)
            if stats[i, cv2.CC_STAT_AREA] >= MIN_OBSTACLE_AREA:
                refined_obstacles_mask[labels == i] = 1
        
        # Find where an obstacle was "filtered out"
        filtered_out = (hard_obstacles == 1) & (refined_obstacles_mask == 0)
        
        # Replace the cost of filtered-out pixels with a safe landcape cost (2)
        cost_map[filtered_out] = 2.0
        
        return cost_map

    def find_safest_path(self, mask, start_pos=None):
        """
        Finds a safest path from the bottom-center region upwards,
        constrained to a central corridor.
        
        Args:
            mask (np.ndarray): Semantic mask.
            start_pos (tuple): (x, y) coordinates of the starting point. 
                               Defaults to safest point in bottom corridor.
        Returns:
            list: List of (x, y) coordinates representing the path.
        """
        cost_map = self.calculate_cost_map(mask)
        h, w = cost_map.shape
        
        # Define Corridor horizontal bounds
        margin_px = int(w * self.CORRIDOR_MARGIN)
        corridor_min_x = margin_px
        corridor_max_x = w - margin_px - 1
        
        if start_pos is None:
            # 1. Define region for robust start selection (bottom 5% of image)
            start_region_h = max(1, int(h * 0.05))
            bottom_region = cost_map[h - start_region_h : h, corridor_min_x : corridor_max_x + 1]
            
            # 2. Average cost per column in this region
            col_avg_costs = np.mean(bottom_region, axis=0)
            
            # 3. Identify "Safe Segments" (where average cost is low)
            SAFE_START_THRESHOLD = 5.0 # Consider costs below 5 as "safe"
            is_safe = col_avg_costs < SAFE_START_THRESHOLD
            
            # Use connected components/labeling to find contiguous safe segments
            # Find transitions from False to True or vice versa
            safe_segments = []
            if np.any(is_safe):
                # Simple segment finding
                start_idx = None
                for idx, safe in enumerate(is_safe):
                    if safe and start_idx is None:
                        start_idx = idx
                    elif not safe and start_idx is not None:
                        safe_segments.append((start_idx, idx - 1))
                        start_idx = None
                if start_idx is not None:
                    safe_segments.append((start_idx, len(is_safe) - 1))
            
            if safe_segments:
                # 4. Evaluate segments based on width and proximity to center
                best_score = -float('inf')
                best_x = corridor_min_x + (corridor_max_x - corridor_min_x) // 2 # default center
                
                center_x_rel = (corridor_max_x - corridor_min_x) // 2
                
                for start_s, end_s in safe_segments:
                    width = end_s - start_s + 1
                    mid_s = (start_s + end_s) // 2
                    dist_from_center = abs(mid_s - center_x_rel) / (w / 2)
                    
                    # Robust Score Calculation:
                    # 1. Base score is width (in pixels)
                    # 2. Subtract centering penalty (scaled down)
                    # 3. Add width bonus for segments over a minimum width (e.g., vehicle width)
                    
                    centering_penalty = self.CENTERING_STRENGTH * dist_from_center
                    score = width - centering_penalty
                    
                    if width > 10: # Bonus for definitely wide enough areas
                        score += 20
                    
                    if score > best_score:
                        best_score = score
                        best_x = corridor_min_x + mid_s
                
                start_pos = (best_x, h - 1)
            else:
                # Fallback to absolute minimum if no safe segments found
                # Apply centering penalty to start position selection as well
                center_x = w // 2
                x_coords_start = np.arange(corridor_min_x, corridor_max_x + 1)
                dist_from_center_start = np.abs(x_coords_start - center_x) / (w / 2)
                centering_penalty_start = self.CENTERING_STRENGTH * dist_from_center_start
                
                # Use bottom row for fallback
                bottom_row_corridor = cost_map[h - 1, corridor_min_x : corridor_max_x + 1]
                combined_start_cost = bottom_row_corridor + centering_penalty_start
                start_x_idx = np.argmin(combined_start_cost)
                start_pos = (corridor_min_x + start_x_idx, h - 1)
            
        path = [start_pos]
        curr_x, curr_y = start_pos
        
        # Parameters for path search
        look_ahead = 5  # pixels to jump per step
        window_size = 60 # width of window to look for lowest cost
        
        while curr_y > 0: # Continue until the top of the image
            next_y = max(0, curr_y - look_ahead)
            
            # 1. Determine search window at next_y, constrained by corridor
            search_min_x = max(corridor_min_x, curr_x - window_size // 2)
            search_max_x = min(corridor_max_x, curr_x + window_size // 2)
            
            if search_min_x >= search_max_x:
                break
                
            window = cost_map[next_y, search_min_x : search_max_x + 1]
            
            if len(window) == 0:
                break
            
            # 2. Check for Sky/Horizon
            if np.mean(mask[next_y, search_min_x : search_max_x + 1] == 9) > 0.5:
                break
                
            # 3. Apply Centering Penalty
            center_x = w // 2
            x_coords = np.arange(search_min_x, search_max_x + 1)
            # Distance from center normalized to [0, 1] within image width
            dist_from_center = np.abs(x_coords - center_x) / (w / 2)
            centering_penalty = self.CENTERING_STRENGTH * dist_from_center
            
            # Combine traversability cost with centering bias
            combined_cost = window + centering_penalty
            
            # 4. Find index of minimum combined cost
            min_idx = np.argmin(combined_cost)
            next_x = search_min_x + min_idx
            
            # 5. Smoothing/Momentum
            next_x = int(0.7 * next_x + 0.3 * curr_x)
            # Ensure smoothed result stays in corridor
            next_x = max(corridor_min_x, min(corridor_max_x, next_x))
            
            curr_x, curr_y = next_x, next_y
            path.append((curr_x, curr_y))
            
        return path

    def visualize_on_image(self, image, path, color=(0, 255, 0), thickness=5):
        """
        Draws the path on an image.
        
        Args:
            image (np.ndarray): Image to draw on.
            path (list): List of (x, y) path coordinates.
        Returns:
            np.ndarray: Image with path drawn.
        """
        vis_image = image.copy()
        if len(path) < 2:
            return vis_image
            
        pts = np.array(path, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(vis_image, [pts], False, color, thickness)
        
        return vis_image
