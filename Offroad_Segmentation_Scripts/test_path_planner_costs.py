import numpy as np
import sys
import os

# Add the scripts directory to path if needed
sys.path.append(os.path.join(os.getcwd(), 'Offroad_Segmentation_Scripts'))

from path_planner import PathPlanner

def test_costs_and_noise():
    planner = PathPlanner()
    
    # Create a 10x10 mask with mostly landscape (8)
    mask = np.full((10, 10), 8, dtype=np.uint8)
    
    # Add some noise: isolated obstacle pixels (Rock=7, Tree=0)
    mask[2, 2] = 7  # Single rock pixel (noise, area 1)
    mask[8, 8] = 0  # Single tree pixel (noise, area 1)
    
    # Add a small cluster (2x2 = area 4) which should also be filtered out (MIN_OBSTACLE_AREA=10)
    mask[0:2, 0:2] = 7
    
    # Add a real obstacle (4x3 = area 12)
    mask[4:8, 4:7] = 7
    
    cost_map = planner.calculate_cost_map(mask)
    
    print("Testing Traversability Costs and Area-Based Noise Filtering...")
    
    # Check that noise (2, 2) and (8, 8) was filtered out (cost should be 2, not 100)
    assert cost_map[2, 2] == 2, f"Area=1 noise at (2, 2) not filtered! Cost: {cost_map[2, 2]}"
    assert cost_map[8, 8] == 2, f"Area=1 noise at (8, 8) not filtered! Cost: {cost_map[8, 8]}"
    
    # Check that small cluster (0,0) was filtered out (cost should be 2)
    assert cost_map[0, 0] == 2, f"Area=4 cluster at (0, 0) not filtered! Cost: {cost_map[0, 0]}"
    
    # Check that real obstacle (4, 4) remains high cost (100)
    assert cost_map[4, 4] == 100, f"Real obstacle (area 12) at (4, 4) incorrectly filtered! Cost: {cost_map[4, 4]}"
    
    # Check vegetation costs
    mask[0, 0] = 1 # Lush Bush
    mask[0, 1] = 3 # Dry Bush
    mask[0, 2] = 4 # Ground Clutter
    mask[0, 3] = 5 # Flowers
    
    cost_map_veg = planner.calculate_cost_map(mask)
    assert cost_map_veg[0, 0] == 20, f"Lush Bush cost mismatch: {cost_map_veg[0, 0]}"
    assert cost_map_veg[0, 1] == 10, f"Dry Bush cost mismatch: {cost_map_veg[0, 1]}"
    assert cost_map_veg[0, 2] == 5, f"Ground Clutter cost mismatch: {cost_map_veg[0, 2]}"
    assert cost_map_veg[0, 3] == 5, f"Flowers cost mismatch: {cost_map_veg[0, 3]}"
    
    print("SUCCESS: Costs and noise filtering are working correctly!")

def test_centering_and_corridor():
    planner = PathPlanner()
    
    # Create a 100x100 mask (larger for easier spatial testing)
    # Background is Landscape (8) - cost 2
    mask = np.full((100, 100), 8, dtype=np.uint8)
    
    # Find safest path
    path = planner.find_safest_path(mask)
    
    # Check 1: Start position should be near bottom-center
    # With the new robust logic, (49, 99) or (50, 99) are both likely center.
    start_pos = path[0]
    print(f"Path starts at: {start_pos}")
    assert abs(start_pos[0] - 50) <= 2 and start_pos[1] == 99, f"Start pos mismatch: {start_pos}"
    
    # Check 2: Corridor constraints. 
    # Corridor Margin is 0.2, so 100 * 0.2 = 20 pixels on each side should be ignored.
    # Path should stay between x=20 and x=79.
    xs = [p[0] for p in path]
    min_x = min(xs)
    max_x = max(xs)
    print(f"Path X range: {min_x} to {max_x}")
    assert min_x >= 20, f"Path entered left margin! min_x: {min_x}"
    assert max_x <= 79, f"Path entered right margin! max_x: {max_x}"
    
    # Check 3: Centering Bias preference
    # Create a "tempting" path on the left (but still inside corridor e.g. x=25) 
    # vs a path in the center (x=50). Center should be chosen.
    # To do this, we'll make the center more expensive but still within reason.
    # Landscape cost is 2. Centering penalty at x=25 (dist 25/50 = 0.5) is 10 * 0.5 = 5. Total = 7.
    # If we make center cost 10, the x=25 path (cost 7) might be better.
    # But if center cost is only 4, path should stay center (cost 4+0=4 < 7).
    mask[:, 45:55] = 1 # Lush Bush (cost 20)
    # Now center is expensive. Let's see if it tries to stay center despite high cost.
    path_bush = planner.find_safest_path(mask)
    xs_bush = [p[0] for p in path_bush]
    print(f"Path X with center bush: Mean X = {np.mean(xs_bush):.2f}")
    
    # Path should stay within corridor even with obstacles on the side
    mask[:, :30] = 7 # Rocks on left
    path_rocks = planner.find_safest_path(mask)
    xs_rocks = [p[0] for p in path_rocks]
    assert min(xs_rocks) >= 20, "Path ignored corridor constraint with obstacles!"
    
    print("SUCCESS: Centering and corridor constraints are working correctly!")

def test_robust_start_selection():
    planner = PathPlanner()
    
    # Create a 100x100 mask
    mask = np.full((100, 100), 8, dtype=np.uint8) # Landscape (cost 2)
    
    # Corridor is 20 to 79. Center is 50.
    
    # Scenario: 
    # Current "absolute minimum" is a narrow gap at x=40
    # A wider safe area is at x=60
    # Center is 50.
    
    # Make everything high cost first
    mask[90:, :] = 0 # Trees (cost 100)
    
    # Create narrow safe gap (1 pixel wide) at x=40
    mask[90:, 40] = 8
    
    # Create wide safe area (10 pixels wide) at x=60 to 70
    mask[90:, 60:71] = 8
    
    path = planner.find_safest_path(mask)
    start_pos = path[0]
    print(f"Robust test start pos: {start_pos}")
    
    # It should prefer the wide segment at x=65 (middle of 60-70) 
    # even though it's slightly further from center (50) than the gap at x=40.
    # The score for x=65: width=11, penalty = centering pixels (15)
    # The score for x=40: width=1, penalty = centering pixels (10)
    
    assert 60 <= start_pos[0] <= 70, f"Expected start in wide segment (60-70), but got {start_pos[0]}"

if __name__ == "__main__":
    test_costs_and_noise()
    test_centering_and_corridor()
    test_robust_start_selection()
