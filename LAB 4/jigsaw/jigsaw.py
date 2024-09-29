
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import copy
from tqdm import tqdm
from collections import deque

def read_matrix_file(filepath):
    data = []
    with open(filepath, "r") as file:
        content = file.readlines()
    matrix_data = content[5:]
    for row in matrix_data:
        row = row.strip()
        if row:
            try:
                data.append(int(row))
            except ValueError:
                print(f"Invalid row skipped: {row}")
    data_array = np.array(data)
    if data_array.size != 512 * 512:
        raise ValueError(f"Expected 262144 elements, but got {data_array.size} elements.")
    return data_array.reshape((512, 512))

def split_into_segments(img):
    segment_size = 128
    segments_per_side = img.shape[0] // segment_size
    segments = {}
    layout = []
    counter = 0
    for i in range(segments_per_side):
        row = []
        for j in range(segments_per_side):
            row.append(counter)
            segment = img[
                i * segment_size : (i + 1) * segment_size,
                j * segment_size : (j + 1) * segment_size,
            ]
            segments[counter] = segment
            counter += 1
        layout.append(row)
    return segments, layout

def assemble_image(segments, arrangement):
    segment_height, segment_width = segments[0].shape[:2]
    rows = len(arrangement)
    cols = len(arrangement[0])
    assembled = np.zeros(
        (rows * segment_height, cols * segment_width), dtype=np.uint8
    )
    for i, row in enumerate(arrangement):
        for j, seg_idx in enumerate(row):
            assembled[
                i * segment_height : (i + 1) * segment_height,
                j * segment_width : (j + 1) * segment_width,
            ] = segments[seg_idx]
    return assembled

def calculate_difference(arr1, arr2):
    return np.sum(np.abs(arr1 - arr2))

def find_best_match(candidates, parent_seg, child_seg, direction):
    min_diff = float('inf')
    best_match = -1
    parent_seg = np.array(parent_seg)
    for candidate in candidates:
        diff = 0
        child_segment = np.array(child_seg[candidate])
        if direction == (0, 1):
            diff = np.sum(np.abs(parent_seg[:, -1] - child_segment[:, 0]))
        elif direction == (0, -1):
            diff = np.sum(np.abs(parent_seg[:, 0] - child_segment[:, -1]))
        elif direction == (1, 0):
            diff = np.sum(np.abs(parent_seg[-1, :] - child_segment[0, :]))
        elif direction == (-1, 0):
            diff = np.sum(np.abs(parent_seg[0, :] - child_segment[-1, :]))
        if diff < min_diff:
            min_diff = diff
            best_match = candidate
    return best_match

def breadth_first_arrangement(layout, segments, options):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    queue = deque([(0, 0)])
    seen = set()
    seen.add((0, 0))
    while queue:
        x, y = queue.popleft()
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 4 and 0 <= ny < 4 and (nx, ny) not in seen:
                queue.append((nx, ny))
                seen.add((nx, ny))
                best_segment = find_best_match(options, segments[layout[x][y]], segments, (dx, dy))
                layout[nx][ny] = best_segment
                options.remove(best_segment)

def display_image(img):
    plt.imshow(img, cmap="gray")
    plt.title("Reconstructed 512x512 Image")
    plt.colorbar()
    plt.show()

def get_adjacent_cells(row, col, grid):
    adjacent = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < len(grid) and 0 <= new_col < len(grid[0]):
            adjacent.append((new_row, new_col))
    return adjacent

def evaluate_arrangement(arrangement, segments):
    total_diff = 0
    for i in range(len(arrangement)):
        for j in range(len(arrangement[0])):
            neighbors = get_adjacent_cells(i, j, arrangement)
            for nr, nc in neighbors:
                if nc == j + 1:
                    total_diff += np.sum(np.abs(segments[arrangement[i][j]][:, -1] - segments[arrangement[nr][nc]][:, 0]))
                elif nc == j - 1:
                    total_diff += np.sum(np.abs(segments[arrangement[nr][nc]][:, -1] - segments[arrangement[i][j]][:, 0]))
                elif nr == i + 1:
                    total_diff += np.sum(np.abs(segments[arrangement[i][j]][-1, :] - segments[arrangement[nr][nc]][0, :]))
                elif nr == i - 1:
                    total_diff += np.sum(np.abs(segments[arrangement[nr][nc]][-1, :] - segments[arrangement[i][j]][0, :]))
    return np.sqrt(total_diff)

def compute_edge_strength(image, threshold=100):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_x[magnitude < threshold] = 0
    grad_y[magnitude < threshold] = 0
    return np.sqrt(np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y)))

def optimize_arrangement(initial_arrangement, segments, initial_score):
    current = copy.deepcopy(initial_arrangement)
    best = copy.deepcopy(initial_arrangement)
    current_score = initial_score
    best_score = current_score
    start_temp = 10
    end_temp = 1
    cooling_rate = 0.995
    temp = start_temp
    while temp > end_temp:
        i1, j1 = random.randint(0, 3), random.randint(0, 3)
        i2, j2 = random.randint(0, 3), random.randint(0, 3)
        current[i1][j1], current[i2][j2] = current[i2][j2], current[i1][j1]
        new_score = evaluate_arrangement(current, segments)
        if new_score < current_score or random.random() < math.exp((current_score - new_score) / temp):
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best = copy.deepcopy(current)
        else:
            current[i1][j1], current[i2][j2] = current[i2][j2], current[i1][j1]
        temp *= cooling_rate
    return best, best_score

def display_comparison(img1, img2, title1, title2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(img1, cmap="gray")
    ax1.set_title(title1)
    ax1.axis('off')
    
    ax2.imshow(img2, cmap="gray")
    ax2.set_title(title2)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Starting image reconstruction process...")
    matrix = read_matrix_file("./Lab 4/jigsaw/scrambled.mat")
    print(f"Loaded matrix shape: {matrix.shape}")
    matrix = matrix.T
    segments, initial_layout = split_into_segments(matrix)
    print(f"Number of segments: {len(segments)}")
    
    optimal_layout = None
    optimal_score = float('inf')
    initial_image = None
    
    print("Beginning optimization process...")
    for start_segment in range(16):
        layout = [[-1 for _ in range(4)] for _ in range(4)]
        layout[0][0] = start_segment
        remaining_segments = list(range(16))
        remaining_segments.remove(start_segment)
        breadth_first_arrangement(layout, segments, remaining_segments)
        
        temp_layout = copy.deepcopy(layout)
        temp_image = assemble_image(segments, temp_layout)
        
        if start_segment == 0:
            initial_image = temp_image
            print(f"Initial edge strength (iteration 0): {compute_edge_strength(initial_image):.2f}")
        
        score = compute_edge_strength(temp_image)
        improved_layout, new_score = optimize_arrangement(temp_layout, segments, score)
        
        print(f"Start segment: {start_segment}, Initial score: {score:.2f}, Improved score: {new_score:.2f}")
        
        if score < optimal_score:
            optimal_layout = layout
            optimal_score = score
        
        best_image = assemble_image(segments, optimal_layout)
        new_score = compute_edge_strength(best_image)
        if new_score < optimal_score:
            optimal_layout = improved_layout
            optimal_score = new_score
    
    final_image = assemble_image(segments, optimal_layout)
    final_score = compute_edge_strength(final_image)
    
    print(f"\nInitial edge strength: {compute_edge_strength(initial_image):.2f}")
    print(f"Final edge strength: {final_score:.2f}")
    print(f"Improvement: {(compute_edge_strength(initial_image) - final_score) / compute_edge_strength(initial_image) * 100:.2f}%")
    
    print("\nDisplaying comparison between initial and final images...")
    display_comparison(initial_image, final_image, "Initial Reconstruction", "Final Reconstruction")
    
    print("\nDisplaying final reconstructed image...")
    display_image(final_image)