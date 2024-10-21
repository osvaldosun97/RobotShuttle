import mmcv
from mmdet3d.apis import init_model, inference_detector
import open3d as o3d
import numpy as np
from math import cos, sin
import torch


# Step 1: Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 2: Define paths for the config and checkpoint files
config_file = './configs/second/second_hv_secfpn_8xb6-80e_kitti-3d-3class.py'
checkpoint_file = './checkpoints/second_hv_secfpn_8xb6-80e_kitti-3d-3class-b086d0a3.pth'

# Step 3: Initialize the model
model = init_model(config_file, checkpoint_file, device=device)

# Step 4: Load the LiDAR point cloud sample from .pcd file
# Replace with your actual .pcd file path
pcd_file = './data/Robotshuttle/1726607290256893.pcd'

# Step 5: Load and visualize the point cloud using Open3D
def load_pcd_file(pcd_file):
    # Load the point cloud from .pcd file using Open3D
    pcd = o3d.io.read_point_cloud(pcd_file)
    point_cloud = np.asarray(pcd.points)  # Convert to numpy array
    
    # The point cloud needs to have shape (N, 4) where the 4th dimension can be zeros (if intensity data is missing)
    if point_cloud.shape[1] == 3:
        # Add intensity (or a dummy column of zeros)
        point_cloud = np.hstack([point_cloud, np.zeros((point_cloud.shape[0], 1))])
    
    # Return the correctly shaped point cloud
    return point_cloud

# Load the point cloud from the .pcd file
point_cloud = load_pcd_file(pcd_file)

# Convert to Open3D point cloud format
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])  # Only use x, y, z

# Convert the point cloud data into a format suitable for inference
data = dict(points=[torch.tensor(point_cloud, dtype=torch.float32).to(device)])  # Explicitly move to GPU

# Step 6: Perform inference (object detection)
result, data = inference_detector(model, data)

# Step 7: Process and display the detected objects
# Classes for the dataset (e.g., Car, Pedestrian, Cyclist)
class_names = model.dataset_meta['classes']
threshold = 0.5  # Confidence threshold for displaying the detected objects

print(f"Detected objects in {pcd_file}:")

# Helper function to compute the 8 corners of a bounding box
def compute_bounding_box_corners(bbox):
    bbox = bbox.tensor.cpu().numpy()  # Convert to numpy array

    all_corners = []
    for box in bbox:
        x, y, z, dx, dy, dz, heading = box

        rot_matrix = np.array(
            [[cos(heading), -sin(heading)], [sin(heading), cos(heading)]])

        corners = np.array([
            [-dx / 2, -dy / 2, -dz / 2], [dx / 2, -dy / 2, -dz / 2],
            [dx / 2, dy / 2, -dz / 2], [-dx / 2, dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, dz / 2], [dx / 2, -dy / 2, dz / 2],
            [dx / 2, dy / 2, dz / 2], [-dx / 2, dy / 2, dz / 2]
        ])

        rotated_corners = np.dot(corners[:, [0, 1]], rot_matrix)
        corners[:, 0:2] = rotated_corners

        corners[:, 0] += x
        corners[:, 1] += y
        corners[:, 2] += z

        all_corners.append(corners)

    return all_corners

# Helper function to create a 3D bounding box in Open3D
def create_bounding_box_lines(bbox):
    all_bbox_corners = compute_bounding_box_corners(bbox)

    line_sets = []
    for bbox_corners in all_bbox_corners:
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbox_corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector(
            [[0, 0, 1] for _ in range(len(edges))])

        line_sets.append(line_set)

    return line_sets


# Prepare bounding boxes for visualization
bounding_boxes = []
pred_instances = result.pred_instances_3d
for i in range(len(pred_instances.bboxes_3d)):
    bbox = pred_instances.bboxes_3d[i]  # Get the bounding box tensor
    score = pred_instances.scores_3d[i].numpy()  # Get confidence score
    label = pred_instances.labels_3d[i].numpy()  # Get class label

    class_name = class_names[label]
    print(f"Detected {class_name} with confidence {score}")
    bounding_box_lines = create_bounding_box_lines(bbox)
    bounding_boxes.extend(bounding_box_lines)

# Step 8: Visualize the point cloud with bounding boxes using Open3D
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

# Add all bounding boxes to the visualizer
for bbox in bounding_boxes:
    vis.add_geometry(bbox)

# Non-blocking visualization
while True:
    vis.poll_events()
    vis.update_renderer()
