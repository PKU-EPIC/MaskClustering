import numpy as np
import pyviz3d.visualizer as viz
from utils.config import get_dataset, get_args
import open3d as o3d

# Since there are hundreds of objects in the scene, assigning visually distinguishable colors to each object is difficult. You can change the random seed to check if the two objects are actually segmented apart.
np.random.seed(4)

def vis_one_object(point_ids, scene_points):
    points = scene_points[point_ids]
    color = (np.random.rand(3) * 0.7 + 0.3) * 255
    colors = np.tile(color, (points.shape[0], 1))
    return point_ids, points, colors, color, np.mean(points, axis=0)


def main(args):
    point_size = 20
    label_colors, labels, centers = [], [], []
    dataset = get_dataset(args)
    mesh = o3d.io.read_triangle_mesh(dataset.mesh_path)
    scene_points = np.asarray(mesh.vertices)
    scene_points = scene_points - np.mean(scene_points, axis=0)
    scene_colors = np.asarray(mesh.vertex_colors)

    # Since the color of raw scan may be too dark, we brighten it tone mapping
    scene_colors = np.power(scene_colors, 1/2.2)
    scene_colors = scene_colors * 255

    instance_colors = np.zeros_like(scene_colors)

    v = viz.Visualizer()

    pred = np.load(f'data/prediction/{args.config}_class_agnostic/{args.seq_name}.npz')
    # pred = np.load(f'data/prediction/{args.config}/{args.seq_name}.npz')

    masks = pred['pred_masks']
    num_instances = masks.shape[1]
    for idx in range(num_instances):
        mask = masks[:, idx]
        point_ids = np.where(mask)[0]

        point_ids, points, colors, label_color, center = vis_one_object(point_ids, scene_points)
        instance_colors[point_ids] = label_color
        label_colors.append(label_color)
        labels.append(str(idx))
        centers.append(center)
        # If you want to visualize each object separately, you can uncomment the following line.
        # v.add_points(f'{idx}', points, colors, visible=True, point_size=point_size)

    v.add_points('RGB', scene_points, scene_colors, visible=False, point_size=point_size)

    labeled_scene_points_mask = np.where(np.sum(instance_colors, axis=1) != 0)
    v.add_points('Instances', scene_points[labeled_scene_points_mask], instance_colors[labeled_scene_points_mask], visible=True, point_size=point_size)

    # If you want to visualize the label id of each object, you can uncomment the following line.
    # v.add_labels('Labels', labels, centers, label_colors)

    v.save(f'data/vis/{args.seq_name}')


if __name__ == '__main__':
    args = get_args()
    main(args)