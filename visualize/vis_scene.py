import numpy as np
import pyviz3d.visualizer as viz
from utils.config import get_dataset, get_args
import open3d as o3d

np.random.seed(1)

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
    scene_colors = np.asarray(mesh.vertex_colors) * 255
    instance_colors = np.zeros_like(scene_colors)

    v = viz.Visualizer()

    pred = np.load(f'data/prediction/{args.config}/{args.seq_name}.npz')
    masks = pred['pred_masks']
    num_instances = masks.shape[1]
    for idx in range(num_instances):
        mask = masks[:, idx]
        point_ids = np.where(mask)[0]

        point_ids, points, colors, label_color, center = vis_one_object(point_ids, scene_points)
        instance_colors[point_ids] = label_color
        # label_colors.append(label_color)
        # labels.append(str(idx))
        # centers.append(center)
        # v.add_points(f'{idx}', points, colors, visible=True, point_size=point_size)

    v.add_points('RGB', scene_points, scene_colors, visible=True, point_size=point_size)
    v.add_points('Instances', scene_points, instance_colors, visible=True, point_size=point_size)
    # v.add_labels('Labels', labels, centers, label_colors, visible=False)
    v.save(f'data/vis/{args.seq_name}')


if __name__ == '__main__':
    args = get_args()
    main(args)