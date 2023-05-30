import os 
import torch
import scipy as sp
import numpy as np
import argparse
import trimesh


def calculate_iou(gt, prediction):
    intersection = torch.logical_and(gt, prediction)
    union = torch.logical_or(gt, prediction)
    return torch.sum(intersection) / torch.sum(union)

def compute_surface_metrics(mesh_pred, mesh_gt):
    """Compute surface metrics (chamfer distance and f-score) for one example.
    Args:
    mesh: trimesh.Trimesh, the mesh to evaluate.
    Returns:
    chamfer: float, chamfer distance.
    fscore: float, f-score.
    """
    # Chamfer
    eval_points = 1000000

    point_gt, idx_gt = mesh_gt.sample(eval_points, return_index=True)
    normal_gt = mesh_gt.face_normals[idx_gt]
    point_gt = point_gt.astype(np.float32)

    point_pred, idx_pred = mesh_pred.sample(eval_points, return_index=True)
    normal_pred = mesh_pred.face_normals[idx_pred]
    point_pred = point_pred.astype(np.float32)

    dist_pred_to_gt, normal_pred_to_gt = distance_field_helper(point_pred, point_gt, normal_pred, normal_gt)
    dist_gt_to_pred, normal_gt_to_pred = distance_field_helper(point_gt, point_pred, normal_gt, normal_pred)

    # TODO: subdivide by 2 following OccNet 
    # https://github.com/autonomousvision/occupancy_networks/blob/406f79468fb8b57b3e76816aaa73b1915c53ad22/im2mesh/eval.py#L136
    chamfer_l1 = np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)

    c1 = np.mean(dist_pred_to_gt)
    c2 = np.mean(dist_gt_to_pred)

    normal_consistency = np.mean(normal_pred_to_gt) + np.mean(normal_gt_to_pred)

    # Fscore
    tau = 1e-4
    eps = 1e-9

    dist_pred_to_gt = (dist_pred_to_gt**2)
    dist_gt_to_pred = (dist_gt_to_pred**2)

    prec_tau = (dist_pred_to_gt <= tau).astype(np.float32).mean() * 100.
    recall_tau = (dist_gt_to_pred <= tau).astype(np.float32).mean() * 100.

    fscore = (2 * prec_tau * recall_tau) / max(prec_tau + recall_tau, eps)

    # Following the tradition to scale chamfer distance up by 10.
    return c1 * 1000., c2 * 1000., normal_consistency / 2., fscore

def distance_field_helper(source, target, normals_src=None, normals_tgt=None):
    target_kdtree = sp.spatial.cKDTree(target)
    distances, idx = target_kdtree.query(source, n_jobs=-1)

    if normals_src is not None and normals_tgt is not None:
        
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)

    else:
        normals_dot_product = np.array(
            [np.nan] * source.shape[0], dtype=np.float32)

    return distances, normals_dot_product



def main(args):

    input_subfolder =  [x for x in sorted(os.listdir(args.input_path)) if x.endswith('obj')]
    gt_subfolder = [x for x in sorted(os.listdir(args.gt_path)) if x.endswith('obj')]

    mean_c1 = 0.
    mean_c2 = 0.
    mean_fscore = 0.
    mean_normal_consistency = 0.

    for pred, gt in zip(input_subfolder, gt_subfolder):
        mesh_pred = trimesh.load(os.path.join(args.input_path, pred))
        mesh_gt = trimesh.load(os.path.join(args.gt_path, gt))

        pred_2_scan, scan_2_pred, normal_consistency, fscore = compute_surface_metrics(mesh_pred, mesh_gt)
        print('Chamfer: {:.3f}, {:.3f}, Normal Consistency: {:.3f}, Fscore: {:.3f}'.format(pred_2_scan, scan_2_pred, normal_consistency, fscore))
        mean_c1 += pred_2_scan
        mean_c2 += scan_2_pred
        mean_fscore += fscore
        mean_normal_consistency += normal_consistency
    
    mean_c1 /= len(input_subfolder)
    mean_c2 /= len(input_subfolder)
    mean_fscore /= len(input_subfolder)
    mean_normal_consistency /= len(input_subfolder)
    print('Mean Chamfer: {:.3f}, {:.3f}, Normal Consistency: {:.3f}, Fscore: {:.3f}'.format(mean_c1, mean_c2, mean_normal_consistency, mean_fscore))
    print('{:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(mean_c1, mean_c2, mean_normal_consistency, mean_fscore))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', required=True ,type=str)
    parser.add_argument('-g', '--gt_path', required=True ,type=str)

    main(parser.parse_args())
