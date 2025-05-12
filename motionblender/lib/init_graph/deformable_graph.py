import fpsample
from collections import Counter
from loguru import logger as guru
try:
    from cuml import DBSCAN 
except Exception as e:
    guru.warning(str(e))
    guru.warning("skipping this exception to continue the code without `cuml`")
import cupy as cp
import networkx as nx
import numpy as np


def _build_deformable_graph(dists, min_num_edges: int):
    degree = np.zeros([len(dists)], dtype=np.int32)
    edge_hashset = set()
    isolated_node_mask = (degree == 0) | (degree == 1)  
    triangles = []
    argsort_cache = {}

    while (len(triangles) * 2) < min_num_edges or isolated_node_mask.sum() > 0:
        if isolated_node_mask.sum() > 0:
            isolated_node_idxes = isolated_node_mask.nonzero()[0]
            source_idx = np.random.choice(isolated_node_idxes)
        else:
            source_idx = np.random.randint(len(dists))
        
        if source_idx not in argsort_cache: argsort_cache[source_idx] = (np.argsort(dists[source_idx]), 0)

        target_indices, start = argsort_cache[source_idx]
        target_idxes = []
        for target_idx in target_indices[start:]:
            start += 1
            if source_idx == target_idx: continue
            if (source_idx, target_idx) not in edge_hashset and (target_idx, source_idx) not in edge_hashset:
                target_idxes.append(target_idx)
                edge_hashset.add((source_idx, target_idx))
                edge_hashset.add((target_idx, source_idx))
            if len(target_idxes) == 2: break

        argsort_cache[source_idx] = (target_indices, start)
        if len(target_idxes) < 2: continue

        degree[source_idx] += 2
        for t in target_idxes: degree[t] += 1
        triangles.append([[source_idx, t] for t in target_idxes])
        isolated_node_mask = (degree == 0) | (degree == 1)
    
    return np.array(triangles)


def build_deformable_graph_cc1(pts: np.ndarray, num_vertices: int|None=None, min_num_edges: int=-1, step_size: int = 5):
    if num_vertices is None: num_vertices = len(pts)
    if min_num_edges <= 0: min_num_edges = num_vertices
    # assert num_vertices <= 3000, "Too many vertices, it will be too slow to compute"
    num_vertices = min(len(pts), num_vertices)
    fps_samples_idx = fpsample.fps_sampling(pts, num_vertices)
    pts = pts[fps_samples_idx]
    dists = np.linalg.norm(pts[:, None] - pts[None], axis=-1)

    num_ccs = 0 
    while num_ccs != 1:
        edge_pairs = _build_deformable_graph(dists, min_num_edges) # (num_of_triangles, 2, 2)

        G = nx.Graph()
        G.add_nodes_from(list(range(len(pts))))
        G.add_edges_from(edge_pairs.reshape(-1, 2))
        num_ccs = nx.number_connected_components(G)

        guru.info(f"min_num_edges: {min_num_edges}, num_ccs: {num_ccs}")
        min_num_edges = max(min_num_edges + step_size, len(edge_pairs)*2)

    return pts, edge_pairs


def remove_outliers_from_pointcloud(tensor_pts, tensor_rgbs, voxel_downsample: bool=True, voxel_downsample_size=0.05, 
                                    statistical_outlier_kwargs={}, 
                                    radius_outlier_kwargs={}, cluster_kwargs={}, skip_clusters=False, skip_radius_outlier=False, skip_statistical_outlier=False):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor_pts.numpy())
    pcd.colors = o3d.utility.Vector3dVector(tensor_rgbs.numpy())

    debug_pcds = [pcd]

    statistical_outlier_kwargs_ = {'nb_neighbors': 30, 'std_ratio': 2.0, **statistical_outlier_kwargs}
    radius_outlier_kwargs_ = {'nb_points': 16, 'radius': 0.05, **radius_outlier_kwargs}
    cluster_kwargs_ = {'min_samples': 5000, 'eps': 0.2, **cluster_kwargs}
        

    if not skip_statistical_outlier:
        cl, ind = pcd.remove_statistical_outlier(**statistical_outlier_kwargs_)
        guru.info(f"remove_statistical_outlier: {len(pcd.points)} -> {len(ind)}")
        inlier_cloud = pcd.select_by_index(ind)
        global_indices = np.array(ind)
    else:
        inlier_cloud = pcd
        global_indices = np.arange(len(pcd.points))

    if not skip_radius_outlier:
        cl, ind = inlier_cloud.remove_radius_outlier(**radius_outlier_kwargs_)
        guru.info(f"remove_radius_outlier: {len(inlier_cloud.points)} -> {len(ind)}")
        inlier_cloud = inlier_cloud.select_by_index(ind)
        global_indices = global_indices[ind]
        debug_pcds.append(inlier_cloud)
    else:
        debug_pcds.append(inlier_cloud)

    # labels = inlier_cloud.cluster_dbscan(eps=0.2, min_points=min_cluster_size)
    # labels = np.asarray(labels)
    if not skip_clusters:
        db_gpu = DBSCAN(**cluster_kwargs_)
        labels = cp.asnumpy(db_gpu.fit_predict(cp.asarray(inlier_cloud.points)))

        mask = labels == Counter(labels).most_common()[0][0]
        ind = mask.nonzero()[0]
        guru.info(f"cluster_dbscan: {len(inlier_cloud.points)} -> {len(ind)}")
        inlier_cloud = inlier_cloud.select_by_index(ind)
        global_indices = global_indices[ind]
        debug_pcds.append(inlier_cloud)

    if voxel_downsample:
        inlier_cloud_ = inlier_cloud.voxel_down_sample(voxel_size=voxel_downsample_size)
        guru.info(f"voxel_downsample: {len(inlier_cloud.points)} -> {len(inlier_cloud_.points)}, note that indices are not traced in downsample!")
        inlier_cloud = inlier_cloud_
        debug_pcds.append(inlier_cloud)
    return debug_pcds, global_indices


def build_deformable_graph_from_dense_points(pcd, num_vertices: int=None):
    import open3d as o3d
    if isinstance(pcd, o3d.geometry.PointCloud):
        pts = np.asarray(pcd.points)
    else:
        pts = pcd
    deformed_graph = build_deformable_graph_cc1(pts, num_vertices=num_vertices)
    return  deformed_graph