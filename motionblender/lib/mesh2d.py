from glob import glob
import os.path as osp
import torch
import numpy as np
from collections import defaultdict
from nanomesh import Image, Mesher2D


def generate_parallelogram_mesh_3d(top_left, top_right, bottom_right, nx=1, ny=1):
    """
    Generate triangular mesh for a 3D parallelogram using Nanomesh.
    Returns vertices, faces, and key point indices.
    """
    # Calculate vectors and fourth point
    tl = np.array(top_left)
    tr = np.array(top_right)
    br = np.array(bottom_right)
    bl = tl + br - tr

    # Create a 2D parametric grid (unit square)
    binary_image = np.zeros((ny+2, nx+2), dtype=np.uint8)
    binary_image[1:-1, 1:-1] = 1  # Create hollow rectangle

    # Generate mesh using Nanomesh
    image = Image(binary_image)
    mesher = Mesher2D(image)
    mesher.generate_contour() # min(np.linalg.norm(tl - tr) / (nx+2), np.linalg.norm(tr - br) / (ny+2))
    mesh = mesher.triangulate(opts='q30a10')

    # Get validated vertices and faces
    vertices_2d = mesh.points
    faces = mesh.cells_dict['triangle']

    # Map to 3D with duplicate checking
    u = tr - tl  # Vector along top edge
    v = br - tr
    u_scaled = u / (nx+1)
    v_scaled = v / (ny+1)
    vertices_3d = []
    seen = set()
    for x, y in vertices_2d[:, ::-1]:  # Swap x/y for correct mapping
        vertex = tl + x*u_scaled + y*v_scaled
        vertex_tuple = tuple(np.round(vertex, 9))  # Prevent float errors
        if vertex_tuple in seen:
            raise ValueError("Duplicate vertex generated")
        seen.add(vertex_tuple)
        vertices_3d.append(vertex)

    # Validate all edges have positive length
    edge_lengths = []
    for face in faces:
        if len(set(face)) != 3:
            raise ValueError(f"Degenerate face: {face}")
        for i in range(3):
            u_idx = face[i]
            v_idx = face[(i+1)%3]
            dist = np.linalg.norm(vertices_3d[u_idx] - vertices_3d[v_idx])
            if dist < 1e-9:
                raise ValueError(f"Zero-length edge between {u_idx} and {v_idx}")
            edge_lengths.append(dist)

       # Find original points using KD-tree
    indices = {}
    for name, p in [('top_left', tl), ('top_right', tr), ('bottom_right', br),  ('bottom_left', bl)]:
        indices[name] = int(np.linalg.norm(vertices_3d - p.reshape(1, 3), axis=1).argmin())
    
    return vertices_3d, mesh.cells_dict['triangle'], indices

def faces_to_edges(faces):
    """ [N, 3] -> [4N, 2]"""
    return torch.cat([
                    torch.cat([faces[:, None, :1], faces[:, None, 1:2]], dim=-1),
                    torch.cat([faces[:, None, :1], faces[:, None, 2:]], dim=-1), 

                    torch.cat([faces[:, None, 1:2], faces[:, None, 2:]], dim=-1), 
                    torch.cat([faces[:, None, 1:2], faces[:, None, :1]], dim=-1), 
                    ], dim=1).reshape(-1, 2)

def edges_to_faces(links):
    links = links.reshape(-1, 4, 2)
    return torch.cat([links[:, 0, :1], links[:, 0, 1:2], links[:, 1, 1:]], dim=-1).reshape(-1, 3)