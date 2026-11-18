import numpy as np
import torch
import torch.nn as nn


def meshing_uv_map(occ):
    pixel_index = np.arange(occ.size).reshape(occ.shape)

    # Determine triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=-1, axis=0) & np.roll(occ, shift=-1, axis=1)
    is_tri_vert[-1, :] = False
    is_tri_vert[:, -1] = False

    verta = pixel_index
    vertb = np.roll(pixel_index, shift=-1, axis=1)
    vertc = np.roll(pixel_index, shift=-1, axis=0)
    face0 = np.stack(
        [verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1
    )

    # Determine the second set of triangles' vertices
    is_tri_vert = occ & np.roll(occ, shift=1, axis=0) & np.roll(occ, shift=1, axis=1)
    is_tri_vert[0, :] = False
    is_tri_vert[:, 0] = False
    verta = pixel_index
    vertb = np.roll(pixel_index, shift=1, axis=1)
    vertc = np.roll(pixel_index, shift=1, axis=0)
    face1 = np.stack(
        [verta[is_tri_vert], vertb[is_tri_vert], vertc[is_tri_vert]], axis=1
    )

    # Combine the two sets of faces
    face = np.concatenate([face0, face1], axis=0)

    return face


def tensor2omg(omage_tensor):
    if omage_tensor.dtype == np.uint16:
        omage_tensor = omage_tensor.astype(np.float32) / 65535.0
    position_map = omage_tensor[..., :3]
    occupancy_map = omage_tensor[..., 3:4]

    omage = dict(position=position_map, occupancy=occupancy_map)
    return omage


def makeGrid(
    bb_min=[0, 0, 0],
    bb_max=[1, 1, 1],
    shape=[10, 10, 10],
    mode="on",
    flatten=True,
    indexing="ij",
):
    """Make a grid of coordinates
    Args:
        bb_min (list or np.array): least coordinate for each dimension
        bb_max (list or np.array): maximum coordinate for each dimension
        shape (list or int): list of coordinate number along each dimension. If it is an int, the number
                            same for all dimensions
        mode (str, optional): 'on' to have vertices lying on the boundary and
                              'in' for keeping vertices and its cell inside of the boundary
                              same as align_corners=True and align_corners=False
        flatten (bool, optional): Return as list of points or as a grid. Defaults to True.
        indexing (["ij" or "xy"]): default to "xy", see https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html
    Returns:
        np.array: return coordinates (XxYxZxD if flatten==False, X*Y*ZxD if flatten==True.
    """
    coords = []
    bb_min = np.array(bb_min)
    bb_max = np.array(bb_max)
    if type(shape) is int:
        shape = np.array([shape] * bb_min.shape[0])
    for i, si in enumerate(shape):
        if mode == "on":
            coord = np.linspace(bb_min[i], bb_max[i], si)
        elif mode == "in":
            offset = (bb_max[i] - bb_min[i]) / 2.0 / si
            # 2*si*w=bmax-bmin
            # w = (bmax-bmin)/2/si
            # start, end = bmax+w, bmin-w
            coord = np.linspace(bb_min[i] + offset, bb_max[i] - offset, si)
        coords.append(coord)
    grid = np.stack(np.meshgrid(*coords, sparse=False, indexing=indexing), axis=-1)
    if flatten == True:
        grid = grid.reshape(-1, grid.shape[-1])
    return grid


def meshing_objamage(omage):
    """Turn an omage into a mesh.
    Args:
        omage: dict, containing 'position', 'occupancy', 'objnormal'.
    """
    # Generate pixel indices array
    if type(omage) == np.ndarray:
        omage = tensor2omg(omage)
    vert = omage["position"][::-1]
    
    # Determine occupancy: use explicit occupancy if provided, otherwise infer from NaN or zeros
    if omage.get("occupancy", None) is not None:
        # Use explicit occupancy mask if provided
        occ = omage["occupancy"][::-1] > 0.5
    elif np.any(np.isnan(vert)):
        # If there are NaNs, use NaN-based occupancy detection
        occ = (1 - np.isnan(vert[..., 0:1])).astype(bool)
    else:
        # If no NaNs, use zero-based occupancy detection (points at origin are unoccupied)
        occ = (np.abs(vert).sum(axis=-1, keepdims=True) > 1e-6).astype(bool)
    
    objnormal = omage["objnormal"] if omage.get("objnormal", None) is not None else None
    vert = vert.reshape(-1, 3)
    objnormal = objnormal.reshape(-1, 3) if objnormal is not None else None

    face = meshing_uv_map(occ)
    if face.shape[0] == 0:  # no face, return empty mesh
        meshing_ret = dict(
            vert=np.zeros((0, 3)),
            face=np.zeros((0, 3)).astype(int),
            uv=np.zeros((0, 2)),
        )
        return meshing_ret

    # flip faces with inconsistent objnormal vs face normal
    if objnormal is not None:
        face_normal = np.cross(
            vert[face[:, 1]] - vert[face[:, 0]], vert[face[:, 2]] - vert[face[:, 0]]
        )
        flip_mask = np.einsum("ij,ij->i", face_normal, objnormal[face[:, 0]]) < 0
        face[flip_mask] = face[flip_mask][:, [0, 2, 1]]

    uv = makeGrid([0, 0], [1, 1], shape=(occ.shape[0], occ.shape[1]), mode="on")
    uv[..., [0, 1]] = uv[..., [1, 0]]  # swap x, y to match the image coordinate system

    # Remove faces that refer to vertices not in vert
    valid_faces_mask = np.all(face < vert.shape[0], axis=1)
    face = face[valid_faces_mask]

    meshing_ret = dict(vert=vert, face=face, uv=uv)

    return meshing_ret


class ChamferLoss(nn.Module):
    def __init__(self, metric="L2"):
        """
        Args:
            metric (str): Distance metric to use ('L1' or 'L2')
        """
        super(ChamferLoss, self).__init__()
        self.metric = metric.upper()
        if self.metric not in ["L1", "L2"]:
            raise ValueError(f"Unsupported metric: {metric}. Use 'L1' or 'L2'")

    def batch_pairwise_dist_L2(self, x, y):
        """Compute squared L2 distances between all point pairs"""
        xx = x.pow(2).sum(dim=2)
        yy = y.pow(2).sum(dim=2)
        zz = torch.bmm(x, y.transpose(2, 1))
        # just repeating of x_1^2, x_2^2 ... x_n^2
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = rx.transpose(2, 1) + ry - 2 * zz  # x^2 + y^2 - 2xy
        return P

    def batch_pairwise_dist_L1(self, x, y):
        """Compute L1 (Manhattan) distances between all point pairs"""
        batch_size = x.size(0)
        num_points_x = x.size(1)
        num_points_y = y.size(1)

        # Reshape for broadcasting: [B, N, 1, D] - [B, 1, M, D] → [B, N, M, D]
        x_expanded = x.unsqueeze(2).expand(batch_size, num_points_x, num_points_y, -1)
        y_expanded = y.unsqueeze(1).expand(batch_size, num_points_x, num_points_y, -1)

        # L1 distance: sum of absolute differences
        P = torch.abs(x_expanded - y_expanded).sum(dim=3)
        return P

    def forward(self, ground_truth, reconstruction):
        """
        Args:
            ground_truth: [B, N, 3]
            reconstruction: [B, M, 3]
        """
        if self.metric == "L2":
            P = self.batch_pairwise_dist_L2(ground_truth, reconstruction)
        else:  # L1
            P = self.batch_pairwise_dist_L1(ground_truth, reconstruction)

        mins, _ = torch.min(P, 1)  # from pred to gts
        loss_1 = torch.mean(torch.clamp(mins, min=1e-10))
        mins, _ = torch.min(P, 2)  # from gts to pred
        loss_2 = torch.mean(torch.clamp(mins, min=1e-10))
        return loss_1 + loss_2  # * 1000
