import torch


def batch_rodrigues(rot_vecs):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3 array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    flatten_K = [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros]
    K = torch.cat(flatten_K, dim=1).reshape((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)[None, ...]
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def get_global_rotation(rotation, parents):
    root_rot = rotation[0]
    for i in parents[1:]:
        root_rot = torch.matmul(root_rot, rotation[i])

    return root_rot
