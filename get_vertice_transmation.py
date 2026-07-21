"""Pure PyTorch MHR body model: given coefficients → returns vertices and 4x4 transformation matrices.

This module depends ONLY on PyTorch and numpy (for asset loading). No pymomentum required.

The transformation pipeline:
  identity_coeffs[45] + model_parameters[204] + face_expr_coeffs[72]
  → Step 1: BlendShape → rest_pose vertices
  → Step 2: ParameterTransform → joint_parameters
  → Step 3: Forward Kinematics → global skeleton state
  → Step 4: Pose Correctives (optional) → corrective offsets
  → Step 5: Linear Blend Skinning → final vertices

Returns:
  vertices: [B, V, 3]
  transform_matrices: [B, J, 4, 4] — per-joint deformation matrices D_j = G_j * IBP_j
"""

import os
import math

import numpy as np
import torch

NUM_IDENTITY_BLENDSHAPES = 45
NUM_FACE_EXPRESSION_BLENDSHAPES = 72
NUM_JOINTS = 127
NUM_VERTICES_LOD1 = 18439
PARAMETERS_PER_JOINT = 7

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "mhr_standalone")


def _quat_to_rot_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w, x, y, z] convention to 3x3 rotation matrix.

    Note: pymomentum uses [x, y, z, w] quaternion convention.
    Args:
        q: [B, J, 4] or [J, 4] quaternions in (x, y, z, w) order
    Returns:
        R: [B, J, 3, 3] or [J, 3, 3] rotation matrices
    """
    # Reorder from (x,y,z,w) to (w,x,y,z) for standard formula
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    two_qx = 2 * qx
    two_qy = 2 * qy
    two_qz = 2 * qz

    row0 = torch.stack([
        1 - two_qy * qy - two_qz * qz,
        two_qx * qy - two_qz * qw,
        two_qx * qz + two_qy * qw,
    ], dim=-1)
    row1 = torch.stack([
        two_qx * qy + two_qz * qw,
        1 - two_qx * qx - two_qz * qz,
        two_qy * qz - two_qx * qw,
    ], dim=-1)
    row2 = torch.stack([
        two_qx * qz - two_qy * qw,
        two_qy * qz + two_qx * qw,
        1 - two_qx * qx - two_qy * qy,
    ], dim=-1)

    return torch.stack([row0, row1, row2], dim=-2)


def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in (x, y, z, w) convention.

    Args:
        q1, q2: [..., 4] quaternions in (x, y, z, w) order
    Returns:
        q_out: [..., 4] quaternion product in (x, y, z, w) order
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def _euler_xyz_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    """Convert Euler XYZ angles to quaternion in (x, y, z, w) convention.

    Args:
        euler: [..., 3] Euler angles in radians
    Returns:
        q: [..., 4] quaternion in (x, y, z, w) order
    """
    rx, ry, rz = euler[..., 0], euler[..., 1], euler[..., 2]
    cx, sx = torch.cos(rx / 2), torch.sin(rx / 2)
    cy, sy = torch.cos(ry / 2), torch.sin(ry / 2)
    cz, sz = torch.cos(rz / 2), torch.sin(rz / 2)

    # XYZ Euler rotation quaternion (x,y,z,w convention)
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    qw = cx * cy * cz + sx * sy * sz

    return torch.stack([qx, qy, qz, qw], dim=-1)


def _batch6d_from_xyz(r: torch.Tensor) -> torch.Tensor:
    """Euler XYZ → 6D rotation representation (two columns of rotation matrix).

    Args:
        r: [B, 125, 3] Euler angles
    Returns:
        6d: [B, 125, 6] 6D representation (first two columns of rotation matrix)
    """
    rc = torch.cos(r)
    rs = torch.sin(r)
    cx, cy, cz = rc[..., 0], rc[..., 1], rc[..., 2]
    sx, sy, sz = rs[..., 0], rs[..., 1], rs[..., 2]

    result = torch.stack([
        cy * cz,
        -cx * sz + sx * sy * cz,
        sx * sz + cx * sy * cz,
        cy * sz,
        cx * cz + sx * sy * sz,
        -sx * cz + cx * sy * sz,
        -sy,
        sx * cy,
        cx * cy,
    ], dim=-1).reshape(list(r.shape[:-1]) + [3, 3])

    return torch.cat([result[..., :, 0], result[..., :, 1]], dim=-1)


class SparseLinear(torch.nn.Module):
    """Sparse linear layer matching pymomentum's implementation."""

    def __init__(self, in_channels, out_channels, sparse_indices, sparse_weight, dense_weight):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparse_indices = torch.nn.Parameter(sparse_indices, requires_grad=False)
        self.sparse_shape = (out_channels, in_channels)
        self.sparse_weight = torch.nn.Parameter(sparse_weight, requires_grad=False)
        self.register_buffer("dense_weight", dense_weight, persistent=False)

    def forward(self, x):
        dense = torch.zeros_like(self.dense_weight)
        dense[self.sparse_indices[0], self.sparse_indices[1]] = self.sparse_weight
        return (dense @ x.T).T


class MHRTorchModel:
    """Pure PyTorch MHR body model. No pymomentum dependency.

    Given MHR coefficients, returns vertices and per-joint 4x4 transformation matrices.
    """

    def __init__(
        self,
        assets_dir: str = ASSETS_DIR,
        lod: int = 1,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lod = lod
        self.device = device
        self.n_joints = NUM_JOINTS

        # Load assets
        npz_path = os.path.join(assets_dir, f"mhr_assets_lod{lod}.npz")
        data = np.load(npz_path, allow_pickle=False)

        # Step 1: BlendShape
        self.base_shape = torch.from_numpy(data["base_shape"]).to(device)
        self.identity_sv = torch.from_numpy(data["identity_shape_vectors"]).to(device)
        self.face_expr_sv = torch.from_numpy(data["face_expr_shape_vectors"]).to(device)

        # Step 2: ParameterTransform
        self.parameter_transform = torch.from_numpy(data["parameter_transform"]).to(device)

        # Step 3: Skeleton (for FK)
        self.joint_parents = torch.from_numpy(data["joint_parents"]).to(device)
        self.joint_offsets = torch.from_numpy(data["joint_offsets"]).to(device)
        self.joint_prerotations = torch.from_numpy(data["joint_prerotations"]).to(device)

        # Step 4: Pose Correctives
        self._build_pose_correctives(assets_dir, lod, device)

        # Step 5: LBS
        self.dense_skin_weights = torch.from_numpy(data["dense_skin_weights"]).to(device)
        self.inverse_bind_pose = torch.from_numpy(data["inverse_bind_pose"]).to(device)

        # Mesh faces (for reference, not used in computation)
        self.faces = torch.from_numpy(data["faces"]).to(device)

        # Pre-compute inverse bind pose 4x4 matrices
        self.ibp_matrices = self._pose_quat_scale_to_matrix(self.inverse_bind_pose)

        # Pre-compute parent ordering for FK (sort joints by depth)
        self._precompute_fk_order()

    def _build_pose_correctives(self, assets_dir, lod, device):
        blendshapes_path = os.path.join(assets_dir, f"corrective_blendshapes_lod{lod}.npz")
        activation_path = os.path.join(assets_dir, "corrective_activation.npz")

        if not os.path.exists(blendshapes_path) or not os.path.exists(activation_path):
            self.pose_correctives = None
            return

        bs_data = np.load(blendshapes_path)
        if "corrective_blendshapes" not in bs_data:
            self.pose_correctives = None
            return

        act_data = np.load(activation_path)
        n_components = bs_data["corrective_blendshapes"].shape[0]
        n_verts = bs_data["corrective_blendshapes"].shape[1]

        # Build SparseLinear layer
        sparse_indices = torch.from_numpy(act_data["0.sparse_indices"]).to(device)
        sparse_weight = torch.from_numpy(act_data["0.sparse_weight"]).to(device)
        sparse_mask = torch.from_numpy(act_data["posedirs_sparse_mask"]).to(device)
        in_features = 125 * 6  # (n_joints - 2) * 6D rotation

        # Build dense weight buffer
        dense_weight = torch.zeros(sparse_mask.shape[0], sparse_mask.shape[1], device=device)

        sparse_linear = SparseLinear(
            in_features,
            125 * 24,
            sparse_indices,
            sparse_weight,
            dense_weight,
        )

        # Linear layer
        weight2 = torch.from_numpy(
            bs_data["corrective_blendshapes"].reshape((n_components, -1)).T
        ).to(device)

        linear2 = torch.nn.Linear(125 * 24, n_verts * 3, bias=False)
        linear2.weight = torch.nn.Parameter(weight2, requires_grad=False)

        self.pose_correctives = torch.nn.Sequential(sparse_linear, torch.nn.ReLU(), linear2).to(device)
        for p in self.pose_correctives.parameters():
            p.requires_grad = False

    def _precompute_fk_order(self):
        """Sort joints by depth for sequential FK traversal."""
        parents = self.joint_parents.cpu().numpy()
        depth = np.zeros(len(parents), dtype=np.int32)
        for i in range(len(parents)):
            p = parents[i]
            d = 0
            while p >= 0:
                d += 1
                p = parents[p]
            depth[i] = d
        self.fk_order = np.argsort(depth)
        self.joint_depth = depth

    def _pose_quat_scale_to_matrix(self, pqs: torch.Tensor) -> torch.Tensor:
        """Convert [position(3), quaternion(4), scale(1)] to 4x4 matrix.

        Args:
            pqs: [J, 8] or [B, J, 8]
        Returns:
            M: [J, 4, 4] or [B, J, 4, 4]
        """
        has_batch = pqs.dim() == 3
        if not has_batch:
            pqs = pqs.unsqueeze(0)

        t = pqs[..., :3]  # [B, J, 3]
        q = pqs[..., 3:7]  # [B, J, 4] (x,y,z,w)
        s = pqs[..., 7]    # [B, J] scalar

        R = _quat_to_rot_matrix(q)  # [B, J, 3, 3]
        sR = s.unsqueeze(-1).unsqueeze(-1) * R  # [B, J, 3, 3]

        # Build 4x4 matrix without inplace operations (for autograd compatibility)
        row0 = torch.cat([sR[..., 0, :], t[..., 0:1]], dim=-1)  # [B, J, 4]
        row1 = torch.cat([sR[..., 1, :], t[..., 1:2]], dim=-1)  # [B, J, 4]
        row2 = torch.cat([sR[..., 2, :], t[..., 2:3]], dim=-1)  # [B, J, 4]
        zeros3 = torch.zeros_like(t[..., :3])
        ones1 = torch.ones_like(t[..., 0:1])
        row3 = torch.cat([zeros3, ones1], dim=-1)  # [0, 0, 0, 1]
        M = torch.stack([row0, row1, row2, row3], dim=-2)  # [B, J, 4, 4]

        if not has_batch:
            M = M.squeeze(0)
        return M

    def blend_shape(self, identity_coeffs: torch.Tensor, face_expr_coeffs: torch.Tensor) -> torch.Tensor:
        """Step 1: Compute rest pose from blendshapes.

        Args:
            identity_coeffs: [B, 45] or [1, 45]
            face_expr_coeffs: [B, 72]
        Returns:
            rest_pose: [B, V, 3]
        """
        if identity_coeffs.dim() == 1:
            identity_coeffs = identity_coeffs.unsqueeze(0)

        # Identity blendshapes
        id_rest = torch.einsum("nvd, ...n -> ...vd", self.identity_sv, identity_coeffs)
        id_rest = id_rest + self.base_shape

        # Face expression blendshapes
        if face_expr_coeffs is not None:
            if face_expr_coeffs.dim() == 1:
                face_expr_coeffs = face_expr_coeffs.unsqueeze(0)
            face_expr_coeffs = face_expr_coeffs.expand(id_rest.shape[0], -1)
            face_offsets = torch.einsum("nvd, ...n -> ...vd", self.face_expr_sv, face_expr_coeffs)
            id_rest = id_rest + face_offsets

        return id_rest

    def parameter_transform_forward(self, model_parameters: torch.Tensor) -> torch.Tensor:
        """Step 2: Convert model parameters to joint parameters.

        Args:
            model_parameters: [B, 204]
        Returns:
            joint_parameters: [B, 889]
        """
        if model_parameters.dim() == 1:
            model_parameters = model_parameters.unsqueeze(0)
        return torch.einsum("dn, ...n -> ...d", self.parameter_transform, model_parameters)

    def forward_kinematics(self, joint_parameters: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step 3: Joint parameters → global skeleton state.

        Args:
            joint_parameters: [B, 889] = [B, 127*7]
        Returns:
            skel_state: [B, 127, 8] — global (position, quaternion, scale) per joint
            global_matrices: [B, 127, 4, 4] — global 4x4 transform per joint
        """
        if joint_parameters.dim() == 1:
            joint_parameters = joint_parameters.unsqueeze(0)

        B = joint_parameters.shape[0]
        jp = joint_parameters.reshape(B, self.n_joints, PARAMETERS_PER_JOINT)

        # Compute local skeleton state
        # Translation: joint_params[:3] + offsets
        local_t = jp[..., :3] + self.joint_offsets.unsqueeze(0)

        # Rotation: euler_xyz_to_quaternion(joint_params[3:6]), then multiply with prerotation
        local_q = _euler_xyz_to_quaternion(jp[..., 3:6])
        local_q = _quat_multiply(self.joint_prerotations.unsqueeze(0).expand(B, -1, -1), local_q)

        # Scale: exp(ln(2) * joint_params[6]) = 2^joint_params[6]
        local_s = torch.exp(jp[..., 6] * 0.69314718246459961)

        # Build local skeleton state [B, 127, 8]
        local_skel = torch.cat([
            local_t,       # [B, 127, 3]
            local_q,       # [B, 127, 4]
            local_s.unsqueeze(-1),  # [B, 127, 1]
        ], dim=-1)

        # Forward kinematics: accumulate global transforms
        # Use lists to avoid inplace operations that break autograd
        parents_cpu = self.joint_parents.cpu().numpy()
        global_t_list = [None] * self.n_joints
        global_q_list = [None] * self.n_joints
        global_s_list = [None] * self.n_joints

        # Process joints in order of depth (FK order)
        for j in self.fk_order:
            p = parents_cpu[j]
            if p < 0:  # root joint
                global_t_list[j] = local_t[:, j]
                global_q_list[j] = local_q[:, j]
                global_s_list[j] = local_s[:, j]
            else:
                # global_position[j] = global_position[p] + global_scale[p] * rotate(global_rotation[p], local_position[j])
                R_p = _quat_to_rot_matrix(global_q_list[p])  # [B, 3, 3]
                rotated_local_t = torch.einsum("bij, bj -> bi", R_p, local_t[:, j])
                global_t_list[j] = global_t_list[p] + global_s_list[p].unsqueeze(-1) * rotated_local_t
                # global_rotation[j] = global_rotation[p] * local_rotation[j]
                global_q_list[j] = _quat_multiply(global_q_list[p], local_q[:, j])
                # global_scale[j] = global_scale[p] * local_scale[j]
                global_s_list[j] = global_s_list[p] * local_s[:, j]

        # Stack lists into tensors
        global_t = torch.stack(global_t_list, dim=1)   # [B, 127, 3]
        global_q = torch.stack(global_q_list, dim=1)   # [B, 127, 4]
        global_s = torch.stack(global_s_list, dim=1)   # [B, 127]

        # Normalize quaternions
        q_norm = torch.norm(global_q, dim=-1, keepdim=True)
        global_q = global_q / q_norm.clamp(min=1e-8)

        # Build global skeleton state [B, 127, 8]
        skel_state = torch.cat([
            global_t,
            global_q,
            global_s.unsqueeze(-1),
        ], dim=-1)

        # Build global 4x4 matrices [B, 127, 4, 4]
        global_matrices = self._pose_quat_scale_to_matrix(skel_state)

        return skel_state, global_matrices

    def pose_correctives_forward(self, joint_parameters: torch.Tensor) -> torch.Tensor:
        """Step 4: Compute pose corrective offsets.

        Args:
            joint_parameters: [B, 889]
        Returns:
            offsets: [B, V, 3]
        """
        if self.pose_correctives is None:
            return torch.zeros(
                joint_parameters.shape[0], NUM_VERTICES_LOD1, 3,
                device=self.device, dtype=joint_parameters.dtype,
            )

        if joint_parameters.dim() == 1:
            joint_parameters = joint_parameters.unsqueeze(0)

        jp = joint_parameters.reshape(joint_parameters.shape[0], self.n_joints, PARAMETERS_PER_JOINT)
        # Extract Euler rotations from joints 2 onwards (skip first 2 global joints)
        joint_euler = jp[:, 2:, 3:6]  # [B, 125, 3]
        pose_6d = _batch6d_from_xyz(joint_euler)  # [B, 125, 6]
        # Subtract identity diagonals (avoid inplace to preserve autograd)
        pose_6d = torch.cat([
            pose_6d[..., :1] - 1,
            pose_6d[..., 1:4],
            pose_6d[..., 4:5] - 1,
            pose_6d[..., 5:6],
        ], dim=-1)
        pose_6d_flat = pose_6d.flatten(1, 2)  # [B, 750]

        offsets = self.pose_correctives(pose_6d_flat)  # [B, V*3]
        return offsets.reshape(joint_parameters.shape[0], -1, 3)

    def linear_blend_skinning(
        self,
        global_matrices: torch.Tensor,
        rest_vertices: torch.Tensor,
    ) -> torch.Tensor:
        """Step 5: LBS using dense skin weights.

        Args:
            global_matrices: [B, 127, 4, 4] global joint transforms
            rest_vertices: [B, V, 3] vertices in rest pose
        Returns:
            posed_vertices: [B, V, 3]
        """
        if global_matrices.dim() == 2:
            global_matrices = global_matrices.unsqueeze(0)

        # Compute deformation matrices: D_j = G_j * IBP_j
        # IBP matrices are pre-computed as self.ibp_matrices [127, 4, 4]
        D = torch.matmul(global_matrices, self.ibp_matrices.unsqueeze(0).expand(global_matrices.shape[0], -1, -1, -1))
        # D: [B, 127, 4, 4]

        # Expand rest_vertices to homogeneous: [B, V, 4]
        V = rest_vertices.shape[1]
        v_homo = torch.cat([
            rest_vertices,
            torch.ones(rest_vertices.shape[0], V, 1, device=self.device, dtype=rest_vertices.dtype),
        ], dim=-1)

        # LBS: v_posed = sum_j w_j * D_j * v_rest
        # weights: [V, 127] → [1, V, 127, 1]
        w = self.dense_skin_weights.unsqueeze(0).unsqueeze(-1)  # [1, V, 127, 1]
        w = w.expand(rest_vertices.shape[0], -1, -1, -1)  # [B, V, 127, 1]

        # D_j * v_rest: [B, 127, 4, 4] @ [B, V, 4, 1] → need broadcasting
        # Reshape: D [B, 127, 4, 4], v [B, V, 4]
        # For each vertex v_i: sum over j of w_ij * D_j * v_i
        # Efficient: D_j * v_i = (D_j @ v_i_homo) where D_j [4,4], v_i [4,1]
        D_expanded = D.unsqueeze(1).expand(-1, V, -1, -1, -1)  # [B, V, 127, 4, 4]
        v_expanded = v_homo.unsqueeze(2).unsqueeze(-1)  # [B, V, 1, 4, 1]

        transformed = torch.matmul(D_expanded, v_expanded)  # [B, V, 127, 4, 1]
        weighted = w.unsqueeze(-2) * transformed  # [B, V, 127, 4, 1]
        v_posed = weighted.sum(dim=2)  # [B, V, 4, 1]

        return v_posed[..., :3, 0]  # [B, V, 3]

    def forward(
        self,
        identity_coeffs: torch.Tensor,
        model_parameters: torch.Tensor,
        face_expr_coeffs: torch.Tensor | None = None,
        apply_correctives: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full MHR pipeline: coefficients → (vertices, transform matrices).

        Args:
            identity_coeffs: [B, 45] or [1, 45]
            model_parameters: [B, 204]
            face_expr_coeffs: [B, 72] or None
            apply_correctives: whether to apply pose correctives

        Returns:
            vertices: [B, V, 3]
            transform_matrices: [B, J, 4, 4] — per-joint deformation matrices D_j = G_j * IBP_j
        """
        if identity_coeffs.dim() == 1:
            identity_coeffs = identity_coeffs.unsqueeze(0)
        if model_parameters.dim() == 1:
            model_parameters = model_parameters.unsqueeze(0)

        B = model_parameters.shape[0]
        identity_coeffs = identity_coeffs.expand(B, -1)

        if face_expr_coeffs is None:
            face_expr_coeffs = torch.zeros(B, NUM_FACE_EXPRESSION_BLENDSHAPES, device=self.device, dtype=model_parameters.dtype)
        elif face_expr_coeffs.dim() == 1:
            face_expr_coeffs = face_expr_coeffs.unsqueeze(0)

        # Step 1: BlendShape
        rest_pose = self.blend_shape(identity_coeffs, face_expr_coeffs)

        # Step 2: ParameterTransform
        joint_parameters = self.parameter_transform_forward(model_parameters)

        # Step 3: Forward Kinematics
        skel_state, global_matrices = self.forward_kinematics(joint_parameters)

        # Step 4: Pose Correctives
        unposed = rest_pose
        if apply_correctives and self.pose_correctives is not None:
            pose_offsets = self.pose_correctives_forward(joint_parameters)
            unposed = unposed + pose_offsets

        # Step 5: LBS
        vertices = self.linear_blend_skinning(global_matrices, unposed)

        # Compute deformation matrices D_j = G_j * IBP_j
        deform_matrices = torch.matmul(
            global_matrices,
            self.ibp_matrices.unsqueeze(0).expand(B, -1, -1, -1),
        )

        return vertices, deform_matrices


def load_model(
    assets_dir: str = ASSETS_DIR,
    lod: int = 1,
    device: torch.device | None = None,
) -> MHRTorchModel:
    """Convenience function to load the pure PyTorch MHR model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return MHRTorchModel(assets_dir=assets_dir, lod=lod, device=device)
