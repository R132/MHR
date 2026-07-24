"""
Standalone MHR mesh → SMPLX parameter conversion (no pymomentum dependency).

Pipeline:
  1. Load MHR PLY meshes [B, 18439, 3] (already in meters, y-up)
  2. Convert m→cm (Conversion class expects cm input)
  3. Barycentric interpolation: MHR [18439] → SMPLX [10475] + cm→m
  4. Optimize SMPLX parameters (staged Adam) to fit target vertices
  5. Save SMPLX parameters, meshes, and errors

All dependencies: torch, numpy, trimesh, smplx, tqdm (no pymomentum).
"""

import os
import sys
import glob
import logging

import numpy as np
import torch
import torch.optim
import trimesh
import tqdm
import smplx

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---- Configuration ----
# Project root: parent of scripts/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = None  # Must be set via --input or the first positional arg
OUTPUT_DIR = None  # Defaults to <input_dir>/smplx_output if not set
SMPLX_MODEL_PATH = os.path.join(_PROJECT_ROOT, "assets", "smplx_models")
MHR_MAPPING_PATH = os.path.join(_PROJECT_ROOT, "tools/mhr_smpl_conversion/assets/mhr2smplx_mapping.npz")
MHR_ASSETS_PATH = os.path.join(_PROJECT_ROOT, "assets/mhr_standalone/mhr_assets_lod1.npz")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


# ============================================================
# Barycentric interpolation (MHR topology → SMPLX topology)
# ============================================================

def load_mhr_faces():
    """Load MHR mesh faces from standalone assets."""
    npz = np.load(MHR_ASSETS_PATH)
    return npz["faces"]  # [36874, 3]


def load_surface_mapping():
    """Load MHR→SMPLX barycentric mapping."""
    mapping = np.load(MHR_MAPPING_PATH)
    return mapping["triangle_ids"], mapping["baryc_coords"]


def barycentric_interpolation_mhr2smplx(mhr_vertices_cm, mhr_faces, triangle_ids, baryc_coords):
    """Map MHR vertices (cm) to SMPLX topology vertices (m) via barycentric interpolation.

    Args:
        mhr_vertices_cm: [B, 18439, 3] MHR vertices in centimeters
        mhr_faces: [36874, 3] MHR face connectivity
        triangle_ids: [10475] triangle index in MHR mesh per SMPLX vertex
        baryc_coords: [10475, 3] barycentric weights per SMPLX vertex

    Returns:
        smplx_vertices_m: [B, 10475, 3] SMPLX-topology vertices in meters
    """
    B = mhr_vertices_cm.shape[0]

    mhr_faces_t = torch.from_numpy(mhr_faces).long().to(mhr_vertices_cm.device)
    tri_ids_t = torch.from_numpy(triangle_ids).long().to(mhr_vertices_cm.device)
    baryc_t = torch.from_numpy(baryc_coords).float().to(mhr_vertices_cm.device)
    baryc_t = baryc_t[None, :, :, None]  # [1, 10475, 3, 1]

    # Convert cm → m first
    mhr_vertices_m = mhr_vertices_cm * 0.01

    # Get triangle vertices for each SMPLX vertex
    # triangles: [B, 10475, 3, 3] — for each SMPLX vertex, the 3 MHR vertices of its mapped triangle
    triangles = mhr_vertices_m[:, mhr_faces_t[tri_ids_t], :]

    # Interpolate: weighted sum of triangle vertices
    target_vertices = (triangles * baryc_t).sum(dim=2)  # [B, 10475, 3]

    return target_vertices


# ============================================================
# SMPLX parameter optimization (staged Adam fitting)
# ============================================================

class SMPLXFitting:
    """Optimize SMPLX parameters to fit target vertices."""

    def __init__(self, smplx_model, device="cuda", batch_size=32):
        self._smpl_model = smplx_model.to(device)
        self._device = device
        self._batch_size = batch_size
        self._smpl_model_type = "smplx"
        self._hand_pose_dim = 6 if smplx_model.use_pca else 45

        # Pre-compute SMPLX edges for edge loss
        v_template_np = smplx_model.v_template.detach().cpu().numpy() if isinstance(smplx_model.v_template, torch.Tensor) else smplx_model.v_template
        faces_np = smplx_model.faces if isinstance(smplx_model.faces, np.ndarray) else smplx_model.faces.cpu().numpy()
        tmp_mesh = trimesh.Trimesh(
            vertices=v_template_np,
            faces=faces_np,
            process=False,
        )
        self._smpl_edges = torch.from_numpy(tmp_mesh.edges_unique).long().to(device)

    def fit(self, target_vertices, single_identity=True):
        """Fit SMPLX parameters to target vertices.

        Args:
            target_vertices: [B, 10475, 3] in meters, SMPLX topology
            single_identity: use same betas for all frames

        Returns:
            dict of SMPLX parameters
        """
        num_frames = target_vertices.shape[0]

        # Center the target vertices for better optimization stability
        target_vertices_center = 0.5 * (
            target_vertices.min(dim=1)[0] + target_vertices.max(dim=1)[0]
        )
        target_vertices_centered = target_vertices - target_vertices_center[:, None, :]

        logger.info("Optimizing SMPLX parameters...")

        # Define trainable variables
        variables = self._define_trainable_variables(num_frames, single_identity)

        # Staged optimization
        variables = self._optimize_smpl(
            target_vertices_centered,
            variables,
            single_identity=single_identity,
        )

        # Restore center translation
        with torch.no_grad():
            variables["transl"] += target_vertices_center

        if single_identity:
            variables["betas"] = variables["betas"][:1].expand(num_frames, -1)

        # Complete SMPLX parameters (add jaw, eyes, etc. if missing)
        variables = self._complete_smplx_parameters(variables, num_frames)

        return variables

    def _define_trainable_variables(self, num_frames, single_identity):
        """Create trainable SMPLX parameter tensors."""
        num_identities = 1 if single_identity else num_frames

        global_orient = torch.zeros(num_frames, 3, device=self._device, requires_grad=True)
        transl = torch.zeros(num_frames, 3, device=self._device, requires_grad=False)
        body_pose = torch.zeros(num_frames, 63, device=self._device, requires_grad=True)
        betas = torch.zeros(num_identities, self._smpl_model.num_betas, device=self._device, requires_grad=True)
        left_hand_pose = torch.zeros(num_frames, self._hand_pose_dim, device=self._device, requires_grad=True)
        right_hand_pose = torch.zeros(num_frames, self._hand_pose_dim, device=self._device, requires_grad=True)
        expression = torch.zeros(num_frames, self._smpl_model.num_expression_coeffs, device=self._device, requires_grad=True)

        return {
            "global_orient": global_orient,
            "transl": transl,
            "body_pose": body_pose,
            "betas": betas,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "expression": expression,
        }

    def _optimize_smpl(self, target_vertices, variables, single_identity=True,
                       iterations=((40, 80, 40), 300), learning_rates=(0.1, 0.01)):
        """Staged SMPLX optimization: coarse pose → fine all params."""
        num_frames = target_vertices.shape[0]
        disable_inner = len(getattr(tqdm, "_instances", [])) > 0

        # Stage 1: Coarse — progressively add parameters
        optimizable_params = [
            ["global_orient"],
            ["global_orient", "body_pose", "betas", "expression"],
            ["global_orient", "body_pose", "betas", "expression", "left_hand_pose", "right_hand_pose"],
        ]
        coarse_iterations = iterations[0]

        for params_group, n_iters in zip(optimizable_params, coarse_iterations):
            optimizer = torch.optim.Adam(
                [variables[p] for p in params_group if p in variables],
                lr=learning_rates[0],
            )
            for batch_start in tqdm.tqdm(
                range(0, num_frames, self._batch_size),
                desc=f"Coarse ({','.join(params_group[:2])}...)",
                disable=disable_inner,
            ):
                batch_end = min(batch_start + self._batch_size, num_frames)
                target_batch = target_vertices[batch_start:batch_end]
                target_edge_vecs = self._compute_edge_vectors(target_batch, self._smpl_edges)

                for _ in range(n_iters):
                    loss = self._compute_loss(variables, batch_start, batch_end,
                                              target_edge_vecs, target_batch,
                                              edge_weight=1.0, vertex_weight=0.1)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            # Update translation after coarse pose
            with torch.no_grad():
                batch_params = self._get_batched_params(variables, 0, min(self._batch_size, num_frames))
                smpl_output = self._smpl_model(**batch_params)
                smpl_verts = smpl_output.vertices
                variables["transl"][:min(self._batch_size, num_frames)] += (
                    (target_vertices[:min(self._batch_size, num_frames)] - smpl_verts).mean(dim=1).detach()
                )
                # Update for remaining batches
                for bs in range(self._batch_size, num_frames, self._batch_size):
                    be = min(bs + self._batch_size, num_frames)
                    bp = self._get_batched_params(variables, bs, be)
                    sv = self._smpl_model(**bp).vertices
                    variables["transl"][bs:be] += (
                        (target_vertices[bs:be] - sv).mean(dim=1).detach()
                    )

        variables["transl"].requires_grad_(True)

        # Stage 2: Fine — optimize all parameters
        logger.info("Fine optimization (all parameters)...")
        optimizer = torch.optim.Adam(
            [v for v in variables.values()],
            lr=learning_rates[1],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 200], gamma=0.1,
        )
        fine_iterations = iterations[1]

        for epoch in tqdm.trange(fine_iterations, desc="Fine optimization", disable=disable_inner):
            edge_weight = 1.0 if epoch < 50 else 0.0
            vertex_weight = 0.1 if epoch < 50 else 1.0

            for batch_start in range(0, num_frames, self._batch_size):
                batch_end = min(batch_start + self._batch_size, num_frames)
                target_batch = target_vertices[batch_start:batch_end]
                target_edge_vecs = self._compute_edge_vectors(target_batch, self._smpl_edges)

                loss = self._compute_loss(variables, batch_start, batch_end,
                                          target_edge_vecs, target_batch,
                                          edge_weight=edge_weight,
                                          vertex_weight=vertex_weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

        return variables

    def _compute_loss(self, variables, batch_start, batch_end,
                      target_edge_vecs, target_verts_batch,
                      edge_weight=1.0, vertex_weight=1.0,
                      expression_reg_weight=1e4):
        """Compute edge + vertex + expression regularization loss."""
        batch_params = self._get_batched_params(variables, batch_start, batch_end)
        smpl_output = self._smpl_model(**batch_params)
        pred_verts = smpl_output.vertices  # [B, 10475, 3]

        # Vertex loss
        vertex_loss = torch.square(pred_verts - target_verts_batch).mean()

        # Edge loss (L1, more robust)
        pred_edge_vecs = self._compute_edge_vectors(pred_verts, self._smpl_edges)
        edge_loss = torch.abs(pred_edge_vecs - target_edge_vecs).mean()

        # Expression regularization (penalize large expressions)
        expr_reg = torch.square(variables["expression"][batch_start:batch_end]).mean()

        loss = edge_weight * edge_loss + vertex_weight * vertex_loss + expression_reg_weight * expr_reg
        return loss

    def _compute_edge_vectors(self, vertices, edges):
        """Compute edge vectors: vertices[:, edges[:,1], :] - vertices[:, edges[:,0], :]."""
        return vertices[:, edges[:, 1], :] - vertices[:, edges[:, 0], :]

    def _get_batched_params(self, variables, batch_start, batch_end):
        """Extract and complete SMPLX parameters for a batch."""
        batch_size = batch_end - batch_start
        params = {}
        for k, v in variables.items():
            if v.shape[0] == 1 and k == "betas":  # single identity
                params[k] = v.expand(batch_size, -1).to(self._device)
            elif v.shape[0] < batch_end:
                params[k] = v.expand(batch_size, -1).to(self._device)
            else:
                params[k] = v[batch_start:batch_end].to(self._device)

        # Complete missing SMPLX parameters
        if "jaw_pose" not in params:
            params["jaw_pose"] = torch.zeros([batch_size, 1, 3], device=self._device)
        if "leye_pose" not in params:
            params["leye_pose"] = torch.zeros([batch_size, 1, 3], device=self._device)
        if "reye_pose" not in params:
            params["reye_pose"] = torch.zeros([batch_size, 1, 3], device=self._device)

        return params

    def _complete_smplx_parameters(self, variables, num_frames):
        """Add default zero SMPLX-specific parameters."""
        if "jaw_pose" not in variables:
            variables["jaw_pose"] = torch.zeros(num_frames, 1, 3, device=self._device)
        if "leye_pose" not in variables:
            variables["leye_pose"] = torch.zeros(num_frames, 1, 3, device=self._device)
        if "reye_pose" not in variables:
            variables["reye_pose"] = torch.zeros(num_frames, 1, 3, device=self._device)
        return variables


def evaluate_fitting_error(smplx_model, params, target_vertices, device, batch_size=32):
    """Compute per-frame mean vertex distance error (in cm)."""
    num_frames = target_vertices.shape[0]
    errors = []

    with torch.no_grad():
        for bs in range(0, num_frames, batch_size):
            be = min(bs + batch_size, num_frames)
            batch_params = {}
            for k, v in params.items():
                if v.shape[0] == 1 and k == "betas":
                    batch_params[k] = v.expand(be - bs, -1).to(device)
                else:
                    batch_params[k] = v[bs:be].to(device)

            if "jaw_pose" not in batch_params:
                batch_params["jaw_pose"] = torch.zeros([be - bs, 1, 3], device=device)
            if "leye_pose" not in batch_params:
                batch_params["leye_pose"] = torch.zeros([be - bs, 1, 3], device=device)
            if "reye_pose" not in batch_params:
                batch_params["reye_pose"] = torch.zeros([be - bs, 1, 3], device=device)

            smpl_output = smplx_model(**batch_params)
            pred_verts = smpl_output.vertices  # [B, 10475, 3] in meters
            target_batch = target_vertices[bs:be]

            # Error in cm (multiply by 100)
            error = torch.sqrt(((pred_verts - target_batch) ** 2).sum(-1)).mean(1) * 100
            errors.extend(error.cpu().numpy().tolist())

    return np.array(errors)


# ============================================================
# Main pipeline
# ============================================================

def load_mhr_meshes(input_dir, device):
    """Load PLY files → [B, 18439, 3] tensor."""
    ply_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    # Exclude our own script
    ply_files = [f for f in ply_files if not f.endswith("mhr2smplx_convert.py")]

    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in {input_dir}")

    vertices_list = []
    for f in tqdm.tqdm(ply_files, desc="Loading PLY meshes"):
        mesh = trimesh.load(f, process=False)
        assert mesh.vertices.shape[0] == 18439, (
            f"Expected 18439 vertices (MHR LOD1), got {mesh.vertices.shape[0]} in {f}"
        )
        vertices_list.append(mesh.vertices)

    vertices_np = np.stack(vertices_list, axis=0)  # [B, 18439, 3]
    vertices = torch.from_numpy(vertices_np).float().to(device)
    logger.info(f"Loaded {len(ply_files)} frames, vertices shape: {vertices.shape}")
    return vertices, ply_files


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MHR mesh → SMPLX parameter conversion (no pymomentum)")
    parser.add_argument("input_dir", help="Directory containing MHR PLY mesh files")
    parser.add_argument("-o", "--output_dir", default=None,
                        help="Output directory (default: <input_dir>/smplx_output)")
    parser.add_argument("--smplx_model_path", default=SMPLX_MODEL_PATH,
                        help=f"Path to SMPLX model directory (default: {SMPLX_MODEL_PATH})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for optimization (default: {BATCH_SIZE})")
    parser.add_argument("--device", default=DEVICE,
                        help=f"Device for computation (default: {DEVICE})")
    parser.add_argument("--single_identity", action="store_true", default=True,
                        help="Use single identity (betas) for all frames (default: True)")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(input_dir, "smplx_output")
    smplx_model_path = args.smplx_model_path
    device = args.device
    batch_size = args.batch_size

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Using device: {device}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    # 1. Load MHR meshes (already in meters, y-up, no flip needed)
    mhr_vertices_m, ply_files = load_mhr_meshes(input_dir, device)

    # 2. Load mapping and MHR faces
    triangle_ids, baryc_coords = load_surface_mapping()
    mhr_faces = load_mhr_faces()

    # 3. Barycentric interpolation: MHR [18439, cm] → SMPLX [10475, m]
    # Input data is in meters. Conversion class expects cm and does cm→m internally.
    # So we convert m→cm, then the interpolation does cm→m.
    mhr_vertices_cm = mhr_vertices_m * 100.0  # m → cm
    target_vertices = barycentric_interpolation_mhr2smplx(
        mhr_vertices_cm, mhr_faces, triangle_ids, baryc_coords
    )
    logger.info(f"Target SMPLX vertices shape: {target_vertices.shape}")

    # 4. Initialize SMPLX model
    smplx_model = smplx.SMPLX(
        model_path=smplx_model_path,
        gender="neutral",
        num_expression_coeffs=10,
        use_pca=True,
        batch_size=1,
    )

    # 5. Optimize SMPLX parameters
    fitting = SMPLXFitting(smplx_model, device=device, batch_size=batch_size)
    params = fitting.fit(target_vertices, single_identity=args.single_identity)

    # 6. Evaluate fitting error
    errors = evaluate_fitting_error(smplx_model, params, target_vertices, device, batch_size)
    logger.info(f"Fitting errors: mean={errors.mean():.4f} cm, max={errors.max():.4f} cm")

    # 7. Save results
    params_dir = os.path.join(output_dir, "parameters")
    os.makedirs(params_dir, exist_ok=True)

    param_dict = {}
    for key, val in params.items():
        arr = val.detach().cpu().numpy()
        param_dict[key] = arr
        logger.info(f"  {key}: shape {arr.shape}")

    np.savez(os.path.join(params_dir, "smplx_params.npz"), **param_dict)
    for key, val in params.items():
        np.save(os.path.join(params_dir, f"{key}.npy"), val.detach().cpu().numpy())

    np.save(os.path.join(output_dir, "fitting_errors.npy"), errors)

    # 8. Save SMPLX meshes for visual verification
    meshes_dir = os.path.join(output_dir, "meshes")
    os.makedirs(meshes_dir, exist_ok=True)

    smplx_faces_np = smplx_model.faces if isinstance(smplx_model.faces, np.ndarray) else smplx_model.faces.cpu().numpy()
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(ply_files)), desc="Saving SMPLX meshes"):
            batch_params = {}
            for k, v in params.items():
                if v.shape[0] == 1 and k == "betas":
                    batch_params[k] = v.expand(1, -1).to(device)
                else:
                    batch_params[k] = v[i:i+1].to(device)
            if "jaw_pose" not in batch_params:
                batch_params["jaw_pose"] = torch.zeros([1, 1, 3], device=device)
            if "leye_pose" not in batch_params:
                batch_params["leye_pose"] = torch.zeros([1, 1, 3], device=device)
            if "reye_pose" not in batch_params:
                batch_params["reye_pose"] = torch.zeros([1, 1, 3], device=device)

            smpl_output = smplx_model(**batch_params)
            verts_np = smpl_output.vertices[0].cpu().numpy()
            mesh = trimesh.Trimesh(vertices=verts_np, faces=smplx_faces_np, process=False)
            mesh.export(os.path.join(meshes_dir, f"{i+1:05d}.ply"))

    logger.info(f"\nConversion complete! Output saved to: {output_dir}")
    logger.info(f"  parameters/: SMPLX parameter files")
    logger.info(f"  meshes/: SMPLX mesh files (.ply)")
    logger.info(f"  fitting_errors.npy: per-frame fitting errors")


if __name__ == "__main__":
    main()
