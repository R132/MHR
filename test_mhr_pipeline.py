"""Pure PyTorch MHR pipeline test: verify LBS deformation correctness.

Key insight:
  LBS only handles skeletal deformation (rest pose → posed), NOT blendshape deformation.
  Therefore, small spheres must be placed on the **rest pose mesh with the same identity
  blendshapes** as the target deformed mesh — NOT on the canonical mesh (zero identity).
  Only when identity_coeffs = 0 do the canonical mesh and rest pose coincide.
  Otherwise, LBS-transformed spheres from the canonical mesh will NOT land on the
  deformed mesh surface, because the rest pose shape has been altered by identity blendshapes.

Test structure (per identity, 5 iterations):
  0. Given identity_coeffs → compute rest_pose mesh → save PLY
  1. Sample 20 surface points on rest_pose, build small spheres → save PLY
  2. Apply random pose via LBS → get posed mesh → save PLY
  3. Apply LBS to sphere centers → get posed spheres → save PLY

Verification: posed spheres should lie on the posed mesh surface.

Dependencies: torch, numpy, trimesh (no pymomentum).
"""

import os
import sys
import torch
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from get_vertice_transmation import (
    MHRTorchModel,
    load_model,
    NUM_IDENTITY_BLENDSHAPES,
    NUM_FACE_EXPRESSION_BLENDSHAPES,
    NUM_VERTICES_LOD1,
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_mhr_output")
SPHERE_RADIUS = 0.5  # cm
N_SPHERES = 20
N_IDENTITIES = 5


def create_sphere_mesh(center, radius=SPHERE_RADIUS, subdivisions=2):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    sphere.vertices += center
    return sphere


def random_surface_points(mesh_faces, mesh_vertices, n_points, seed=42):
    """Area-weighted random surface sampling with barycentric coordinates.

    Returns:
        points: [n_points, 3]
        face_indices: [n_points]
        bary_coords: [n_points, 3] barycentric weights per point
    """
    np.random.seed(seed)

    v0 = mesh_vertices[mesh_faces[:, 0]]
    v1 = mesh_vertices[mesh_faces[:, 1]]
    v2 = mesh_vertices[mesh_faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    probs = areas / areas.sum()

    face_indices = np.random.choice(len(mesh_faces), size=n_points, p=probs)

    r1 = np.random.rand(n_points)
    r2 = np.random.rand(n_points)
    sqrt_r1 = np.sqrt(r1)

    bary0 = 1 - sqrt_r1
    bary1 = sqrt_r1 * (1 - r2)
    bary2 = r2 * sqrt_r1

    bary_coords = np.stack([bary0, bary1, bary2], axis=1)  # [n_points, 3]

    points = (
        bary0[:, None] * v0[face_indices]
        + bary1[:, None] * v1[face_indices]
        + bary2[:, None] * v2[face_indices]
    )

    return points, face_indices, bary_coords


def apply_lbs_to_points(points, deform_matrices, skin_weights):
    """Apply LBS: v_posed = Σ_j w_j * D_j * v_rest.

    Args:
        points: [1, N, 3] rest pose points
        deform_matrices: [1, J, 4, 4] D_j = G_j * IBP_j
        skin_weights: [N, J] per-point skinning weights
    Returns:
        transformed_points: [1, N, 3]
    """
    N = points.shape[1]

    # Homogeneous coordinates
    v_homo = torch.cat([
        points,
        torch.ones(1, N, 1, device=points.device, dtype=points.dtype),
    ], dim=-1)

    w_exp = skin_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(1, -1, -1, -1, -1)
    D_exp = deform_matrices.unsqueeze(1).expand(-1, N, -1, -1, -1)
    v_exp = v_homo.unsqueeze(2).unsqueeze(-1)

    transformed = torch.matmul(D_exp, v_exp)
    weighted = w_exp * transformed
    result = weighted.sum(dim=2)[:, :, :3, 0]

    return result


def get_point_skin_weights(model, face_indices, bary_coords):
    """Interpolate vertex skin weights at surface sample points.

    Args:
        model: MHRTorchModel
        face_indices: [N] face index for each point
        bary_coords: [N, 3] barycentric weights per point
    Returns:
        point_weights: [N, J] normalized skin weights
    """
    faces = model.faces.cpu().numpy()
    weights = model.dense_skin_weights.cpu().numpy()

    v0_idx = faces[face_indices, 0]
    v1_idx = faces[face_indices, 1]
    v2_idx = faces[face_indices, 2]

    w0 = weights[v0_idx]
    w1 = weights[v1_idx]
    w2 = weights[v2_idx]

    point_weights = (
        bary_coords[:, 0:1] * w0
        + bary_coords[:, 1:2] * w1
        + bary_coords[:, 2:3] * w2
    )

    row_sums = point_weights.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-8, 1.0, row_sums)
    point_weights = point_weights / row_sums

    return torch.from_numpy(point_weights).to(model.device)


def merge_sphere_meshes(sphere_meshes):
    return trimesh.util.concatenate(sphere_meshes)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(lod=1, device=device)
    faces_np = model.faces.cpu().numpy()
    print(f"Model loaded. Vertices: {NUM_VERTICES_LOD1}, Faces: {faces_np.shape[0]}, Joints: {model.n_joints}")

    # 5 random identities with varying body shape
    identity_configs = [
        ("id1_thin",      101, 0.3),
        ("id2_average",   102, 0.8),
        ("id3_large",     103, 1.5),
        ("id4_random1",   104, 1.0),
        ("id5_random2",   105, 0.5),
    ]

    for label, seed, id_scale in identity_configs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        id_c = id_scale * torch.randn(1, NUM_IDENTITY_BLENDSHAPES, device=device)

        # 0. Rest pose mesh (zero pose, given identity)
        #    NOTE: We use zero face_expr and zero pose, so this is the pure
        #    rest shape for this identity — the "canonical" shape of this body.
        print(f"\n[{label}] Step 0: Computing rest pose mesh...")
        with torch.no_grad():
            fe_zero = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES, device=device)
            mp_zero = torch.zeros(1, 204, device=device)
            rest_verts, _ = model.forward(id_c, mp_zero, fe_zero, apply_correctives=False)

        rest_verts_np = rest_verts[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=rest_verts_np, faces=faces_np, process=False)
        mesh.export(os.path.join(OUTPUT_DIR, f"{label}_rest_mesh.ply"))
        print(f"  Saved {label}_rest_mesh.ply")

        # 1. Place spheres on rest pose surface
        #    CRITICAL: Spheres must be on the rest pose mesh (with this identity),
        #    NOT on the zero-identity canonical mesh. LBS transforms from rest pose
        #    to posed space; if the rest pose shape is wrong, LBS output will be wrong.
        print(f"  Step 1: Sampling {N_SPHERES} surface points on rest pose...")
        points, point_face_indices, bary_coords = random_surface_points(
            faces_np, rest_verts_np, N_SPHERES, seed=seed,
        )

        point_skin_weights = get_point_skin_weights(model, point_face_indices, bary_coords)

        sphere_meshes = []
        for j in range(N_SPHERES):
            sphere_meshes.append(create_sphere_mesh(points[j]))

        merged = merge_sphere_meshes(sphere_meshes)
        merged.export(os.path.join(OUTPUT_DIR, f"{label}_rest_spheres.ply"))
        print(f"  Saved {label}_rest_spheres.ply")

        # 2. Apply a random pose to get deformed mesh
        print(f"  Step 2: Applying random pose...")
        torch.manual_seed(seed + 1000)
        mp = 0.2 * (torch.rand(1, 204, device=device) - 0.5)
        fe_c = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES, device=device)

        with torch.no_grad():
            posed_verts, deform_matrices = model.forward(id_c, mp, fe_c)

        posed_verts_np = posed_verts[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=posed_verts_np, faces=faces_np, process=False)
        mesh.export(os.path.join(OUTPUT_DIR, f"{label}_posed_mesh.ply"))
        print(f"  Saved {label}_posed_mesh.ply")

        # 3. LBS-transform the spheres
        #    v_posed = Σ_j w_j * D_j * v_rest
        #    Since spheres are on the correct rest pose, they should land on the posed mesh.
        print(f"  Step 3: LBS-transforming spheres...")
        points_tensor = torch.from_numpy(points).float().to(device).unsqueeze(0)
        transformed_centers = apply_lbs_to_points(points_tensor, deform_matrices, point_skin_weights)
        transformed_centers_np = transformed_centers[0].cpu().numpy()

        transformed_sphere_meshes = []
        for j in range(N_SPHERES):
            transformed_sphere_meshes.append(create_sphere_mesh(transformed_centers_np[j]))

        merged = merge_sphere_meshes(transformed_sphere_meshes)
        merged.export(os.path.join(OUTPUT_DIR, f"{label}_posed_spheres.ply"))
        print(f"  Saved {label}_posed_spheres.ply")

        # Verify: nearest-vertex distance between sphere centers and posed mesh
        diffs = transformed_centers_np[:, None, :] - posed_verts_np[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = dists.min(axis=1)
        print(f"  Verification: mean={min_dists.mean():.4f} cm, max={min_dists.max():.4f} cm")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    for label, _, _ in identity_configs:
        print(f"  {label}:")
        print(f"    {label}_rest_mesh.ply     - rest pose mesh (identity shape, zero pose)")
        print(f"    {label}_rest_spheres.ply   - spheres on rest pose surface")
        print(f"    {label}_posed_mesh.ply     - deformed mesh (identity + random pose)")
        print(f"    {label}_posed_spheres.ply  - LBS-transformed spheres")
    print()
    print("Verification: each posed_spheres should sit on the corresponding posed_mesh.")
    print("=" * 60)


if __name__ == "__main__":
    main()
