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
  1. Place spheres on 20 mesh vertices of rest_pose → save PLY
  2. Apply random pose via LBS → get posed mesh → save PLY
  3. Use vertex_transform_matrices to LBS-transform the sphere centers → save PLY

Dependencies: torch, numpy, trimesh (no pymomentum).
"""

import os
import sys
import torch
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mhr_torch import (
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


def merge_sphere_meshes(sphere_meshes):
    return trimesh.util.concatenate(sphere_meshes)


def select_vertex_indices(n_vertices, n_select, seed=42):
    """Randomly select n_select vertex indices from [0, n_vertices)."""
    np.random.seed(seed)
    return np.sort(np.random.choice(n_vertices, size=n_select, replace=False))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(lod=1, device=device)
    faces_np = model.faces.cpu().numpy()
    print(f"Model loaded. Vertices: {NUM_VERTICES_LOD1}, Faces: {faces_np.shape[0]}, Joints: {model.n_joints}")

    identity_configs = [
        ("id1_thin",      101, 0.3),
        ("id2_average",   102, 0.8),
        ("id3_large",     103, 1.5),
        ("id4_random1",   104, 1.0),
        ("id5_random2",   105, 0.5),
    ]

    for label, seed, id_scale in identity_configs:
        torch.manual_seed(seed)
        id_c = id_scale * torch.randn(1, NUM_IDENTITY_BLENDSHAPES, device=device)

        # 0. Rest pose mesh (zero pose, given identity)
        print(f"\n[{label}] Step 0: Computing rest pose mesh...")
        with torch.no_grad():
            fe_zero = torch.zeros(1, NUM_FACE_EXPRESSION_BLENDSHAPES, device=device)
            mp_zero = torch.zeros(1, 204, device=device)
            rest_verts, _ = model.forward(id_c, mp_zero, fe_zero, apply_correctives=False)

        rest_verts_np = rest_verts[0].cpu().numpy()
        mesh = trimesh.Trimesh(vertices=rest_verts_np, faces=faces_np, process=False)
        mesh.export(os.path.join(OUTPUT_DIR, f"{label}_rest_mesh.ply"))
        print(f"  Saved {label}_rest_mesh.ply")

        # 1. Place spheres on 20 mesh vertices
        #    Since spheres are on mesh vertices, we can directly use vertex_transform_matrices
        #    without any skin weight interpolation.
        print(f"  Step 1: Selecting {N_SPHERES} mesh vertices and placing spheres...")
        vert_indices = select_vertex_indices(NUM_VERTICES_LOD1, N_SPHERES, seed=seed)
        selected_verts = rest_verts_np[vert_indices]  # [20, 3]

        sphere_meshes = [create_sphere_mesh(v) for v in selected_verts]
        merged = merge_sphere_meshes(sphere_meshes)
        merged.export(os.path.join(OUTPUT_DIR, f"{label}_rest_spheres.ply"))
        print(f"  Saved {label}_rest_spheres.ply  (vertex indices: {vert_indices.tolist()})")

        # 2. Apply random pose → posed mesh
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

        # 3. LBS-transform the sphere centers using vertex_transform_matrices
        #    M_i = Σ_j w_ij * D_j  (per-vertex 4x4 transform)
        #    v_posed_i = M_i @ v_rest_i
        #    Since spheres are on mesh vertices, we index directly into vertex_matrices.
        print(f"  Step 3: LBS-transforming spheres via vertex_transform_matrices...")
        with torch.no_grad():
            vertex_matrices = model.vertex_transform_matrices(deform_matrices)  # [1, V, 4, 4]

        # Get per-vertex matrices for the selected vertices
        M_selected = vertex_matrices[0, vert_indices]  # [20, 4, 4]

        # Transform: v_posed = M @ v_rest (homogeneous)
        v_rest_homo = torch.cat([
            torch.from_numpy(selected_verts).float().to(device),
            torch.ones(N_SPHERES, 1, device=device),
        ], dim=-1)  # [20, 4]
        v_posed_homo = torch.matmul(M_selected, v_rest_homo.unsqueeze(-1))  # [20, 4, 1]
        transformed_centers = v_posed_homo[:, :3, 0].cpu().numpy()  # [20, 3]

        # Save posed spheres
        posed_sphere_meshes = [create_sphere_mesh(v) for v in transformed_centers]
        merged = merge_sphere_meshes(posed_sphere_meshes)
        merged.export(os.path.join(OUTPUT_DIR, f"{label}_posed_spheres.ply"))
        print(f"  Saved {label}_posed_spheres.ply")

        # Verify: compare transformed sphere centers with the ground-truth posed vertices
        #         Since spheres are on mesh vertices, the ground truth is posed_verts[vert_indices].
        gt_posed = posed_verts_np[vert_indices]  # [20, 3]
        errors = np.linalg.norm(transformed_centers - gt_posed, axis=1)
        print(f"  Verification (vs ground-truth posed vertices):")
        print(f"    mean={errors.mean():.6f} cm, max={errors.max():.6f} cm")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    for label, _, _ in identity_configs:
        print(f"  {label}:")
        print(f"    {label}_rest_mesh.ply     - rest pose mesh (identity shape, zero pose)")
        print(f"    {label}_rest_spheres.ply   - spheres on mesh vertices of rest pose")
        print(f"    {label}_posed_mesh.ply     - deformed mesh (identity + random pose)")
        print(f"    {label}_posed_spheres.ply  - LBS-transformed spheres (vertex_transform_matrices)")
    print()
    print("Verification: posed spheres should exactly match posed mesh vertices.")
    print("  Error should be ~0 (numerical precision) since spheres are on vertices,")
    print("  and vertex_transform_matrices + M@v is mathematically equivalent to forward().")
    print("=" * 60)


if __name__ == "__main__":
    main()
