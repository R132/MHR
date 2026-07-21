#!/usr/bin/env python3
"""Extract MHR model assets into standalone npz files.

Requires pixi environment with pymomentum. Run: pixi run python tools/extract_assets.py

The extracted assets allow get_vertice_transmation.py to run with only PyTorch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import pymomentum.geometry as pym_geometry
from pymomentum.torch.character import Character as TorchCharacter
from mhr.mhr import set_blendshape_parameter_sets

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
OUTPUT_DIR = os.path.join(ASSETS_DIR, "mhr_standalone")


def extract_assets(lod=1):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load character
    fbx_path = os.path.join(ASSETS_DIR, f"lod{lod}.fbx")
    model_path = os.path.join(ASSETS_DIR, "compact_v6_1.model")
    character = pym_geometry.Character.load_fbx(
        fbx_path, model_path, load_blendshapes=True
    )

    # ---- Parameter transform (pose-only, [889, 204]) ----
    # Blendshape columns are all zeros, so pose-only [889, 204] is sufficient
    pt_pose = character.parameter_transform  # before with_blend_shape
    from pymomentum.torch.character import Character as TorchCharacter
    ct_pose = TorchCharacter(character)
    pt_matrix = ct_pose.parameter_transform.parameter_transform.numpy()
    print(f"parameter_transform: {pt_matrix.shape}")

    # ---- BlendShape data ----
    character_with_bs = character.with_blend_shape(character.blend_shape)
    set_blendshape_parameter_sets(character_with_bs)
    bs = character_with_bs.blend_shape
    base_shape = np.array(bs.base_shape, dtype=np.float32)
    shape_vectors = np.array(bs.shape_vectors, dtype=np.float32)
    # Split into identity [45] and face_expr [72]
    identity_sv = shape_vectors[:45]
    face_expr_sv = shape_vectors[45:]
    print(f"base_shape: {base_shape.shape}, identity_sv: {identity_sv.shape}, face_expr_sv: {face_expr_sv.shape}")

    # ---- Skeleton data ----
    skel = character_with_bs.skeleton
    joint_parents = np.array(skel.joint_parents, dtype=np.int32)
    joint_offsets = np.array(skel.offsets, dtype=np.float32)
    joint_prerotations = np.array(skel.pre_rotations, dtype=np.float32)
    joint_names = list(skel.joint_names)
    print(f"joint_parents: {joint_parents.shape}, offsets: {joint_offsets.shape}, prerots: {joint_prerotations.shape}")

    # ---- Skin weights (dense [V, 127]) ----
    sw = character_with_bs.skin_weights
    dense_sw = np.array(sw.to_dense(character_with_bs.skeleton.size), dtype=np.float32)
    print(f"dense_skin_weights: {dense_sw.shape}")

    # ---- Also save sparse skin weights from TorchScript model ----
    ts_model = torch.jit.load(os.path.join(ASSETS_DIR, "mhr_model.pt"))
    sw_index = ts_model.get_lbsw()[0].numpy()
    sw_weight = ts_model.get_lbsw()[1].numpy()
    print(f"sparse_sw_index: {sw_index.shape}, sparse_sw_weight: {sw_weight.shape}")

    # ---- Inverse bind pose ----
    ibp = ts_model.character_torch.linear_blend_skinning.inverse_bind_pose.numpy()
    print(f"inverse_bind_pose: {ibp.shape}")

    # ---- Mesh faces ----
    faces = np.array(character_with_bs.mesh.faces, dtype=np.int32)
    print(f"faces: {faces.shape}")

    # ---- Save main assets ----
    save_dict = {
        "parameter_transform": pt_matrix,
        "base_shape": base_shape,
        "identity_shape_vectors": identity_sv,
        "face_expr_shape_vectors": face_expr_sv,
        "joint_parents": joint_parents,
        "joint_offsets": joint_offsets,
        "joint_prerotations": joint_prerotations,
        "dense_skin_weights": dense_sw,
        "sparse_sw_index": sw_index,
        "sparse_sw_weight": sw_weight,
        "inverse_bind_pose": ibp,
        "faces": faces,
    }
    npz_path = os.path.join(OUTPUT_DIR, f"mhr_assets_lod{lod}.npz")
    np.savez(npz_path, **save_dict)
    print(f"Saved main assets to {npz_path}")

    # ---- Save joint names as text file ----
    names_path = os.path.join(OUTPUT_DIR, f"joint_names_lod{lod}.txt")
    with open(names_path, "w") as f:
        for name in joint_names:
            f.write(name + "\n")
    print(f"Saved joint names to {names_path}")

    # ---- Copy pose correctives data (already exists as separate npz) ----
    blendshapes_src = os.path.join(ASSETS_DIR, f"corrective_blendshapes_lod{lod}.npz")
    activation_src = os.path.join(ASSETS_DIR, "corrective_activation.npz")
    blendshapes_dst = os.path.join(OUTPUT_DIR, f"corrective_blendshapes_lod{lod}.npz")
    activation_dst = os.path.join(OUTPUT_DIR, "corrective_activation.npz")
    import shutil
    shutil.copy2(blendshapes_src, blendshapes_dst)
    shutil.copy2(activation_src, activation_dst)
    print(f"Copied pose correctives data to {OUTPUT_DIR}")

    # ---- Verify: compare TorchScript model output with extracted data ----
    print("\n--- Verification ---")
    identity_coeffs = torch.randn(2, 45)
    model_parameters = 0.2 * (torch.rand(2, 204) - 0.5)
    face_expr_coeffs = torch.randn(2, 72)

    with torch.no_grad():
        ts_verts, ts_skel = ts_model(identity_coeffs, model_parameters, face_expr_coeffs)

    print(f"TorchScript verts: {ts_verts.shape}, skel_state: {ts_skel.shape}")

    # Also verify with pymomentum-based model
    from mhr.mhr import MHR
    mhr_model = MHR.from_files(device=torch.device("cpu"), lod=lod)
    with torch.no_grad():
        pm_verts, pm_skel = mhr_model(identity_coeffs, model_parameters, face_expr_coeffs)

    print(f"pymomentum verts diff: {torch.abs(ts_verts - pm_verts).mean().item():.6f}")
    print(f"pymomentum skel diff: {torch.abs(ts_skel - pm_skel).mean().item():.6f}")

    print("\nDone! All assets extracted to", OUTPUT_DIR)


if __name__ == "__main__":
    extract_assets()
