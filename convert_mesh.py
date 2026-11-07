#!/usr/bin/env python3
"""Mesh -> FaithC -> cancel even crossings -> fit anchors -> mesh.

Examples:
    python convert_mesh.py model.glb
    python convert_mesh.py model.obj -r 512 -o output/model.ply

Requires CUDA PyTorch, Atom3d, trimesh, NumPy and Open3D >= 0.18.
FBX input additionally requires Blender (auto-detected or --blender).
Input geometry must already contain any desired tessellation/displacement.
The even-crossing heuristic was validated on closed, consistently oriented meshes.
"""

import argparse
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def resolution_arg(value):
    value = positive_int(value)
    if value < 2 or value & (value - 1):
        raise argparse.ArgumentTypeError("resolution must be a power of two, at least 2")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="input mesh, or NPZ with v and f arrays")
    parser.add_argument("-r", "--resolution", type=resolution_arg, default=2048)
    parser.add_argument("-o", "--output", type=Path, help="default: <input_stem>_faithc_<resolution>.glb beside input")
    parser.add_argument("--steps", type=positive_int, default=44, help="Adam fitting steps (default: 44)")
    parser.add_argument("--batch-size", type=positive_int, default=65536, help="fitting/query chunk size (default: 65536)")
    parser.add_argument("--samples", type=positive_int, default=850000, help="reference samples per refresh (default: 850000)")
    parser.add_argument("--seed", type=int, default=1197)
    parser.add_argument("--blender", type=Path, help="Blender executable for FBX input")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    args.output = (args.output or args.input.with_name(
        f"{args.input.stem}_faithc_{args.resolution}.glb"
    )).expanduser().resolve()
    if args.output == args.input:
        parser.error("output must differ from input")
    if args.output.suffix.lower() not in {".glb", ".obj", ".ply", ".stl", ".off", ".npz"}:
        parser.error("output extension must be .glb, .obj, .ply, .stl, .off or .npz")
    return args


def load_fbx(path, blender):
    executable = str(blender) if blender else shutil.which("blender")
    if executable is None and os.name == "nt":
        installed = sorted(Path(os.environ.get("ProgramFiles", "C:/Program Files")).glob(
            "Blender Foundation/Blender */blender.exe"
        ))
        if installed:
            executable = str(installed[-1])
    if not executable:
        raise ValueError("FBX input requires Blender; supply --blender /path/to/blender")
    script = """
import bpy, numpy as np, sys
source, target = sys.argv[sys.argv.index('--') + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=source, use_image_search=False)
vertices, faces, offset = [], [], 0
for obj in bpy.context.scene.objects:
    if obj.type != 'MESH':
        continue
    mesh = obj.data
    mesh.calc_loop_triangles()
    v = np.asarray([obj.matrix_world @ item.co for item in mesh.vertices])
    f = np.asarray([item.vertices[:] for item in mesh.loop_triangles])
    vertices.append(v)
    faces.append(f + offset)
    offset += len(v)
if not vertices:
    raise ValueError('FBX contains no meshes')
np.savez(target, v=np.concatenate(vertices), f=np.concatenate(faces))
"""
    with tempfile.TemporaryDirectory(prefix="faithc-fbx-") as directory:
        script_path = Path(directory) / "import_mesh.py"
        mesh_path = Path(directory) / "mesh.npz"
        script_path.write_text(script, encoding="utf-8")
        result = subprocess.run(
            [executable, "-b", "--python-exit-code", "1", "--python", str(script_path), "--", str(path), str(mesh_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if result.returncode or not mesh_path.exists():
            raise RuntimeError(f"Blender FBX import failed:\n{result.stdout}")
        with np.load(mesh_path, allow_pickle=False) as data:
            return data["v"], data["f"]


def load_mesh(path, blender=None):
    if path.suffix.lower() == ".fbx":
        vertices, faces = load_fbx(path, blender)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            vertices, faces = data["v"], data["f"]
    else:
        mesh = trimesh.load(str(path), force="mesh", process=False)
        vertices, faces = mesh.vertices, mesh.faces
    vertices, faces = np.asarray(vertices), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ValueError("input must contain at least three 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("input must contain triangle faces")
    if not np.isfinite(vertices).all() or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("vertices must be finite and face indices must be integers")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("face index is outside the vertex array")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces):
        raise ValueError("input has no nondegenerate triangles")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def ray_scene(vertices, faces):
    scene = o3d.t.geometry.RaycastingScene(nthreads=8)
    scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)))
    return scene


def nearest(scene, points):
    return scene.compute_closest_points(o3d.core.Tensor(np.ascontiguousarray(points, dtype=np.float32)), nthreads=8)


def edge_keys(ijk, axis, resolution):
    width = resolution + 1
    return ((ijk[:, 0] * width + ijk[:, 1]) * width + ijk[:, 2]) * 3 + axis


def cancel_even_crossings(tokens, grid, reference, resolution, batch_size):
    edges, mapping = grid.voxel_unique_edges(tokens.active_voxel_indices)
    flux = torch.zeros(len(edges), dtype=torch.int8, device="cuda")
    flux[mapping.reshape(-1)] = tokens.edge_flux_sign.reshape(-1)
    active = torch.nonzero(flux, as_tuple=False).flatten()
    keys = []
    for chunk in active.split(batch_size):
        endpoints = grid.edge_endpoints(edges[chunk])
        ijk = torch.round((endpoints[:, 0] + 1) * (resolution / 2)).long().cpu().numpy()
        axis = (endpoints[:, 1] - endpoints[:, 0]).argmax(dim=1).cpu().numpy()
        keys.append(edge_keys(ijk, axis, resolution))
    if not keys:
        return 0
    key = np.concatenate(keys)
    order = np.argsort(key)
    sorted_keys = key[order]
    counts = np.zeros(len(active), dtype=np.int32)
    width = resolution + 1
    ijk = np.column_stack((key // 3 // width // width, key // 3 // width % width, key // 3 % width))
    for axis in range(3):
        other = [dimension for dimension in range(3) if dimension != axis]
        # Only cast lines that contain a currently marked grid edge.
        lines = np.unique(ijk[key % 3 == axis][:, other], axis=0)
        for begin in range(0, len(lines), batch_size):
            transverse = lines[begin:begin + batch_size]
            rays = np.zeros((len(transverse), 6), dtype=np.float32)
            rays[:, other] = transverse * (2.0 / resolution) - 1
            rays[:, axis] = -2
            rays[:, axis + 3] = 1
            hits = reference.list_intersections(o3d.core.Tensor(rays), nthreads=8)
            ray = hits["ray_ids"].numpy()
            distance = hits["t_hit"].numpy().astype(np.float64)
            along = np.floor((distance - 1) * (resolution / 2)).astype(np.int64)
            valid = (along >= 0) & (along < resolution)
            ray, along = ray[valid], along[valid]
            coords = np.zeros((len(ray), 3), dtype=np.int64)
            coords[:, other] = transverse[ray]
            coords[:, axis] = along
            hit_keys, hit_counts = np.unique(edge_keys(coords, axis, resolution), return_counts=True)
            positions = np.searchsorted(sorted_keys, hit_keys)
            in_range = positions < len(sorted_keys)
            positions, hit_keys, hit_counts = positions[in_range], hit_keys[in_range], hit_counts[in_range]
            matched = sorted_keys[positions] == hit_keys
            counts[order[positions[matched]]] = hit_counts[matched]
        print(f"  Intersection counts complete for axis {axis + 1}/3", flush=True)
    cancelled = (counts > 0) & (counts % 2 == 0)
    flux[active[torch.from_numpy(cancelled).to(active.device)]] = 0
    tokens.edge_flux_sign = flux[mapping]
    return int(cancelled.sum())


def encode_decode(vertices, faces, resolution, batch_size):
    bounds = torch.tensor([[-1.] * 3, [1.] * 3], device="cuda")
    bvh = MeshBVH(torch.from_numpy(vertices).cuda(), torch.from_numpy(faces).cuda())
    grid = OctreeIndexer(max_level=resolution.bit_length() - 1, bounds=bounds, device="cuda")
    tokens = FCTEncoder(bvh, grid, device="cuda").encode(
        min_level=min(4, resolution.bit_length() - 1),
        solver_weights=dict(lambda_n=1., lambda_d=1e-3, weight_power=1),
        compute_flux=True, clamp_anchors=True,
    )
    del bvh
    print(f"  Encoded {len(tokens.anchor):,} active voxels", flush=True)
    cancelled = cancel_even_crossings(tokens, grid, ray_scene(vertices, faces), resolution, batch_size)
    print(f"  Cancelled {cancelled:,} even-crossing edges", flush=True)
    decoded = FCTDecoder(resolution=resolution, bounds=bounds, device="cuda").decode_from_result(tokens)
    if not len(decoded.faces):
        raise ValueError("reconstruction has no triangles; increase resolution or check input geometry")
    # Preserve the exact voxel identity, including anchors that share a position.
    lower, upper = grid.cube_aabb_level(decoded.voxel_indices)
    centers = ((lower + upper) / 2).cpu().numpy()
    return decoded.vertices.cpu().numpy(), decoded.faces.cpu().numpy(), centers


def fitting_targets(x, faces, reference, ref_vertices, ref_faces, normals, probabilities, bary, samples, rng, batch_size):
    current = x.detach().cpu().numpy()
    targets = np.empty((len(faces), 4, 3), dtype=np.float32)
    target_normals = np.empty((len(faces), 3), dtype=np.float32)
    for begin in range(0, len(faces), batch_size):
        end = min(begin + batch_size, len(faces))
        points = np.einsum("ij,kjl->kil", bary, current[faces[begin:end]]).reshape(-1, 3)
        hits = nearest(reference, points)
        targets[begin:end] = hits["points"].numpy().reshape(-1, 4, 3)
        ids = hits["primitive_ids"].numpy()[::4]
        uv = hits["primitive_uvs"].numpy()[::4]
        weights = np.column_stack([1 - uv.sum(1), uv])
        interpolated = (normals[ref_faces[ids]] * weights[:, :, None]).sum(1)
        target_normals[begin:end] = interpolated / np.maximum(np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-12)
    ids = rng.choice(len(ref_faces), size=samples, p=probabilities)
    uv = rng.random((samples, 2)).astype(np.float32)
    uv = np.where(uv.sum(1, keepdims=True) > 1, 1 - uv, uv)
    triangles = ref_vertices[ref_faces[ids]]
    source = triangles[:, 0] + uv[:, 0, None] * (triangles[:, 1] - triangles[:, 0]) + uv[:, 1, None] * (triangles[:, 2] - triangles[:, 0])
    reconstruction = ray_scene(current, faces)
    reverse_ids = np.empty(samples, dtype=np.int64)
    reverse_weights = np.empty((samples, 3), dtype=np.float32)
    for begin in range(0, samples, batch_size):
        end = min(begin + batch_size, samples)
        hits = nearest(reconstruction, source[begin:end])
        uv = hits["primitive_uvs"].numpy()
        reverse_ids[begin:end] = hits["primitive_ids"].numpy()
        reverse_weights[begin:end] = np.column_stack([1 - uv.sum(1), uv])
    return targets, target_normals, source, reverse_ids, reverse_weights


def fit_anchors(vertices, faces, centers, ref_vertices, ref_faces, resolution, steps, samples, batch_size, seed):
    # Work in voxel units; the normalized encoder cube has side length two.
    ref_vertices = ref_vertices * (resolution / 2)
    reference = ray_scene(ref_vertices, ref_faces)
    triangles = ref_vertices[ref_faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.linalg.norm(cross, axis=1).astype(np.float64)
    probabilities = areas / areas.sum()
    normals = np.zeros_like(ref_vertices)
    for corner in range(3):
        np.add.at(normals, ref_faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-10)
    del triangles, cross, areas
    device = torch.device("cuda")

    def gpu(array):
        return torch.as_tensor(array, device=device)

    center = gpu(centers * (resolution / 2))
    initial = gpu(vertices * (resolution / 2))
    x = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([x], lr=.038, eps=1e-9, foreach=False)
    face_ids = gpu(faces.astype(np.int64))
    bary = np.array([[1/3, 1/3, 1/3], [.76, .12, .12], [.12, .76, .12], [.12, .12, .76]], dtype=np.float32)
    bary_gpu = gpu(bary)
    rng = np.random.default_rng(seed)

    def penalty(delta):
        return torch.sqrt(delta.square().sum(-1) + .015**2) - .015

    for step in range(steps):
        if step % 4 == 0:
            targets, target_normals, source, reverse_ids, reverse_weights = fitting_targets(
                x, faces, reference, ref_vertices, ref_faces, normals, probabilities,
                bary, samples, rng, batch_size,
            )
        if step == 24:
            optimizer.param_groups[0]["lr"] = .018
        if step == 36:
            optimizer.param_groups[0]["lr"] = .009
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        # Accumulate the original full-mesh loss in chunks before one Adam update.
        for begin in range(0, len(faces), batch_size):
            end = min(begin + batch_size, len(faces))
            ids = face_ids[begin:end]
            tri = x[ids]
            geom = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
            area = geom.detach().norm(dim=-1).clamp(.05, 4)
            centered = center[ids]
            axis = torch.nn.functional.normalize(torch.cross(centered[:, 1] - centered[:, 0], centered[:, 2] - centered[:, 0], dim=-1), dim=-1)
            pos = torch.einsum("ij,kjl->kil", bary_gpu, tri)
            forward = (penalty(pos - gpu(targets[begin:end])).mean(1) * area).sum()
            align = (torch.nn.functional.normalize(geom, dim=-1, eps=1e-7) * gpu(target_normals[begin:end])).sum(1)
            along = (geom * axis).sum(1)
            loss = (forward + .045 * ((1 - align) * area).sum() + .7 * (.07 - along).relu().sum() + .55 * (.3 - align).relu().sum()) / len(faces)
            loss.backward()
            total += loss.item()
        for begin in range(0, samples, batch_size):
            end = min(begin + batch_size, samples)
            tri = x[face_ids[gpu(reverse_ids[begin:end])]]
            point = (tri * gpu(reverse_weights[begin:end])[:, :, None]).sum(1)
            loss = penalty(point - gpu(source[begin:end])).sum() / samples
            loss.backward()
            total += loss.item()
        tether = .003 * (x - initial).square().sum(1).mean()
        tether.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(center - .49999, center + .49999)
        if step % 4 == 3 or step == steps - 1:
            print(f"  Fit {step + 1}/{steps}, loss {total + tether.item():.6f}", flush=True)
    return x.detach().cpu().numpy() * (2.0 / resolution)


def save_mesh(path, vertices, faces):
    if not np.isfinite(vertices).all():
        raise ValueError("fitting produced non-finite coordinates")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep an existing output intact if serialization fails midway.
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix.lower(), dir=path.parent)
    os.close(descriptor)
    try:
        if path.suffix.lower() == ".npz":
            np.savez(temporary, v=vertices, f=faces)
        else:
            trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(temporary, file_type=path.suffix[1:].lower())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("FaithC conversion requires CUDA-enabled PyTorch and a CUDA GPU")
    started = time.perf_counter()
    vertices, faces = load_mesh(args.input, args.blender)
    center = (vertices.min(0) + vertices.max(0)) / 2
    width = float(np.ptp(vertices, axis=0).max()) / .95
    if not np.isfinite(width) or width <= 0:
        raise ValueError("input mesh has invalid or zero extent")
    normalized = np.asarray((vertices - center) * (2 / width), dtype=np.float32)
    print(f"Input {len(vertices):,} vertices, {len(faces):,} triangles; resolution {args.resolution}", flush=True)
    print("[1/3] FaithC encoding, crossing cancellation and decoding", flush=True)
    with torch.no_grad():
        reconstructed, reconstructed_faces, centers = encode_decode(normalized, faces, args.resolution, args.batch_size)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[2/3] Fitting {len(reconstructed):,} anchors to the input surface", flush=True)
    fitted = fit_anchors(reconstructed, reconstructed_faces, centers, normalized, faces,
                         args.resolution, args.steps, args.samples, args.batch_size, args.seed)
    print("[3/3] Exporting geometry in the input coordinate system", flush=True)
    save_mesh(args.output, fitted.astype(np.float64) * (width / 2) + center, reconstructed_faces)
    print(f"Saved {args.output}\nTriangles {len(reconstructed_faces):,}; elapsed {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    args = parse_args()
    try:
        import numpy as np
        import open3d as o3d
        import torch
        import trimesh
        sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
        from atom3d import MeshBVH
        from atom3d.grid import OctreeIndexer
        from faithcontour import FCTEncoder, FCTDecoder
        main(args)
    except (ImportError, RuntimeError, ValueError, OSError, KeyError) as error:
        print(f"Conversion failed: {error}", file=sys.stderr)
        sys.exit(1)
