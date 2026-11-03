#!/usr/bin/env python3
"""From-scratch sparse-view DTU Scan65 reconstruction stages."""
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "DTU" / "scan65"
VIEWS = (25, 28, 22, 40, 44)

def load_cameras(path: Path):
    archive = np.load(path)
    result = {}
    for i in range(49):
        w, s = f"world_mat_{i}", f"scale_mat_{i}"
        if w not in archive or s not in archive:
            continue
        P = (archive[w] @ archive[s])[:3, :4].astype(np.float64)
        # RQ decomposition gives intrinsics and world-to-camera rotation.
        import cv2
        K, R, t, *_ = cv2.decomposeProjectionMatrix(P)
        K = K / K[2, 2]
        translation = t[:3, 0] / t[3, 0] if t.shape[0] == 4 else t.reshape(3)
        center = (-R.T @ translation).reshape(3)
        result[i] = {"projection": P, "K": K, "R": R, "center": center}
    return result

def image_path(data: Path, view: int):
    p = data / "image" / f"{view:06d}.png"
    if not p.is_file():
        raise FileNotFoundError(f"Missing view {view}: {p}")
    return p

def prepare(data: Path, out: Path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    cams = load_cameras(data / "cameras.npz")
    out1 = out / "phase1"
    for sub in ("", "cameras", "metadata"):
        (out1 / sub).mkdir(parents=True, exist_ok=True)
    meta = {"scan": 65, "view_to_image": {}, "camera_archive": str((data / "cameras.npz").resolve()),
            "camera_convention": "world_mat_i @ scale_mat_i is projection; camera centers from projection decomposition; camera entries indexed by image stem"}
    images, centers, dirs = [], [], []
    for v in VIEWS:
        p = image_path(data, v)
        with Image.open(p) as im:
            rgb = im.convert("RGB")
            images.append((v, rgb.copy()))
            meta["view_to_image"][str(v)] = {"image": str(p.resolve()), "size_wh": list(rgb.size), "mode": "RGB", "format": im.format}
        if v not in cams:
            raise ValueError(f"Camera archive has no entry for view {v}")
        c = cams[v]
        centers.append(c["center"])
        direction = -c["R"][2, :]
        dirs.append(direction / np.linalg.norm(direction))
        np.savez_compressed(out1 / "cameras" / f"view_{v:02d}.npz", K=c["K"], R_world_to_camera=c["R"],
                            camera_center=c["center"], projection=c["projection"])
    centers, dirs = np.asarray(centers), np.asarray(dirs)
    np.save(out1 / "camera_positions.npy", centers)
    np.save(out1 / "camera_directions.npy", dirs)
    # Pairwise angles use normalized camera viewing directions.
    cos = np.clip(dirs @ dirs.T, -1, 1)
    angles = np.degrees(np.arccos(cos))
    meta["views"] = list(VIEWS)
    meta["pairwise_viewing_angle_degrees"] = angles.tolist()
    meta["evaluation_views"] = [v for v in range(49) if v not in VIEWS]
    (out1 / "metadata" / "dataset.json").write_text(json.dumps(meta, indent=2))
    fig, axes = plt.subplots(1, len(images), figsize=(18, 4))
    for ax, (v, im) in zip(axes, images):
        ax.imshow(im); ax.set_title(f"View {v}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out1 / "selected_views.png", dpi=150); plt.close(fig)
    fig = plt.figure(figsize=(7, 7)); ax = fig.add_subplot(111, projection="3d")
    center = centers.mean(axis=0)
    for v, c, d in zip(VIEWS, centers, dirs):
        ax.scatter(*c, s=40); ax.text(*c, str(v)); ax.quiver(*c, *d, length=np.linalg.norm(centers-center)*.18 or 1)
    ax.scatter(*center, marker="x", s=80); ax.set_title("Scan65 selected camera centers and viewing directions")
    fig.tight_layout(); fig.savefig(out1 / "view_distribution.png", dpi=150); plt.close(fig)
    print(f"M1 complete: {len(images)} selected images; outputs: {out1}")

def run_vggt(data: Path, out: Path, repo: Path | None, checkpoint: str | None, image_size: int):
    import torch
    from PIL import Image
    if repo:
        sys.path.insert(0, str(repo.resolve()))
    try:
        from vggt.models.vggt import VGGT
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as e:
        raise RuntimeError("VGGT package not installed. Create environment from environment.yml, then rerun.") from e
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint:
        model = VGGT()
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    else:
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to(device)
    # VGGT expects RGB tensors normalized to [0,1], dimensions divisible by 14.
    size = image_size - image_size % 14
    tensors, original_sizes = [], []
    for v in VIEWS:
        with Image.open(image_path(data, v)) as im:
            im = im.convert("RGB"); original_sizes.append(im.size)
            im = im.resize((size, size), Image.Resampling.BICUBIC)
            tensors.append(torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).float()/255.)
    x = torch.stack(tensors, 0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)
        pose = pred["pose_enc"]
        extr, intr = pose_encoding_to_extri_intri(pose, x.shape[-2:])
    root = out / "phase2_vggt"
    for d in ("depth", "confidence", "cameras", "point_maps", "visualizations"):
        (root/d).mkdir(parents=True, exist_ok=True)
    depths, confs = pred.get("depth"), pred.get("depth_conf")
    if depths is None or confs is None:
        raise RuntimeError("This VGGT version did not return depth and depth_conf")
    dnp, cnp = depths.detach().float().cpu().numpy(), confs.detach().float().cpu().numpy()
    enp, inp = extr.detach().float().cpu().numpy(), intr.detach().float().cpu().numpy()
    np.save(root/"cameras"/"intrinsics.npy", inp); np.save(root/"cameras"/"extrinsics_w2c.npy", enp)
    if "world_points" in pred:
        np.save(root/"point_maps"/"world_points.npy", pred["world_points"].detach().float().cpu().numpy())
    import matplotlib.pyplot as plt
    for i, v in enumerate(VIEWS):
        depth = np.squeeze(dnp[0,i]); conf = np.squeeze(cnp[0,i])
        if depth.shape != conf.shape: raise ValueError(f"Depth/confidence shape mismatch for {v}: {depth.shape}/{conf.shape}")
        np.save(root/"depth"/f"view_{v:02d}.npy", depth); np.save(root/"confidence"/f"view_{v:02d}.npy", conf)
        for arr, name in ((depth,"depth"),(conf,"confidence")):
            fig, ax = plt.subplots(); im=ax.imshow(arr, cmap="magma"); fig.colorbar(im, ax=ax); ax.set_title(f"View {v}: {name}"); fig.savefig(root/"visualizations"/f"view_{v:02d}_{name}.png", dpi=120); plt.close(fig)
    meta = {"views":VIEWS,"device":device,"input_size_wh":[size,size],"original_sizes_wh":original_sizes,
            "preprocessing":"RGB, bicubic square resize, float [0,1]; K/extrinsics at resized input resolution",
            "extrinsics_convention":"world-to-camera 3x4", "intrinsics_convention":"pixel coordinates at model input size"}
    (root/"metadata.json").write_text(json.dumps(meta,indent=2)); print(f"VGGT predictions: {root}")

def unproject(data: Path, out: Path, confidence_percentile: float, max_depth: float):
    import open3d as o3d
    from PIL import Image
    base=out/"phase2_vggt"; dest=out/"phase3_points"; dest.mkdir(parents=True,exist_ok=True)
    intr=np.load(base/"cameras"/"intrinsics.npy"); extr=np.load(base/"cameras"/"extrinsics_w2c.npy")
    clouds=[]
    for i,v in enumerate(VIEWS):
        depth=np.load(base/"depth"/f"view_{v:02d}.npy").squeeze()
        conf=np.load(base/"confidence"/f"view_{v:02d}.npy").squeeze()
        rgb=np.asarray(Image.open(image_path(data,v)).convert("RGB").resize((depth.shape[1],depth.shape[0]),Image.Resampling.BICUBIC))
        maskp=data/"mask"/f"{v:03d}.png"
        mask=np.asarray(Image.open(maskp).convert("L").resize((depth.shape[1],depth.shape[0]),Image.Resampling.NEAREST))>0 if maskp.exists() else np.ones(depth.shape,bool)
        # Confidence percentile filters the weakest predictions; reject invalid/background depth.
        threshold=np.percentile(conf[np.isfinite(conf)],confidence_percentile)
        valid=np.isfinite(depth)&(depth>0)&(depth<max_depth)&np.isfinite(conf)&(conf>=threshold)&mask
        yy,xx=np.nonzero(valid); K=intr[0,i]; cam=np.linalg.inv(K)@np.stack([xx,yy,np.ones_like(xx)])*depth[yy,xx]
        T=np.eye(4); T[:3,:4]=extr[0,i]; world=(np.linalg.inv(T)@np.vstack([cam,np.ones((1,cam.shape[1]))]))[:3].T
        p=o3d.geometry.PointCloud(); p.points=o3d.utility.Vector3dVector(world); p.colors=o3d.utility.Vector3dVector(rgb[yy,xx]/255.)
        o3d.io.write_point_cloud(str(dest/f"view_{v:02d}.ply"),p); clouds.append(p)
    fused=clouds[0]
    for p in clouds[1:]: fused+=p
    o3d.io.write_point_cloud(str(dest/"fused_points.ply"),fused)
    print(f"Saved per-view and fused clouds ({len(fused.points)} points) to {dest}")

def mesh(data: Path, out: Path, voxel: float, sdf_trunc: float, confidence_percentile: float):
    import open3d as o3d
    import trimesh
    pred=out/"phase2_vggt"; intr=np.load(pred/"cameras"/"intrinsics.npy"); extr=np.load(pred/"cameras"/"extrinsics_w2c.npy")
    volume=o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=voxel,sdf_trunc=sdf_trunc,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i,v in enumerate(VIEWS):
        depth=np.load(pred/"depth"/f"view_{v:02d}.npy").squeeze().astype(np.float32)
        conf=np.load(pred/"confidence"/f"view_{v:02d}.npy").squeeze()
        threshold=np.percentile(conf[np.isfinite(conf)],confidence_percentile)
        valid=np.isfinite(depth)&(depth>0)&(depth<10)&np.isfinite(conf)&(conf>=threshold)
        mp=data/"mask"/f"{v:03d}.png"
        if mp.exists():
            from PIL import Image
            mask=np.asarray(Image.open(mp).convert("L").resize((depth.shape[1],depth.shape[0]),Image.Resampling.NEAREST))>0
            valid &= mask
        depth[~valid]=0
        from PIL import Image
        rgb=np.asarray(Image.open(image_path(data,v)).convert("RGB").resize((depth.shape[1],depth.shape[0]),Image.Resampling.BICUBIC))
        color=o3d.geometry.Image(np.ascontiguousarray(rgb)); dimg=o3d.geometry.Image(np.ascontiguousarray(depth))
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(color,dimg,depth_scale=1.0,depth_trunc=10.0,convert_rgb_to_intensity=False)
        K=intr[0,i]; h,w=depth.shape
        camera=o3d.camera.PinholeCameraIntrinsic(w,h,float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
        T=np.eye(4); T[:3,:4]=extr[0,i]
        volume.integrate(rgbd,camera,T)
    initial=volume.extract_triangle_mesh(); initial.compute_vertex_normals()
    stage5=out/"phase5_mesh"; stage5.mkdir(parents=True,exist_ok=True)
    o3d.io.write_triangle_mesh(str(stage5/"initial_mesh.ply"),initial)
    # Open3D's scalable volume does not expose serialization; save fusion settings and source cameras as a reproducibility manifest.
    np.savez_compressed(stage5/"tsdf_volume.npz",voxel_length=np.float32(voxel),sdf_trunc=np.float32(sdf_trunc),
                        intrinsics=intr,extrinsics_w2c=extr,views=np.asarray(VIEWS),format=np.asarray("Open3D ScalableTSDFVolume settings and camera manifest; use saved phase2 depth maps to reintegrate"))
    p=o3d.io.read_point_cloud(str(out/"phase3_points"/"fused_points.ply"))
    p, _=p.remove_statistical_outlier(nb_neighbors=24,std_ratio=2.0)
    cdir=out/"phase4_clean_points"; cdir.mkdir(parents=True,exist_ok=True); o3d.io.write_point_cloud(str(cdir/"filtered_points.ply"),p)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=32)); p.orient_normals_consistent_tangent_plane(30)
    o3d.io.write_point_cloud(str(cdir/"normals.ply"),p)
    pts=np.asarray(p.points)
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8,7)); ax=fig.add_subplot(111,projection="3d"); stride=max(1,len(pts)//12000); ax.scatter(pts[::stride,0],pts[::stride,1],pts[::stride,2],s=.2); ax.set_title("Filtered Scan65 point cloud"); fig.tight_layout(); fig.savefig(cdir/"pointcloud_visualization.png",dpi=140); plt.close(fig)
    # If the depth-based volume is empty or invalid, reconstruct with Poisson and document the fallback.
    method="Open3D scalable TSDF + marching cubes"
    if len(initial.vertices)==0 or len(initial.triangles)==0:
        initial,densities=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(p,depth=9)
        dens=np.asarray(densities); initial.remove_vertices_by_mask(dens<np.quantile(dens,.03)); method="Open3D Poisson fallback (TSDF produced no surface)"
    mesh=initial; mesh.remove_degenerate_triangles(); mesh.remove_duplicated_triangles(); mesh.remove_duplicated_vertices(); mesh.remove_non_manifold_edges(); mesh.compute_vertex_normals()
    triangle_clusters, cluster_sizes, _ = mesh.cluster_connected_triangles()
    component_count=len(cluster_sizes); largest_component_faces=int(max(cluster_sizes)) if component_count else 0
    retained_largest_component=False
    if component_count>1 and len(mesh.triangles) and largest_component_faces/len(mesh.triangles)>=.90:
        mesh.remove_triangles_by_mask(np.asarray(triangle_clusters)!=int(np.argmax(cluster_sizes)))
        mesh.remove_unreferenced_vertices(); mesh.compute_vertex_normals(); retained_largest_component=True
    mdir=out/"phase6_clean_mesh"; mdir.mkdir(parents=True,exist_ok=True)
    o3d.io.write_triangle_mesh(str(mdir/"clean_mesh.ply"),mesh); o3d.io.write_triangle_mesh(str(mdir/"clean_mesh.obj"),mesh)
    t=trimesh.load(mdir/"clean_mesh.ply",force="mesh",process=True)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(9,8)); ax=fig.add_subplot(111,projection="3d"); stride=max(1,len(t.faces)//15000)
    ax.add_collection3d(Poly3DCollection(t.vertices[t.faces[::stride]],facecolor=(.72,.72,.76),edgecolor="none",linewidth=0,alpha=1))
    bounds=t.bounds; ax.set_xlim(*bounds[:,0]); ax.set_ylim(*bounds[:,1]); ax.set_zlim(*bounds[:,2]); ax.set_box_aspect(bounds[1]-bounds[0]); ax.set_title("Clean Scan65 TSDF mesh")
    fig.tight_layout(); fig.savefig(mdir/"mesh_preview.png",dpi=140); plt.close(fig)
    edges=np.sort(np.vstack((t.faces[:,[0,1]],t.faces[:,[1,2]],t.faces[:,[2,0]])),axis=1)
    _, edge_counts=np.unique(edges,axis=0,return_counts=True)
    stats={"vertices":len(t.vertices),"faces":len(t.faces),"connected_components":len(t.split(only_watertight=False)),"components_before_largest_component_filter":component_count,"retained_largest_component":retained_largest_component,"largest_component_face_fraction_before_filter":largest_component_faces/max(1,int(sum(cluster_sizes))),"boundary_edges":int(np.count_nonzero(edge_counts==1)),"non_manifold_edges":int(np.count_nonzero(edge_counts>2)),"degenerate_faces":int(np.count_nonzero(t.area_faces<1e-12)),"watertight":bool(t.is_watertight),"surface_area":float(t.area),"bounds":t.bounds.tolist(),"voxel_length":voxel,"tsdf_truncation":sdf_trunc,"method":method}
    (mdir/"mesh_statistics.json").write_text(json.dumps(stats,indent=2)); print(json.dumps(stats,indent=2))

def evaluate(data: Path, out: Path):
    """Write a transparent report; never substitute GT cameras into VGGT reconstruction."""
    truth_candidates=list(data.glob("*.ply"))+list(data.glob("*.obj"))+list((data/"ground_truth").glob("**/*")) if (data/"ground_truth").exists() else list(data.glob("*.ply"))+list(data.glob("*.obj"))
    report={"scan":65,"input_views":list(VIEWS),"held_out_views":[v for v in range(49) if v not in VIEWS],
            "geometry":{"status":"not_run_no_ground_truth_geometry" if not truth_candidates else "ground_truth_available_metrics_not_yet_configured","ground_truth_candidates":[str(p) for p in truth_candidates]},
            "camera":{"status":"predictions_and_DTUs_calibration_preserved_separately; similarity-aligned camera error not computed"},
            "novel_view_rendering":{"status":"not_run_no_appearance_representation","required":"a texture or trained appearance representation"}}
    dest=out/"phase7_evaluation"; dest.mkdir(parents=True,exist_ok=True)
    (dest/"evaluation_report.md").write_text("# Scan65 evaluation status\n\n"+"\n".join(f"- **{k}:** {v['status']}" if isinstance(v,dict) else f"- **{k}:** {v}" for k,v in report.items())+"\n")
    (dest/"evaluation_status.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("stage",choices=["prepare","infer","points","mesh","evaluate","all"]); ap.add_argument("--data",type=Path,default=DEFAULT_DATA); ap.add_argument("--out",type=Path,default=ROOT/"outputs"); ap.add_argument("--vggt-repo",type=Path); ap.add_argument("--checkpoint"); ap.add_argument("--image-size",type=int,default=518); ap.add_argument("--confidence-percentile",type=float,default=35); ap.add_argument("--max-depth",type=float,default=10.0); ap.add_argument("--voxel",type=float,default=.01); ap.add_argument("--sdf-trunc",type=float,default=.04)
    a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    if a.stage in ("prepare","all"): prepare(a.data,a.out)
    if a.stage in ("infer","all"): run_vggt(a.data,a.out,a.vggt_repo,a.checkpoint,a.image_size)
    if a.stage in ("points","all"): unproject(a.data,a.out,a.confidence_percentile,a.max_depth)
    if a.stage in ("mesh","all"): mesh(a.data,a.out,a.voxel,a.sdf_trunc,a.confidence_percentile)
    if a.stage in ("evaluate","all"): evaluate(a.data,a.out)
if __name__=="__main__": main()
