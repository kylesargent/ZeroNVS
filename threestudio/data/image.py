import bisect
import math
import os
from dataclasses import dataclass, field
import sys
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from PIL import Image
import skimage.measure
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import lpips
import pickle
import copy
import open3d as o3d

from resources import MIPNERF360_DATASET_PATH, VALIDATION_SCENES_PATH

import threestudio
from threestudio import register
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *



def get_dtu_ground_plane():
    a, b, c = 0.114901, 0.414158, 1
    d = -0.459126 + 0.2

    normal = np.array([a, b, c])
    intercept = d
    return normal, intercept


def filter_ground_plane(points):
    normal, intercept = get_dtu_ground_plane()

    prod = np.matmul(points, normal[..., None])[..., 0]
    return points[prod > intercept]


@dataclass
class SingleImageDataModuleConfig:
    # height and width should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    height: Any = 96
    width: Any = 96
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    default_elevation_deg: float = 0.0
    default_azimuth_deg: float = -180.0
    default_camera_distance: float = 1.2
    default_fovy_deg: float = 60.0
    image_path: str = ""
    use_random_camera: bool = True
    random_camera: dict = field(default_factory=dict)
    view_synthesis: Optional[dict] = field(default_factory=dict)
    rays_noise_scale: float = 2e-3
    batch_size: int = 1
    requires_depth: bool = False
    requires_normal: bool = False
    n_aux_c2w: int = 0


class SingleImageDataBase:
    def setup(self, cfg, split):
        self.split = split
        self.rank = get_rank()
        self.cfg: SingleImageDataModuleConfig = cfg

        if self.cfg.use_random_camera:
            random_camera_cfg = parse_structured(
                RandomCameraDataModuleConfig, self.cfg.get("random_camera", {})
            )
            if split == "train":
                self.random_pose_generator = RandomCameraIterableDataset(
                    random_camera_cfg
                )
            else:
                self.random_pose_generator = RandomCameraDataset(
                    random_camera_cfg, split
                )

        elevation_deg = torch.FloatTensor([self.cfg.default_elevation_deg])
        azimuth_deg = torch.FloatTensor([self.cfg.default_azimuth_deg])
        camera_distance = torch.FloatTensor([self.cfg.default_camera_distance])

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        def get_defaults(azimuth):
            camera_position: Float[Tensor, "1 3"] = torch.stack(
                [
                    camera_distance * torch.cos(elevation) * torch.cos(azimuth),
                    camera_distance * torch.cos(elevation) * torch.sin(azimuth),
                    camera_distance * torch.sin(elevation),
                ],
                dim=-1,
            )

            center: Float[Tensor, "1 3"] = torch.zeros_like(camera_position)
            up: Float[Tensor, "1 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
                None
            ]

            light_position: Float[Tensor, "1 3"] = camera_position
            lookat: Float[Tensor, "1 3"] = F.normalize(center - camera_position, dim=-1)
            right: Float[Tensor, "1 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
            up = F.normalize(torch.cross(right, lookat), dim=-1)
            c2w: Float[Tensor, "1 3 4"] = torch.cat(
                [
                    torch.stack([right, up, -lookat], dim=-1),
                    camera_position[:, :, None],
                ],
                dim=-1,
            )
            return c2w, light_position, camera_position

        c2w, light_position, camera_position = get_defaults(azimuth)

        self.aux_c2ws = []
        if self.cfg.n_aux_c2w > 0:
            for i in range(self.cfg.n_aux_c2w):
                aux_c2w, _, _ = get_defaults(
                    np.deg2rad(azimuth_deg + (360 * (i + 1) / (self.cfg.n_aux_c2w + 1)))
                )
                self.aux_c2ws.append(homogenize_poses(aux_c2w))

        self.c2w = c2w
        self.camera_position = camera_position
        self.light_position = light_position

        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distance = camera_distance
        self.fovy = torch.deg2rad(torch.FloatTensor([self.cfg.default_fovy_deg]))

        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        assert len(self.heights) == len(self.widths)
        self.resolution_milestones: List[int]
        if len(self.heights) == 1 and len(self.widths) == 1:
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.directions_unit_focals = [
            get_ray_directions(H=height, W=width, focal=1.0)
            for (height, width) in zip(self.heights, self.widths)
        ]
        self.focal_lengths = [
            0.5 * height / torch.tan(0.5 * self.fovy) for height in self.heights
        ]

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.directions_unit_focal = self.directions_unit_focals[0]
        self.focal_length = self.focal_lengths[0]
        self.set_rays()
        self.load_images()
        self.prev_height = self.height

    def set_rays(self):
        # get directions by dividing directions_unit_focal by focal length
        directions: Float[Tensor, "1 H W 3"] = self.directions_unit_focal[None]
        directions[:, :, :, :2] = directions[:, :, :, :2] / self.focal_length

        rays_o, rays_d = get_rays(
            directions, self.c2w, keepdim=True, noise_scale=self.cfg.rays_noise_scale
        )

        proj_mtx: Float[Tensor, "4 4"] = get_projection_matrix(
            self.fovy, self.width / self.height, 0.1, 100.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "4 4"] = get_mvp_matrix(self.c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx

    def load_images(self):
        # load image
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"
        rgba = cv2.cvtColor(
            cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = (
            cv2.resize(
                rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
            ).astype(np.float32)
            / 255.0
        )
        rgb = rgba[..., :3]
        self.rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        self.mask: Float[Tensor, "1 H W 1"] = (
            torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0).to(self.rank)
        )
        print(
            f"[INFO] single image dataset: load image {self.cfg.image_path} {self.rgb.shape}"
        )

        # load depth
        if self.cfg.requires_depth:
            depth_path = self.cfg.image_path.replace("_rgba.png", "_depth.npy")
            assert os.path.exists(depth_path)
            # depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.load(depth_path)[..., None]
            depth = cv2.resize(
                depth, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            self.depth: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(depth.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            print(
                f"[INFO] single image dataset: load depth {depth_path} {self.depth.shape}"
            )
        else:
            self.depth = None

        # load normal
        if self.cfg.requires_normal:
            normal_path = self.cfg.image_path.replace("_rgba.png", "_normal.png")
            assert os.path.exists(normal_path)
            normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
            normal = cv2.resize(
                normal, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            self.normal: Float[Tensor, "1 H W 3"] = (
                torch.from_numpy(normal.astype(np.float32) / 255.0)
                .unsqueeze(0)
                .to(self.rank)
            )
            print(
                f"[INFO] single image dataset: load normal {normal_path} {self.normal.shape}"
            )
        else:
            self.normal = None

    def get_all_images(self):
        return self.rgb

    def get_all_images_fullres(self):
        assert os.path.exists(
            self.cfg.image_path
        ), f"Could not find image {self.cfg.image_path}!"
        rgba = cv2.cvtColor(
            cv2.imread(self.cfg.image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
        )
        rgba = rgba.astype(np.float32) / 255.
        # rgba = (
        #     cv2.resize(
        #         rgba, (self.width, self.height), interpolation=cv2.INTER_AREA
        #     ).astype(np.float32)
        #     / 255.0
        # )
        rgb = rgba[..., :3]
        rgb: Float[Tensor, "1 H W 3"] = (
            torch.from_numpy(rgb).unsqueeze(0).contiguous().to(self.rank)
        )
        return rgb

    def update_step_(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        if self.height == self.prev_height:
            return

        self.prev_height = self.height
        self.width = self.widths[size_ind]
        self.directions_unit_focal = self.directions_unit_focals[size_ind]
        self.focal_length = self.focal_lengths[size_ind]
        threestudio.debug(f"Training height: {self.height}, width: {self.width}")
        self.set_rays()
        self.load_images()


def homogenize_poses(p):
    assert p.shape[-2:] == (3, 4)

    return torch.concatenate(
        [
            p,
            torch.tensor([0, 0, 0, 1], dtype=p.dtype).broadcast_to(
                p[..., :1, :4].shape
            ),
        ],
        dim=-2,
    )


class SingleImageIterableDataset(IterableDataset, SingleImageDataBase, Updateable):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def collate(self, batch) -> Dict[str, Any]:
        batch = {
            "rays_o": self.rays_o,
            "rays_d": self.rays_d,
            "mvp_mtx": self.mvp_mtx,
            "camera_positions": self.camera_position,
            "light_positions": self.light_position,
            "elevation": self.elevation_deg,
            "azimuth": self.azimuth_deg,
            "camera_distances": self.camera_distance,
            "rgb": self.rgb,
            "ref_depth": self.depth,
            "ref_normal": self.normal,
            "mask": self.mask,
            "height": self.cfg.height,
            "width": self.cfg.width,
            "canonical_c2w": homogenize_poses(self.c2w),
        }
        if self.cfg.use_random_camera:
            batch["random_camera"] = self.random_pose_generator.collate(None)
            batch["random_camera"]["canonical_c2w"] = homogenize_poses(self.c2w)
            batch["random_camera"]["aux_c2ws"] = self.aux_c2ws

        return batch

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        self.update_step_(epoch, global_step, on_load_weights)
        self.random_pose_generator.update_step(epoch, global_step, on_load_weights)

    def __iter__(self):
        while True:
            yield {}


class SingleImageDataset(Dataset, SingleImageDataBase):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.setup(cfg, split)

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        return self.random_pose_generator[index]


matmul = np.matmul
mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]


def fov_to_intrinsics(fov_deg, resolution):
    cx = cy = resolution // 2
    fx = cx / np.tan(np.deg2rad(fov_deg / 2))
    fy = fx

    camtopix = np.array(
        [
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [
                0,
                0,
                1,
                0,
            ],
            [0, 0, 0, 1],
        ]
    )
    pixtocam = np.linalg.inv(camtopix)
    return pixtocam


def dumb_rasterizer(points, intrinsics, extrinsics):
    homogeneous_visible_points = homogenize_points(points)
    pixel_idxs, z = rasterize_homogeneous_point_cloud(
        homogeneous_visible_points,
        intrinsics,
        extrinsics,
    )

    mask = ((pixel_idxs >= 0) & (pixel_idxs < 256)).all(axis=1)
    pixel_idxs = pixel_idxs[mask]
    z = z[mask]

    # import pdb
    # pdb.set_trace()

    depth_image = np.ones(shape=(256, 256)) * np.inf

    for (j, i), z in zip(pixel_idxs, z):
        depth_image[i][j] = min(depth_image[i][j], z)
    depth_image = np.where(np.isinf(depth_image), 0, depth_image)
    return depth_image


def homogenize_points(points):
    assert points.ndim == 2
    assert points.shape[1] == 3

    return np.concatenate([points, np.ones_like(points[..., :1])], axis=1)


def rasterize_homogeneous_point_cloud(
    homogeneous_worldspace_locations,
    intrinsics,
    extrinsics,
):
    # n = h * w

    pixtocam = intrinsics[:3, :3]
    camtopix = np.linalg.inv(pixtocam)

    ## point cloud -> pixels
    novel_worldtocam = np.linalg.inv(extrinsics)
    novel_cameraspace_locations = mat_vec_mul(
        novel_worldtocam, homogeneous_worldspace_locations
    )[..., :3]
    novel_cameraspace_locations = matmul(
        novel_cameraspace_locations, np.diag(np.array([1.0, -1.0, -1.0]))
    )

    z = novel_cameraspace_locations[..., 2]
    pixels = mat_vec_mul(
        camtopix, novel_cameraspace_locations / np.maximum(z[..., None], 1e-3)
    )[..., :2]

    pixel_idxs = pixels.astype(np.int32)
    return pixel_idxs, z


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # source: https://github.com/google-research/multinerf/blob/31d857bc668785b374c009200d94f6b811051259/internal/camera_utils.py#L191C1-L227C37
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform


def look_at(
    eye: np.ndarray,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
    up: np.ndarray = np.array([0, 0, 1]),
    at: np.ndarray = np.array([0, 0, 0]),
    eps: float = 1e-5,
) -> np.ndarray:
    """Returns the rotation matrix for rotating eye to at.

    Args:
      eye: (3,) float array describing origin location.
      up: (3,) float array describing the coordinate system that represents up,
        e.g. [0, 0, 1] means z-up.
      at: (3,) float array describing location to rotate to.
      eps: epsilon value to avoid division by zero.
    Returns:
      r_mat: (3, 3, 1) matrix of rotation to be applied in order to transform eye
        to at.
    """
    at = np.reshape(at.astype(np.float64), (1, 3))
    up = np.reshape(up.astype(np.float64), (1, 3))
    eye = np.reshape(eye, (1, 3))
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    # Forward vector.
    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    # Right vector.
    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    # Up vector.
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate(
        (x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)),
        axis=2,
    )
    r_mat = r_mat[0]

    pose = np.eye(4)
    pose[:3, :3] = r_mat
    pose[:3, -1] = eye
    return pose


def get_angle(rotation):
    cos_theta = (np.trace(rotation) - 1) / 2
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return theta


def get_angle_between_rotations(pose1, pose2):
    r1 = pose1[:3, :3]
    r2 = pose2[:3, :3]
    return get_angle(np.dot(r1.T, r2))


def get_scene(cfg):
    if cfg.view_synthesis.dataset == "DTU":
        from ldm.data import pixelnerf

        scene_uid = cfg.view_synthesis.scene_uid
        assert scene_uid in pixelnerf.DTU_TEST_WHITELIST
        scene = pixelnerf.get_dtu_scene(
            base_dir=pixelnerf.DTU_BASE_DIR, scene_uid=scene_uid
        )

        scene["original_extrinsics"] = copy.copy(scene["extrinsics"])
        scene["extrinsics"][:, :3, -1] /= cfg.view_synthesis.scale

        _coord_trans_world = np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        )

        intrinsics = fov_to_intrinsics(scene["fovy_deg"], 256)
        scene["intrinsics"] = intrinsics

        scene["test_input_depth_map"] = np.zeros_like(scene['images'][0, ..., :1])
        scene["elevation_deg"] = 20.0  # hardcoded
        return scene
    elif cfg.view_synthesis.dataset in ["mipnerf360", "acid", "re10k", "co3d"]:
        from ldm.data import webdataset_co3d

        # scene_uids: ['bicycle', 'bonsai', 'counter', 'garden', 'kitchen', 'room', 'stump']
        if cfg.view_synthesis.dataset == "mipnerf360":
            with open(
                MIPNERF360_DATASET_PATH,
                "rb",
            ) as fp:
                mipnerf360_datasets = pickle.load(fp)
            scene = mipnerf360_datasets[cfg.view_synthesis.scene_uid]
        else:
            with open(
                os.path.join(
                    VALIDATION_SCENES_PATH,
                    cfg.view_synthesis.dataset,
                    f"{cfg.view_synthesis.scene_uid}.pkl",
                ),
                "rb",
            ) as fp:
                scene = pickle.load(fp)

        transformed_poses, _ = transform_poses_pca(scene["extrinsics"])
        scene["extrinsics"] = pad_poses(transformed_poses)
        world_data = webdataset_co3d.get_world_data(
            worldtocams=np.linalg.inv(scene["extrinsics"])[:, :3]
        )
        print(world_data["focus_pt"].shape)
        scene["extrinsics"][:, :3, -1] -= world_data["focus_pt"]

        lookat_extrinsics = np.array(
            [look_at(eye=loc) for loc in scene["extrinsics"][:, :3, -1]]
        )

        # import pdb
        # pdb.set_trace()

        scene["lookat_extrinsics"] = lookat_extrinsics

        angles = np.array(
            [
                get_angle_between_rotations(p1, p2)
                for (p1, p2) in zip(scene["extrinsics"], scene["lookat_extrinsics"])
            ]
        )
        scene["angles"] = angles

        # scale for compat
        scene["images"] = scene["images"] * 2 - 1

        angle_idxs = np.argsort(angles)
        scene["test_input_idx"] = angle_idxs[cfg.view_synthesis.input_view_idx]

        # import pdb
        # pdb.set_trace()

        input_pose = scene["extrinsics"][scene["test_input_idx"]]
        x, y, z = input_pose[:3, -1]
        xy = (x**2 + y**2) ** 0.5
        elevation_deg = np.rad2deg(np.arctan2(z, xy))

        print("elevation_deg: ", elevation_deg.item())
        scene["elevation_deg"] = elevation_deg.item()

        scale = world_data["scene_radius_focus_pt"]
        # scene['focus_pt'] = world_data['focus_pt']
        scene["extrinsics"][:, :3, -1] /= scale
        scene["lookat_extrinsics"][:, :3, -1] /= scale

        scene["test_input_depth_map"] = (
            scene["depths"][scene["test_input_idx"], ..., 0] / scale
        )

        # should all be the same
        # compute fov deg
        pixtocam = scene["intrinsics"][0]
        camtopix = np.linalg.inv(pixtocam)
        fx = camtopix[0, 0]
        cx = camtopix[0, 2]
        assert cx == 128
        fov_deg = np.rad2deg(2 * np.arctan2(cx, fx))
        scene["fovy_deg"] = fov_deg

        scene["uncropped_images"] = scene["images"]
        return scene
    else:
        raise NotImplementedError


class ViewSynthesisCameraDataset(Dataset):
    def __init__(
        self, cfg: Any, scene: Any, threestudio_input_pose, gt_to_pred_scale
    ) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg

        # if split == "val":
        #     self.n_views = self.cfg.n_val_views
        # else:
        #     self.n_views = self.cfg.n_test_views

        # azimuth_deg: Float[Tensor, "B"]
        # if self.split == "val":
        #     # make sure the first and last view are not the same
        #     azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
        # else:
        #     azimuth_deg = torch.linspace(0, 360.0, self.n_views)
        # elevation_deg: Float[Tensor, "B"] = torch.full_like(
        #     azimuth_deg, self.cfg.eval_elevation_deg
        # )
        # camera_distances: Float[Tensor, "B"] = torch.full_like(
        #     elevation_deg, self.cfg.eval_camera_distance
        # )

        # elevation = elevation_deg * math.pi / 180
        # azimuth = azimuth_deg * math.pi / 180

        # # convert spherical coordinates to cartesian coordinates
        # # right hand coordinate system, x back, y right, z up
        # # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        # camera_positions: Float[Tensor, "B 3"] = torch.stack(
        #     [
        #         camera_distances * torch.cos(elevation) * torch.cos(azimuth),
        #         camera_distances * torch.cos(elevation) * torch.sin(azimuth),
        #         camera_distances * torch.sin(elevation),
        #     ],
        #     dim=-1,
        # )
        self.test_input_idx = scene["test_input_idx"]
        DTU_input_pose = scene["extrinsics"][scene["test_input_idx"]]
        # novel_extrinsics = np.delete(
        #     scene["extrinsics"], scene["test_input_idx"], axis=0
        # )
        novel_extrinsics = scene["extrinsics"]

        eye = np.eye(4)
        eye[:3] = threestudio_input_pose.numpy()[0, :3]
        threestudio_input_pose = eye

        threestudio_input_pose = threestudio_input_pose
        assert DTU_input_pose.shape == threestudio_input_pose.shape == (4, 4), (
            DTU_input_pose.shape,
            threestudio_input_pose.shape,
        )
        print(threestudio_input_pose.shape, DTU_input_pose.shape)
        transform_1 = np.linalg.inv(DTU_input_pose)
        c2w = np.matmul(transform_1, novel_extrinsics)

        if gt_to_pred_scale is not None:
            print("Scaling gt poses to match pred scale!")
            c2w[:, :3, -1] /= gt_to_pred_scale

        transform_2 = threestudio_input_pose
        c2w = np.matmul(transform_2, c2w)

        c2w = torch.from_numpy(c2w.astype(np.float32))

        self.n_views = len(c2w)

        fovy_deg = torch.ones((self.n_views,)) * scene["fovy_deg"].astype(np.float32)

        self.fov_deg = fovy_deg

        camera_positions = c2w[:, :3, -1]

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)

        # default camera up direction as +z
        # up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        #     None, :
        # ].repeat(self.cfg.eval_batch_size, 1)

        # fovy_deg: Float[Tensor, "B"] = torch.full_like(
        #     elevation_deg, self.cfg.eval_fovy_deg
        # )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        # lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        # right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        # up = F.normalize(torch.cross(right, lookat), dim=-1)
        # c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        #     [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        #     dim=-1,
        # )
        # c2w: Float[Tensor, "B 4 4"] = torch.cat(
        #     [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        # )
        # c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        # self.elevation, self.azimuth = elevation, azimuth
        # self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        # self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "test_input_idx": self.test_input_idx,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "fov_deg": self.fov_deg[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            # "elevation": None,
            # "azimuth": None,
            # "camera_distances": None,
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch


class ViewSynthesisImageDataset(Dataset, SingleImageDataBase):
    def __init__(
        self, cfg: Any, scene, threestudio_input_pose, split: str, gt_to_pred_scale
    ) -> None:
        super().__init__()
        self.setup(cfg, split)

        self.random_pose_generator = ViewSynthesisCameraDataset(
            cfg.random_camera, scene, threestudio_input_pose, gt_to_pred_scale
        )

        self.images = scene["uncropped_images"]

    def __len__(self):
        return len(self.random_pose_generator)

    def __getitem__(self, index):
        item = self.random_pose_generator[index]
        item["gt_rgb"] = self.images[item["index"]]
        return item

def get_guidance_image_path(image_path):
    base, ext = os.path.splitext(image_path)
    return base + '_guidance' + ext


@register("single-image-datamodule")
class SingleImageDataModule(pl.LightningDataModule):
    cfg: SingleImageDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        if cfg.view_synthesis is not None:
            scene = get_scene(cfg)
            self.scene = scene

            print("View synthesis is enabled for eval. Modifying config.")
            print("Right now we just override the fov deg.")
            print(
                "Whenever you override something here, need to check in the yaml config"
            )
            print("And manually resolve any of its deps.")

            assert cfg.default_azimuth_deg == 0.0
            # set elevation
            # TODO(kylesargent): Can this be tuned more intelligently?
            if cfg.view_synthesis.dataset != "DTU":
                elevation_deg = scene["elevation_deg"]
                cfg.default_elevation_deg = elevation_deg
                min_elevation_deg = round(
                    np.clip(elevation_deg - 30, min(5, elevation_deg), 80), 2
                ).item()
                max_elevation_deg = round(np.clip(elevation_deg + 50, 5, 80), 2).item()
                print(min_elevation_deg, max_elevation_deg)
                cfg.random_camera.elevation_range = [
                    min_elevation_deg,
                    max_elevation_deg,
                ]

                cfg.random_camera.eval_elevation_deg
                cfg.random_camera.eval_elevation_deg = elevation_deg

            # set fov_deg
            fovy_deg = scene["fovy_deg"].item()
            cfg.default_fovy_deg = fovy_deg
            cfg.random_camera.fovy_range = [fovy_deg, fovy_deg]
            cfg.random_camera.eval_fovy_deg = fovy_deg

            # write img to tmp and override cfg path
            input_image = scene["uncropped_images"][scene["test_input_idx"]] / 2 + 0.5
            input_image = Image.fromarray(
                np.clip(input_image * 255, 0, 255).astype(np.uint8)
            )
            image_path = cfg.image_path
            input_image.convert("RGBA").save(image_path)
            # make sure this was set earlier
            assert cfg.image_path == image_path, (cfg.image_path, image_path)

            # do the same thing for guidance image
            input_image = scene["images"][scene["test_input_idx"]] / 2 + 0.5
            assert input_image.shape == (256, 256, 3)
            input_image = Image.fromarray(
                np.clip(input_image * 255, 0, 255).astype(np.uint8)
            )
            guidance_image_path = get_guidance_image_path(image_path)
            input_image.convert("RGBA").save(guidance_image_path)

            self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)

            canonical_batch_cfg = copy.copy(self.cfg)
            canonical_batch_cfg.random_camera.eval_height
            canonical_batch_cfg.random_camera.eval_height = 256
            canonical_batch_cfg.random_camera.eval_width
            canonical_batch_cfg.random_camera.eval_width = 256

            _unused_train_dataset = SingleImageIterableDataset(self.cfg, "train")
            self.canonical_c2w = _unused_train_dataset.c2w

            self.unscaled_test_dataset = ViewSynthesisImageDataset(
                self.cfg,
                self.scene,
                self.canonical_c2w,
                "test",
                gt_to_pred_scale=None,
            )

            self.dtu_to_threestudio_transform = np.matmul(
                self.canonical_c2w,
                np.linalg.inv(self.scene["extrinsics"][self.scene["test_input_idx"]]),
            )
            self.canonical_batch = self.unscaled_test_dataset[
                self.scene["test_input_idx"]
            ]
        else:
            self.cfg = parse_structured(SingleImageDataModuleConfig, cfg)
            self.scene = None

        self.gt_to_pred_scale = None
        super().__init__()

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = SingleImageIterableDataset(self.cfg, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = SingleImageDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            if self.cfg.view_synthesis is not None:
                if self.cfg.view_synthesis.rescale:
                    if self.cfg.view_synthesis.manual_gt_to_pred_scale is not None:
                        self.gt_to_pred_scale = (
                            self.cfg.view_synthesis.manual_gt_to_pred_scale
                        )
                    else:
                        # should be set by on_fit_end
                        assert self.gt_to_pred_scale is not None

                self.test_dataset = ViewSynthesisImageDataset(
                    self.cfg,
                    self.scene,
                    self.canonical_c2w,
                    "test",
                    gt_to_pred_scale=self.gt_to_pred_scale,
                )
            else:
                self.test_dataset = SingleImageDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
