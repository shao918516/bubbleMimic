#!/usr/bin/env python3
"""Generate a MuJoCo terrain test course for tiangong3 (tg3)."""

from __future__ import annotations

import argparse
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterable

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESOURCES_DIR = os.path.join(PROJECT_ROOT, "resources")

ROBOT = "tg3"
ROBOT_XML = "tiangong3/urdf/tiangong3.xml"
SCENE_XML = "tiangong3/urdf/scene_terrain.xml"

TERRAIN_FRICTION = "1.5 0.005 0.0001"
HFIELD_DIR = os.path.join(RESOURCES_DIR, "terrain", "hfields")


def list_to_str(values: Iterable[float]) -> str:
    return " ".join(f"{v:.6g}" for v in values)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cx = math.cos(roll / 2)
    sx = math.sin(roll / 2)
    cy = math.cos(pitch / 2)
    sy = math.sin(pitch / 2)
    cz = math.cos(yaw / 2)
    sz = math.sin(yaw / 2)
    return np.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ],
        dtype=np.float64,
    )


def rot2d(x: float, y: float, yaw: float) -> tuple[float, float]:
    return (
        x * math.cos(yaw) - y * math.sin(yaw),
        x * math.sin(yaw) + y * math.cos(yaw),
    )


def euler_to_rot_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rot_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ],
        dtype=np.float64,
    )
    rot_y = np.array(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rot_z = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rot_z @ rot_y @ rot_x


def rotate_vec(roll: float, pitch: float, yaw: float, vec: np.ndarray) -> np.ndarray:
    return euler_to_rot_mat(roll, pitch, yaw) @ vec


def box_min_corner_z(half_size: np.ndarray, euler: tuple[float, float, float]) -> float:
    hx, hy, hz = half_size
    min_z = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = rotate_vec(*euler, np.array([sx * hx, sy * hy, sz * hz]))
                min_z = min(min_z, corner[2])
    return min_z


def save_gray_png(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "terrain_generator requires Pillow. Install with: pip install pillow"
        ) from exc
    Image.fromarray(image.astype(np.uint8)).save(path)


def _blur_2d(image: np.ndarray, sigma: float, passes: int) -> np.ndarray:
    kernel_radius = max(1, int(sigma * 2))
    kernel = np.exp(
        -0.5 * (np.arange(-kernel_radius, kernel_radius + 1) / sigma) ** 2
    )
    kernel /= kernel.sum()
    blurred = image
    for _ in range(passes):
        blurred = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, blurred
        )
        blurred = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 0, blurred
        )
    return blurred


def _enhance_height_contrast(
    heights: np.ndarray,
    variation_strength: float,
    contrast_gamma: float,
) -> np.ndarray:
    low, high = np.percentile(heights, [6.0, 94.0])
    if high > low:
        heights = (heights - low) / (high - low)
    heights = np.clip(heights, 0.0, 1.0)

    mean = heights.mean()
    heights = mean + (heights - mean) * variation_strength
    heights = np.clip(heights, 0.0, 1.0)
    heights = heights ** contrast_gamma
    heights -= heights.min()
    if heights.max() > 0:
        heights /= heights.max()
    return heights


def _upsample_bilinear(coarse: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    from PIL import Image

    coarse_u8 = (np.clip(coarse, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(coarse_u8).resize((ncol, nrow), Image.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0


def _generate_detail_layer(
    rng: np.random.Generator,
    cells: tuple[int, int],
    nrow: int,
    ncol: int,
    blur_sigma: float,
    amplitude_scale: float = 2.0,
) -> np.ndarray:
    """零均值细节层；amplitude_scale 放大局部起伏振幅。"""
    layer = rng.random(cells, dtype=np.float64)
    layer = _upsample_bilinear(layer, nrow, ncol)
    if blur_sigma > 0.0:
        layer = _blur_2d(layer, blur_sigma, passes=1)
    layer -= layer.mean()
    span = layer.max() - layer.min()
    if span > 0:
        layer = (layer / span) * amplitude_scale
    return layer


def generate_smooth_heightmap(
    nrow: int,
    ncol: int,
    seed: int,
    smooth_sigma: float = 1.2,
    edge_blend_fraction: float = 0.30,
    variation_strength: float = 1.6,
    contrast_gamma: float = 0.55,
    coarse_cells: tuple[int, int] = (12, 10),
    medium_detail_weight: float = 0.52,
    fine_detail_weight: float = 0.34,
    micro_detail_weight: float = 0.20,
    medium_detail_cells: tuple[int, int] | None = None,
    fine_detail_cells: tuple[int, int] | None = None,
    micro_detail_cells: tuple[int, int] | None = None,
) -> np.ndarray:
    """用粗网格控制起伏间距，对比度拉伸后再叠加中/细/微三层小起伏。"""
    rng = np.random.default_rng(seed)

    coarse = rng.random(coarse_cells, dtype=np.float64)
    heights = _upsample_bilinear(coarse, nrow, ncol)
    heights = _blur_2d(heights, smooth_sigma, passes=1)

    heights -= heights.min()
    if heights.max() > 0:
        heights /= heights.max()

    heights = _enhance_height_contrast(heights, variation_strength, contrast_gamma)
    base_peak = float(heights.max())

    if medium_detail_weight > 0.0:
        med_cells = medium_detail_cells or (
            max(8, coarse_cells[0] * 3),
            max(6, coarse_cells[1] * 3),
        )
        medium = _generate_detail_layer(
            rng, med_cells, nrow, ncol, max(0.28, smooth_sigma * 0.28), 2.0
        )
        heights += medium_detail_weight * medium

    if fine_detail_weight > 0.0:
        fine_cells = fine_detail_cells or (
            max(12, coarse_cells[0] * 7),
            max(10, coarse_cells[1] * 7),
        )
        fine = _generate_detail_layer(
            rng, fine_cells, nrow, ncol, max(0.15, smooth_sigma * 0.15), 2.2
        )
        heights += fine_detail_weight * fine

    if micro_detail_weight > 0.0:
        micro_cells = micro_detail_cells or (
            max(16, coarse_cells[0] * 10),
            max(14, coarse_cells[1] * 10),
        )
        micro = _generate_detail_layer(rng, micro_cells, nrow, ncol, 0.0, 2.5)
        heights += micro_detail_weight * micro

    global_min = float(heights.min())
    heights -= global_min
    coarse_peak_level = base_peak - global_min
    if coarse_peak_level > 0:
        heights /= coarse_peak_level / 0.99
    heights = np.clip(heights, 0.0, 1.0)

    blend_pixels = max(4, int(min(nrow, ncol) * edge_blend_fraction))
    row_idx = np.arange(nrow, dtype=np.float64)[:, None]
    col_idx = np.arange(ncol, dtype=np.float64)[None, :]
    dist_edge = np.minimum(
        np.minimum(row_idx, nrow - 1 - row_idx),
        np.minimum(col_idx, ncol - 1 - col_idx),
    )
    edge_mask = np.clip(dist_edge / blend_pixels, 0.0, 1.0)
    edge_mask = edge_mask * edge_mask * (3.0 - 2.0 * edge_mask)
    return heights * edge_mask


def write_heightfield_png(
    filename: str,
    nrow: int,
    ncol: int,
    seed: int,
    smooth_sigma: float,
    edge_blend_fraction: float = 0.30,
    variation_strength: float = 1.6,
    contrast_gamma: float = 0.55,
    coarse_cells: tuple[int, int] = (12, 10),
    medium_detail_weight: float = 0.52,
    fine_detail_weight: float = 0.34,
    micro_detail_weight: float = 0.20,
    medium_detail_cells: tuple[int, int] | None = None,
    fine_detail_cells: tuple[int, int] | None = None,
    micro_detail_cells: tuple[int, int] | None = None,
) -> str:
    heights = generate_smooth_heightmap(
        nrow,
        ncol,
        seed,
        smooth_sigma,
        edge_blend_fraction,
        variation_strength,
        contrast_gamma,
        coarse_cells,
        medium_detail_weight,
        fine_detail_weight,
        micro_detail_weight,
        medium_detail_cells,
        fine_detail_cells,
        micro_detail_cells,
    )
    image = (heights * 255).astype(np.uint8)
    output_path = os.path.join(HFIELD_DIR, filename)
    save_gray_png(output_path, image)
    return output_path


@dataclass
class TerrainLayout:
    """沿 +X 方向排列：复杂地形在前，结构化地形在后。"""
    undulating_easy_origin: tuple[float, float] = (5.0, 0.0)
    undulating_origin: tuple[float, float] = (14.5, 0.0)
    rough_road_origin: tuple[float, float] = (23.5, 0.0)
    plum_origin: tuple[float, float] = (31.0, 0.0)
    stairs_low_origin: tuple[float, float] = (35.0, -4.0)
    stairs_mid_origin: tuple[float, float] = (35.0, 0.0)
    stairs_high_origin: tuple[float, float] = (35.0, 4.0)
    ramp_row_x: float = 45.0
    ramp_length: float = 3.0


class TerrainSceneBuilder:
    def __init__(self, seed: int = 42) -> None:
        self.robot_name = ROBOT
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.layout = TerrainLayout()
        self.hfield_assets: list[dict] = []
        self.geoms: list[ET.Element] = []

    def _add_geom(
        self,
        name: str,
        geom_type: str,
        pos: Iterable[float],
        size: Iterable[float],
        quat: Iterable[float] | None = None,
        euler: Iterable[float] | None = None,
        rgba: str = "0.45 0.52 0.42 1",
    ) -> None:
        geom = ET.Element("geom")
        geom.attrib["name"] = name
        geom.attrib["type"] = geom_type
        geom.attrib["pos"] = list_to_str(pos)
        geom.attrib["size"] = list_to_str(size)
        geom.attrib["rgba"] = rgba
        geom.attrib["friction"] = TERRAIN_FRICTION
        geom.attrib["condim"] = "3"
        geom.attrib["contype"] = "1"
        geom.attrib["conaffinity"] = "1"
        if quat is not None:
            geom.attrib["quat"] = list_to_str(quat)
        elif euler is not None:
            geom.attrib["quat"] = list_to_str(euler_to_quat(*euler))
        self.geoms.append(geom)

    def _add_box(
        self,
        name: str,
        pos: Iterable[float],
        size: Iterable[float],
        euler: Iterable[float] = (0.0, 0.0, 0.0),
        rgba: str = "0.45 0.52 0.42 1",
    ) -> None:
        half_size = 0.5 * np.asarray(size, dtype=np.float64)
        self._add_geom(name, "box", pos, half_size, euler=euler, rgba=rgba)

    def add_staircase(
        self,
        prefix: str,
        init_pos: tuple[float, float],
        step_height: float,
        step_width: float,
        step_length: float,
        step_count: int,
        yaw: float = 0.0,
        rgba: str = "0.55 0.48 0.40 1",
    ) -> None:
        """上楼梯与下楼梯首尾相连，下楼梯结束后回到地面高度。"""
        local_pos = [0.0, 0.0, 0.5 * step_height]
        for index in range(step_count):
            local_pos[0] += step_width
            local_pos[2] += step_height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self._add_box(
                f"{prefix}_up_{index + 1}",
                [init_pos[0] + x, init_pos[1] + y, local_pos[2]],
                [step_width, step_length, step_height],
                euler=[0.0, 0.0, yaw],
                rgba=rgba,
            )

        for index in range(step_count):
            local_pos[0] += step_width
            local_pos[2] -= step_height
            x, y = rot2d(local_pos[0], local_pos[1], yaw)
            self._add_box(
                f"{prefix}_down_{index + 1}",
                [init_pos[0] + x, init_pos[1] + y, local_pos[2]],
                [step_width, step_length, step_height],
                euler=[0.0, 0.0, yaw],
                rgba=rgba,
            )

    def _place_ramp_box(
        self,
        name: str,
        anchor_pos: np.ndarray,
        anchor_local: np.ndarray,
        length: float,
        width: float,
        thickness: float,
        euler: tuple[float, float, float],
        rgba: str,
        ground_z: float,
        lock_anchor_z: bool = False,
    ) -> np.ndarray:
        half_size = np.array([0.5 * length, 0.5 * width, 0.5 * thickness])
        rotation = euler_to_rot_mat(*euler)
        min_corner_z = box_min_corner_z(half_size, euler)
        center = anchor_pos - rotation @ anchor_local
        if lock_anchor_z:
            actual_min = center[2] + min_corner_z
            if actual_min < ground_z:
                center[2] += ground_z - actual_min
        else:
            center[2] = anchor_pos[2] - min_corner_z
        self._add_box(
            name,
            center.tolist(),
            [length, width, thickness],
            euler=list(euler),
            rgba=rgba,
        )
        return center

    def add_ramp_pair(
        self,
        prefix: str,
        start_pos: tuple[float, float, float],
        length: float,
        width: float,
        pitch_deg: float,
        yaw: float = 0.0,
        rgba: str = "0.50 0.55 0.45 1",
    ) -> None:
        """上坡与下坡在顶点首尾相连，坡道底面贴地不悬空。"""
        pitch = math.radians(pitch_deg)
        thickness = 0.06
        hx = 0.5 * length
        hz = 0.5 * thickness
        ground = np.array(start_pos, dtype=np.float64)

        up_euler = (0.0, -pitch, yaw)
        up_anchor_local = np.array([-hx, 0.0, -hz])
        up_center = self._place_ramp_box(
            f"{prefix}_up",
            ground,
            up_anchor_local,
            length,
            width,
            thickness,
            up_euler,
            rgba,
            ground_z=start_pos[2],
            lock_anchor_z=False,
        )

        up_rotation = euler_to_rot_mat(*up_euler)
        crest_local = np.array([hx, 0.0, hz])
        crest_world = up_center + up_rotation @ crest_local

        down_euler = (0.0, pitch, yaw)
        down_anchor_local = np.array([-hx, 0.0, hz])
        self._place_ramp_box(
            f"{prefix}_down",
            crest_world,
            down_anchor_local,
            length,
            width,
            thickness,
            down_euler,
            rgba,
            ground_z=start_pos[2],
            lock_anchor_z=True,
        )

    def add_plum_blossom_piles(
        self,
        origin: tuple[float, float],
        rows: int = 4,
        cols: int = 5,
        spacing_x: float = 0.78,
        spacing_y: float = 0.72,
        radius: float = 0.25,
        min_height: float = 0.08,
        max_height: float = 0.36,
    ) -> None:
        """沿 +X 方向由低到高再降低，便于从平地走上梅花桩。"""
        for row in range(rows):
            y_offset = (row - (rows - 1) / 2.0) * spacing_y
            row_shift = 0.25 * spacing_x if row % 2 == 1 else 0.0
            for col in range(cols):
                x_offset = (col - (cols - 1) / 2.0) * spacing_x + row_shift
                t = col / (cols - 1) if cols > 1 else 0.5
                profile = 4.0 * t * (1.0 - t)
                pile_height = min_height + profile * (max_height - min_height)
                z_center = 0.5 * pile_height
                self._add_geom(
                    f"plum_pile_{row + 1}_{col + 1}",
                    "cylinder",
                    [origin[0] + x_offset, origin[1] + y_offset, z_center],
                    [radius, 0.5 * pile_height],
                    rgba="0.62 0.42 0.30 1",
                )

    def add_heightfield(
        self,
        name: str,
        image_name: str,
        position: tuple[float, float, float],
        size_xy: tuple[float, float],
        height_scale: float,
        negative_height: float = 0.05,
        nrow: int = 128,
        ncol: int = 128,
        smooth_sigma: float = 3.0,
        edge_blend_fraction: float = 0.30,
        variation_strength: float = 1.6,
        contrast_gamma: float = 0.55,
        coarse_cells: tuple[int, int] = (12, 10),
        medium_detail_weight: float = 0.52,
        fine_detail_weight: float = 0.34,
        micro_detail_weight: float = 0.20,
        medium_detail_cells: tuple[int, int] | None = None,
        fine_detail_cells: tuple[int, int] | None = None,
        micro_detail_cells: tuple[int, int] | None = None,
    ) -> None:
        write_heightfield_png(
            image_name,
            nrow,
            ncol,
            self.seed + len(self.hfield_assets),
            smooth_sigma,
            edge_blend_fraction,
            variation_strength,
            contrast_gamma,
            coarse_cells,
            medium_detail_weight,
            fine_detail_weight,
            micro_detail_weight,
            medium_detail_cells,
            fine_detail_cells,
            micro_detail_cells,
        )
        rel_path = f"../../terrain/hfields/{image_name}"
        self.hfield_assets.append(
            {
                "name": name,
                "file": rel_path,
                "size": [
                    0.5 * size_xy[0],
                    0.5 * size_xy[1],
                    height_scale,
                    negative_height,
                ],
                "position": position,
            }
        )

    def add_random_road(
        self,
        origin: tuple[float, float],
        rows: int = 9,
        cols: int = 12,
        tile_size: float = 0.24,
        gap: float = 0.14,
        height_range: tuple[float, float] = (0.03, 0.16),
    ) -> None:
        for row in range(rows):
            for col in range(cols):
                x_offset = (col - (cols - 1) / 2.0) * (tile_size + gap)
                y_offset = (row - (rows - 1) / 2.0) * (tile_size + gap)
                tile_height = float(self.rng.uniform(*height_range))
                size_jitter = float(self.rng.uniform(0.85, 1.15))
                tile_w = tile_size * size_jitter
                tile_l = tile_size * float(self.rng.uniform(0.85, 1.15))
                roll = float(self.rng.uniform(-0.12, 0.12))
                pitch = float(self.rng.uniform(-0.12, 0.12))
                yaw = float(self.rng.uniform(-0.18, 0.18))
                self._add_box(
                    f"rough_tile_{row + 1}_{col + 1}",
                    [
                        origin[0] + x_offset,
                        origin[1] + y_offset,
                        0.5 * tile_height,
                    ],
                    [tile_l, tile_w, tile_height],
                    euler=[roll, pitch, yaw],
                    rgba="0.48 0.46 0.44 1",
                )

    def build(self) -> ET.ElementTree:
        layout = self.layout

        self.add_heightfield(
            "undulating_terrain_easy",
            "undulating_terrain_easy.png",
            position=(layout.undulating_easy_origin[0], layout.undulating_easy_origin[1], 0.0),
            size_xy=(8.0, 6.0),
            height_scale=0.20,
            nrow=128,
            ncol=128,
            smooth_sigma=2.0,
            edge_blend_fraction=0.38,
            variation_strength=2.8,
            contrast_gamma=0.50,
            coarse_cells=(8, 7),
            medium_detail_weight=0.12,
            fine_detail_weight=0.06,
            micro_detail_weight=0.0,
        )

        self.add_heightfield(
            "undulating_terrain",
            "undulating_terrain.png",
            position=(layout.undulating_origin[0], layout.undulating_origin[1], 0.0),
            size_xy=(10.0, 8.0),
            height_scale=0.42,
            nrow=160,
            ncol=160,
            smooth_sigma=1.2,
            edge_blend_fraction=0.35,
            variation_strength=6.5,
            contrast_gamma=0.24,
            coarse_cells=(15, 12),
            medium_detail_weight=0.52,
            fine_detail_weight=0.34,
            micro_detail_weight=0.20,
        )

        self.add_random_road(layout.rough_road_origin)

        self.add_plum_blossom_piles(layout.plum_origin)

        self.add_staircase(
            "stairs_low",
            layout.stairs_low_origin,
            step_height=0.08,
            step_width=0.28,
            step_length=1.2,
            step_count=6,
            rgba="0.58 0.50 0.42 1",
        )
        self.add_staircase(
            "stairs_mid",
            layout.stairs_mid_origin,
            step_height=0.12,
            step_width=0.28,
            step_length=1.2,
            step_count=8,
            rgba="0.54 0.46 0.38 1",
        )
        self.add_staircase(
            "stairs_high",
            layout.stairs_high_origin,
            step_height=0.18,
            step_width=0.28,
            step_length=1.2,
            step_count=10,
            rgba="0.50 0.42 0.34 1",
        )

        ramp_specs = [
            ("ramp_10", 10.0, -3.0),
            ("ramp_20", 20.0, 0.0),
            ("ramp_30", 30.0, 3.0),
        ]
        for name, pitch_deg, y_offset in ramp_specs:
            self.add_ramp_pair(
                name,
                start_pos=(layout.ramp_row_x, y_offset, 0.0),
                length=layout.ramp_length,
                width=1.2,
                pitch_deg=pitch_deg,
            )

        robot_xml_name = os.path.basename(ROBOT_XML)
        root = ET.Element("mujoco", model=f"{ROBOT}_terrain_scene")
        ET.SubElement(root, "include", file=robot_xml_name)

        asset = ET.SubElement(root, "asset")
        ET.SubElement(
            asset,
            "material",
            name="terrain_mat",
            rgba="0.45 0.52 0.42 1",
            specular="0.2",
            shininess="0.3",
        )
        for hfield in self.hfield_assets:
            item = ET.SubElement(asset, "hfield", name=hfield["name"])
            item.attrib["size"] = list_to_str(hfield["size"])
            item.attrib["file"] = hfield["file"]

        worldbody = ET.SubElement(root, "worldbody")
        for geom in self.geoms:
            worldbody.append(geom)
        for hfield in self.hfield_assets:
            hfield_geom = ET.SubElement(worldbody, "geom")
            hfield_geom.attrib["name"] = hfield["name"] + "_geom"
            hfield_geom.attrib["type"] = "hfield"
            hfield_geom.attrib["hfield"] = hfield["name"]
            hfield_geom.attrib["pos"] = list_to_str(hfield["position"])
            hfield_geom.attrib["material"] = "terrain_mat"
            hfield_geom.attrib["friction"] = TERRAIN_FRICTION
            hfield_geom.attrib["condim"] = "3"

        return ET.ElementTree(root)

    def save(self) -> str:
        scene_path = os.path.join(RESOURCES_DIR, SCENE_XML)
        os.makedirs(os.path.dirname(scene_path), exist_ok=True)
        tree = self.build()
        ET.indent(tree, space="  ")
        tree.write(scene_path, encoding="unicode", xml_declaration=False)
        self._prepend_xml_header(scene_path)
        return scene_path

    @staticmethod
    def _prepend_xml_header(path: str) -> None:
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        if not content.startswith("<?xml"):
            with open(path, "w", encoding="utf-8") as handle:
                handle.write('<?xml version="1.0" encoding="utf-8"?>\n')
                handle.write(content)


def validate_scene(scene_path: str) -> bool:
    try:
        import mujoco
    except ImportError:
        print(f"Skipped MuJoCo validation (mujoco not installed): {scene_path}")
        return False

    mujoco.MjModel.from_xml_path(scene_path)
    print(f"Validated scene: {scene_path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate tiangong3 (tg3) terrain scene for xsim_mujoco"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    builder = TerrainSceneBuilder(seed=args.seed)
    scene_path = builder.save()
    validate_scene(scene_path)


if __name__ == "__main__":
    main()
