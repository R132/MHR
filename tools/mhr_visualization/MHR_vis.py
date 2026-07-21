"""
ScenePic 3D Visualization Utilities

This module provides a high-level Python wrapper around the ScenePic library for creating
interactive 3D visualizations of computer vision and robotics assets. It simplifies the
process of visualizing complex 3D data including meshes, point clouds, line set animations,
and coordinate frames in both Jupyter notebook environments and standalone HTML files. It
is designed to be used for FAST debugging purposes during development and research.

Key Features:
    • Multi-asset visualization: Support for meshes, point clouds, line sets, coordinate frames
    • IDE integration: Can be used in Bento
    • Animation support: Multi-frame temporal data with timeline controls
    • Flexible rendering: Automatic and manual camera positioning, customizable lighting
    • Interactive output: Mouse navigation (rotation, zoom, pan) in browser
    • Export capabilities: Self-contained HTML files for sharing and archiving
    • Color management: Automatic rainbow colormaps or custom color schemes
    • Label support: Text labels for point cloud data visualization

Components:
    ScenepicVisualization: Main visualization class providing the high-level interface
    Helper functions and type alias: Color validation, normalization, and utility functions

Typical Workflow:
    1. Create a ScenepicVisualization instance
    2. Add 3D assets using add_meshes(), add_point_clouds(), add_line_sets(), or add_coordinate_frames()
    3. Optionally customize lighting, camera position, or colors
    4. Display in Jupyter with show() or save to HTML with save_to_html()

Example Usage:
    ```python
    import numpy as np
    from xrcia.projects.tracked_assets.core.visualization.scenepic_visualization import ScenepicVisualization

    # Create visualization instance
    viz = ScenepicVisualization()

    # Add some sample data
    points = np.random.rand(100, 3)
    viz.add_point_clouds([[points]], point_labels=[['Point ' + str(i) for i in range(100)]])

    # Add coordinate frame at origin
    origins = [[[0, 0, 0]]]
    orientations = [[[np.eye(3)]]]
    viz.add_coordinate_frames(origins, orientations)

    # Display in notebook
    viz.show()

    # Or save to file
    viz.save_to_html("visualization.html")
    ```

Dependencies:
    • scenepic: Core 3D visualization library with WebGL rendering
    • trimesh: Mesh processing and validation
    • numpy: Numerical computation and array handling
    • matplotlib: Color management and colormaps
    • IPython: Bento display integration

Units and Coordinate System:
    • All measurements are assumed to be in METERS
    • Uses right-handed coordinate system (X=right, Y=up, Z=forward)
    • Coordinate frames follow standard RGB coloring (X=Red, Y=Green, Z=Blue)

Performance Considerations:
    • Large datasets may impact browser performance
    • Multi-frame animations require more memory and processing
    • Point cloud instancing is optimized for performance
    • HTML export includes all data inline (no external dependencies)

Error Handling:
    • Comprehensive input validation with descriptive error messages
    • Type checking for data structures and color formats
    • Frame count consistency validation across asset types
    • Graceful handling of edge cases (empty data, single frames, etc.)

Browser Compatibility:
    • Requires modern browser with WebGL support
    • Tested on Chrome, Firefox, Safari, Edge
    • Mobile browser support varies based on device capabilities

Meta Internal Usage:
    This module is designed for use within Meta's computer vision and robotics projects,
    particularly for visualizing tracked assets, motion capture data, and 3D scene
    understanding results. It integrates with internal tooling and Bento notebook environments.

TODO(T238674328):
    • Support mesh texture
    • Support untracked sequence of data
"""

from functools import reduce
from typing import Any, Union

import IPython
import torch
import scipy.spatial.transform as R

import matplotlib.pyplot as plt
import numpy as np
import scenepic as sp
import trimesh

_mhr_model: torch.jit.ScriptModule | None = None


def load_model(path: str) -> torch.jit.ScriptModule:
    """Load the MHR scripted model from the given path and store it as a module-level global."""
    global _mhr_model
    _mhr_model = torch.jit.load(path)
    return _mhr_model

# Custom type definition for mesh colors configuration
# Supports n x 3 arrays of both integers and floats
MeshMonoColors = Union[
    # Per-mesh colors
    np.ndarray[tuple[int], np.dtype[np.uint8]],  # 3 uint8 (0-255)
    np.ndarray[tuple[int], np.dtype[np.int32]],  # 3 int32 (0-255)
    np.ndarray[tuple[int], np.dtype[np.int64]],  # 3 int64 (0-255)
    np.ndarray[tuple[int], np.dtype[np.float32]],  # 3 float32 (0.0-1.0)
    np.ndarray[tuple[int], np.dtype[np.float64]],  # 3 float64 (0.0-1.0)
]
MeshPervertexColors = Union[
    # Per-vertex colors
    np.ndarray[tuple[int, int], np.dtype[np.uint8]],  # v x 3 uint8 (0-255)
    np.ndarray[tuple[int, int], np.dtype[np.int32]],  # v x 3 int32 (0-255)
    np.ndarray[tuple[int, int], np.dtype[np.int64]],  # v x 3 int64 (0-255)
    np.ndarray[tuple[int, int], np.dtype[np.float32]],  # v x 3 float32 (0.0-1.0)
    np.ndarray[tuple[int, int], np.dtype[np.float64]],  # v x 3 float64 (0.0-1.0)
]
# Define colors based on xrcia.projects.tracked_assets.core.visualization.image_draw.Color, but with float32 dtype
_WHITE: MeshMonoColors = np.array([1.0, 1.0, 1.0])
_BLACK: MeshMonoColors = np.array([0.0, 0.0, 0.0])
_LIGHT_GRAY: MeshMonoColors = np.array([0.7, 0.7, 0.7])
_DARK_GRAY: MeshMonoColors = np.array([0.3, 0.3, 0.3])


class ScenepicVisualization:
    """
    A wrapper around the ScenePic library to facilitate 3D visualization of various assets.

    This class provides a high-level interface for creating interactive 3D visualizations
    using the ScenePic library. It supports multiple types of 3D assets including meshes,
    point clouds, line sets, and coordinate frames. The visualization can be displayed
    in Bento or saved as standalone HTML files.

    Features:
        - Multi-frame animation support for temporal data
        - Automatic color generation using rainbow colormaps
        - Flexible camera positioning (automatic or manual)
        - Support for mesh, point cloud, line set, and coordinate frame visualization
        - Customizable shading and lighting
        - Label support for point clouds
        - Interactive HTML output for web viewing

    * Important Note: This class assumes METERS as the unit of measurement for all assets.

    Example:
        ```python
        # Create a visualization
        viz = ScenepicVisualization()

        # Add meshes (single frame or multi-frame)
        viz.add_meshes(meshes_list)

        # Add point clouds with labels (all visible by default)
        viz.add_point_clouds(point_clouds_list, point_labels=labels)

        # Add line sets with individual visibility control
        viz.add_line_sets(start_points, end_points, hide_line_sets=[False, True, False])

        # Add coordinate frames (all visible by default)
        viz.add_coordinate_frames(origins, orientations)

        # Display or save
        viz.show()
        viz.save_to_html("output.html")
        ```
    """

    def __init__(self) -> None:
        """
        Initialize a new ScenepicVisualization instance.

        Creates a new ScenePic scene with default camera, lighting, and shading settings.
        All asset types (meshes, point clouds, line sets, coordinate frames) are initially
        disabled and will be enabled when assets are added.

        The scene bounding box is automatically calculated as assets are added, and the
        camera will be positioned automatically unless manually set.

        Default Settings:
            - White background with standard ambient and directional lighting
            - Automatic camera positioning based on scene bounding box
            - Rainbow colormap for automatically generated colors
            - Point size: 0.01 units
            - Line thickness: 0.01 units for line set visualization
            - Coordinate frame size: 0.1 units
            - All asset types visible by default
        """
        self.scene: sp.Scene = sp.Scene()

        self.num_meshes_per_frame: int = 0
        self.meshes: list[list[trimesh.Trimesh]] = []
        self.mesh_names: list[str] = []
        self.mesh_colors: list[MeshMonoColors | MeshPervertexColors] = []
        self.mesh_opacity: list[float] = []
        self.mesh_vertex_uvs: (
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]] | None] | None
        ) = None
        self.mesh_texture_images: (
            list[np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] | None] | None
        ) = None

        self.num_point_clouds_per_frame: int = 0
        self.point_clouds: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ] = []
        self.point_cloud_names: list[str] = []
        self.point_cloud_colors: list[MeshMonoColors | MeshPervertexColors] = []
        self.point_cloud_opacity: list[float] = []
        self.point_size: float = 0.01
        self.point_labels: list[list[str]] = []
        self.point_label_opacity: list[float] = []
        self.label_color: MeshMonoColors | MeshPervertexColors = _BLACK
        self.point_label_offset: float = 0.01
        self.label_size_in_pixel: int = 20

        self.num_line_sets_per_frame: int = 0
        self.start_points: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ] = []
        self.end_points: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ] = []
        self.line_set_names: list[str] = []
        self.line_set_colors: list[MeshMonoColors | MeshPervertexColors] = []
        self.line_set_opacity: list[float] = []
        self.line_type: str = "thickline"
        self.line_start_thickness: float = 0.01

        self.num_coordinate_frames_per_frame: int = 0
        self.coordinate_frame_origins: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ] = []
        self.coordinate_frame_orientations: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ] = []
        self.coordinate_frame_names: list[str] = []
        self.coordinate_frame_opacity: list[float] = []
        self.frame_size: float = 0.1

        self.scene_bbox: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.zeros(
            (2, 3), dtype=np.float32
        )
        self.scene_bbox[0] = -np.inf * np.ones(3)
        self.scene_bbox[1] = np.inf * np.ones(3)

        # Initialize with default camera parameters using simplified constructor
        self.camera: sp.Camera = sp.Camera(
            center=np.array([0.0, 0.0, 3.0], dtype=np.float32),
            look_at=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            fov_y_degrees=30.0,
            aspect_ratio=1.0,
            far_crop_distance=20.0,
            near_crop_distance=0.01,
        )
        self.camera_is_manually_set = False

        # Set the default background color to white.
        self.shading = sp.Shading(
            bg_color=_WHITE,
            ambient_light_color=_LIGHT_GRAY,
            directional_light_color=_DARK_GRAY,
            directional_light_dir=np.array([2, 1, 2]),
        )

    def add_meshes(
        self,
        meshes: list[list[trimesh.Trimesh]],
        mesh_names: list[str] | None = None,
        mesh_colors: list[MeshMonoColors | MeshPervertexColors | None] | None = None,
        mesh_opacity: list[float] | None = None,
        mesh_vertex_uvs: list[np.ndarray[tuple[int, int], np.dtype[np.float32]] | None]
        | None = None,
        mesh_texture_images: list[
            np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] | None
        ]
        | None = None,
    ) -> None:
        """
        Add meshes to the scene for visualization.

        Meshes can be provided as a single frame or multiple frames for animation.
        Each mesh will be rendered with customizable colors and names. Optionally,
        meshes can be textured using UV coordinates and texture images.

        Args:
            meshes: List of lists containing Trimesh objects. Each inner list represents
                   a frame, and each Trimesh is a mesh in that frame. For single frame
                   data, provide a list containing one list of meshes.
            mesh_names: Optional names for each mesh. If not provided, names will be
                       automatically generated as "mesh_0", "mesh_1", etc.
            mesh_colors: Optional colors for each mesh. Can be per-mesh uniform colors,
                        per-vertex colors, or None for individual meshes. If not provided,
                        colors will be automatically generated using a rainbow colormap.
                        For mixed rendering with textures, individual elements can be None
                        for textured meshes while providing colors for non-textured meshes.
            mesh_opacity: Optional list of opacity values for specific meshes. Each value
                          should be between 0.0 (fully transparent) and 1.0 (fully opaque).
                          If provided, must have the same length as the number of meshes per frame.
                          If not provided, all meshes will be fully opaque (opacity=1.0) by default.
            mesh_vertex_uvs: Optional list of UV coordinate arrays for per-vertex texture mapping. Each array
                     should have shape (N, 2) where N is the number of vertices in the corresponding
                     mesh. UV coordinates should be in range [0, 1]. Individual elements can be None
                     for non-textured meshes in mixed rendering scenarios. If provided, must have the
                     same length as meshes and be paired with mesh_texture_images.
            mesh_texture_images: Optional list of texture image arrays. Each array should be
                                in RGB format with shape (H, W, 3) and uint8 values [0, 255].
                                If provided, must have the same length as meshes and be paired
                                with mesh_uvs.

        Raises:
            ValueError: If meshes are not in the correct list of lists format, or if UV
                       coordinates and texture images are not properly paired.

        Note:
            - All frames must contain the same number of meshes
            - Mesh colors can be uniform (one color per mesh) or per-vertex
            - When textures are provided, vertex colors are ignored
            - UV coordinates and texture images must be provided together
            - Automatically updates the scene bounding box
            - Individual mesh visibility can be controlled via the hide_meshes parameter
        """
        self.meshes, self.num_meshes_per_frame = self._validate_input_is_list_of_lists(
            meshes, "Meshes"
        )

        self.mesh_names = (
            mesh_names
            if mesh_names is not None
            else [f"mesh_{i}" for i in range(self.num_meshes_per_frame)]
        )

        self.mesh_opacity = (
            mesh_opacity
            if mesh_opacity is not None
            else [1.0] * self.num_meshes_per_frame
        )

        self._setup_mesh_texture_and_vertex_color(
            mesh_colors, mesh_vertex_uvs, mesh_texture_images
        )

        # Get the bounding box for the added meshes.
        max_mesh_corner = -np.inf * np.ones(3)
        min_mesh_corner = np.inf * np.ones(3)
        for one_frame_meshes in self.meshes:
            for mesh in one_frame_meshes:
                vertices = mesh.vertices
                max_mesh_corner = np.maximum(max_mesh_corner, np.max(vertices, axis=0))
                min_mesh_corner = np.minimum(min_mesh_corner, np.min(vertices, axis=0))

        # Update the scene bounding box according to the bounding box of all the added meshes.
        self.scene_bbox[0] = np.maximum(self.scene_bbox[0], max_mesh_corner)
        self.scene_bbox[1] = np.minimum(self.scene_bbox[1], min_mesh_corner)

    def add_point_clouds(
        self,
        point_clouds: list[list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]],
        point_cloud_names: list[str] | None = None,
        point_cloud_colors: list[MeshMonoColors | MeshPervertexColors | None]
        | None = None,
        point_cloud_opacity: list[float] | None = None,
        point_size: float = 0.01,
        point_labels: list[list[str]] | None = None,
        point_label_opacity: list[float] | None = None,
        label_color: MeshMonoColors | MeshPervertexColors = _BLACK,
        label_offset: float = 0.01,
        label_size_in_pixel: int = 20,
    ) -> None:
        """
        Add point clouds to the scene for visualization.

        Point clouds are rendered as instanced spheres with customizable size and colors.
        Optional text labels can be attached to individual points.

        Args:
            point_clouds: List of lists containing point cloud arrays. Each inner list
                         represents a frame, and each array contains 3D points (Nx3).
            point_cloud_names: Optional names for each point cloud. If not provided,
                              names will be automatically generated as "point_cloud_0", etc.
            point_cloud_colors: Optional colors for each point cloud. Can be uniform colors
                               or per-point colors. If not provided, colors will be
                               automatically generated using a rainbow colormap.
            point_cloud_opacity: Optional list of opacity values for specific point clouds.
                                 Each value should be between 0.0 (fully transparent) and 1.0 (fully opaque).
                                 If provided, must have the same length as the number of point
                                 clouds per frame. If not provided, all point clouds will be
                                 fully opaque (opacity=1.0) by default.
            point_size: Radius of the spheres used to render each point. Default is 0.01.
            point_labels: Optional text labels for individual points. Must match the
                         structure of point_clouds (list of lists of strings).
            point_label_opacity: Optional list of opacity values for specific point cloud labels.
                                 Each value should be between 0.0 (fully transparent) and 1.0 (fully opaque).
                                 If provided, must have the same length as the number of point
                                 clouds per frame. If not provided, all point labels will be
                                 fully opaque (opacity=1.0) by default. Note: The effective opacity
                                 of labels will be the minimum of the label opacity and the corresponding
                                 point cloud opacity.
            label_color: Color for text labels. Default is black.
            label_offset: Offset distance for labels from point positions. Default is 0.01.
            label_size_in_pixel: Font size for labels in pixels. Default is 20.

        Raises:
            ValueError: If point_clouds are not in the correct list of lists format,
                       or if point_labels don't match the point cloud structure.

        Note:
            - All frames must contain the same number of point clouds
            - Point labels, if provided, must match the number of points in each cloud
            - Automatically updates the scene bounding box
            - Individual point cloud visibility can be controlled via the hide_point_clouds parameter
            - Individual point label visibility can be controlled via the hide_point_labels parameter
        """
        self.point_clouds, self.num_point_clouds_per_frame = (
            self._validate_input_is_list_of_lists(point_clouds, "Point clouds")
        )
        self.point_cloud_opacity = (
            point_cloud_opacity
            if point_cloud_opacity is not None
            else [1.0] * self.num_point_clouds_per_frame
        )
        self.point_size = point_size

        self.point_cloud_names = (
            point_cloud_names
            if point_cloud_names is not None
            else [f"point_cloud_{i}" for i in range(self.num_point_clouds_per_frame)]
        )

        # Generate evenly spaced values from 0 to 0.8 for rainbow_r colormap, so that 0 -> red, 0.85 -> blue.
        color_indices = np.linspace(0, 0.85, self.num_point_clouds_per_frame)
        rainbow_cmap = plt.cm.rainbow_r
        # We keep only the RGB values and discard the alpha channel
        self.point_cloud_colors = rainbow_cmap(color_indices)[:, :3]
        if point_cloud_colors is not None:
            # Sample self.num_point_clouds_per_frame colors according to the rainbow color map in matplotlib.
            _validate_mesh_colors(point_cloud_colors, self.num_point_clouds_per_frame)
            self.point_cloud_colors = _normalize_mesh_colors(
                point_cloud_colors, self.point_cloud_colors
            )

        # Get the bounding box for the added point clouds.
        max_point_cloud_corner = -np.inf * np.ones(3)
        min_point_cloud_corner = np.inf * np.ones(3)
        for one_frame_point_clouds in self.point_clouds:
            for point_cloud in one_frame_point_clouds:
                max_point_cloud_corner = np.maximum(
                    max_point_cloud_corner, np.max(point_cloud, axis=0)
                )
                min_point_cloud_corner = np.minimum(
                    min_point_cloud_corner, np.min(point_cloud, axis=0)
                )

        # Update the scene bounding box according to the bounding box of all the added point clouds.
        self.scene_bbox[0] = np.maximum(self.scene_bbox[0], max_point_cloud_corner)
        self.scene_bbox[1] = np.minimum(self.scene_bbox[1], min_point_cloud_corner)

        if point_labels is not None:
            if not isinstance(point_labels[0], list):
                raise ValueError(
                    "Point labels must be a list of lists, where each inner list represents a single frame of point labels."
                )
            self.point_labels = point_labels
            for i_point_cloud, point_label in enumerate(self.point_labels):
                if not self.point_clouds[0][i_point_cloud].shape[0] == len(point_label):
                    raise ValueError(
                        "List of point labels should match the size of each point cloud in one frame."
                        f"Got {len(point_label)} point labels for a point cloud with size of f{self.point_clouds[0][i_point_cloud].shape[0]}"
                    )
            self.label_color = label_color
            self.point_label_offset = label_offset
            self.label_size_in_pixel = label_size_in_pixel

            # Handle point_label_opacity parameter
            self.point_label_opacity = (
                point_label_opacity
                if point_label_opacity is not None
                else [1.0] * self.num_point_clouds_per_frame
            )

    def add_line_sets(
        self,
        start_points: list[list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]],
        end_points: list[list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]],
        line_set_names: list[str] | None = None,
        line_set_colors: list[MeshMonoColors | MeshPervertexColors | None]
        | None = None,
        line_set_opacity: list[float] | None = None,
        line_type: str = "thickline",  # "line", or "thickline".
        line_start_thickness: float = 0.01,  # Only used when line_type is "thickline".
    ) -> None:
        """
        Add line sets to the scene for visualization.

        Line sets are rendered as lines or thick lines connecting start and end points.
        Each line set can have multiple lines defined by start-end points pairs.

        Args:
            start_points: List of lists containing start point positions. Each inner list
                          represents a frame, and each array contains point positions (Nx3).
            end_points: List of lists containing end point positions. Must have the same
                         structure as start_points. Each start point connects to the
                         corresponding end point to form a line.
            line_set_names: Optional names for each line set. If not provided, names will be
                           automatically generated as "line_set_0", "line_set_1", etc.
            line_set_colors: Optional colors for each line set. If not provided, colors will
                            be automatically generated using a rainbow colormap.
            line_set_opacity: Optional list of opacity values for specific line sets.
                              Each value should be between 0.0 (fully transparent) and 1.0 (fully opaque).
                              If provided, must have the same length as the number of line sets
                              per frame. If not provided, all line sets will be fully opaque
                              (opacity=1.0) by default.
            line_type: Type of line rendering. Either "line" for simple lines or "thickline"
                      for variable-thickness lines. Default is "thickline".
            line_start_thickness: Starting thickness for lines when using "thickline" type.
                                 Only used when line_type is "thickline". Default is 0.01.

        Raises:
            ValueError: If start_points and end_points don't have matching structures,
                       or if line_type is not valid.

        Note:
            - All frames must contain the same number of line sets
            - Start and end points must have matching array shapes within each frame
            - Each start point connects to its corresponding end point
            - Automatically updates the scene bounding box
            - Individual line set visibility can be controlled via the hide_line_sets parameter
        """
        self.start_points, self.end_points, self.num_line_sets_per_frame = (
            self._validate_input_pair_are_list_of_lists(
                start_points, "Start points", end_points, "End points"
            )
        )

        self.line_set_opacity = (
            line_set_opacity
            if line_set_opacity is not None
            else [1.0] * self.num_line_sets_per_frame
        )

        self.line_set_names = (
            line_set_names
            if line_set_names is not None
            else [f"line_set_{i}" for i in range(self.num_line_sets_per_frame)]
        )

        # Generate evenly spaced values from 0 to 0.8 for rainbow_r colormap, so that 0 -> red, 0.85 -> blue.
        color_indices = np.linspace(0, 0.85, self.num_line_sets_per_frame)
        rainbow_cmap = plt.cm.rainbow_r
        # We keep only the RGB values and discard the alpha channel
        self.line_set_colors = rainbow_cmap(color_indices)[:, :3]
        if line_set_colors is not None:
            # Sample self.num_line_sets_per_frame colors according to the rainbow color map in matplotlib.
            _validate_mesh_colors(line_set_colors, self.num_line_sets_per_frame)
            self.line_set_colors = _normalize_mesh_colors(
                line_set_colors, self.line_set_colors
            )

        # Get the bounding box for the added line sets.
        max_line_set_corner = -np.inf * np.ones(3)
        min_line_set_corner = np.inf * np.ones(3)
        for frame_idx in range(len(self.start_points)):
            for line_set_idx in range(self.num_line_sets_per_frame):
                line_starts = self.start_points[frame_idx][line_set_idx]
                line_ends = self.end_points[frame_idx][line_set_idx]
                all_points = np.concatenate([line_starts, line_ends], axis=0)
                max_line_set_corner = np.maximum(
                    max_line_set_corner, np.max(all_points, axis=0)
                )
                min_line_set_corner = np.minimum(
                    min_line_set_corner, np.min(all_points, axis=0)
                )

        # Update the scene bounding box according to the bounding box of all the added line sets.
        self.scene_bbox[0] = np.maximum(self.scene_bbox[0], max_line_set_corner)
        self.scene_bbox[1] = np.minimum(self.scene_bbox[1], min_line_set_corner)

        if not line_type.lower() in ["line", "thickline"]:
            raise ValueError(
                f"Invalid line type {line_type}. Must be one of 'line', or 'thickline'."
            )
        self.line_type = line_type.lower()
        self.line_start_thickness = line_start_thickness

    def add_coordinate_frames(
        self,
        coordinate_frame_origins: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ],
        coordinate_frame_orientations: list[
            list[np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        ],
        coordinate_frame_opacity: list[float] | None = None,
        frame_size: float = 0.1,
        coordinate_frame_names: list[str] | None = None,
    ) -> None:
        """
        Add coordinate frames to the scene for visualization.

        Coordinate frames are rendered using ScenePic's built-in coordinate axes visualization,
        showing standard RGB-colored axes (X=Red, Y=Green, Z=Blue).

        Args:
            coordinate_frame_origins: List of lists containing coordinate frame origins.
                                    Each inner list represents a frame, and each array
                                    contains 3D origin positions (Nx3).
            coordinate_frame_orientations: List of lists containing coordinate frame orientations.
                                         Must have the same structure as origins. Each array
                                         contains 3x3 rotation matrices defining frame orientation.
            coordinate_frame_opacity: Optional list of opacity values for specific coordinate frame.
                              Each value should be between 0.0 (fully transparent) and 1.0 (fully opaque).
                              If provided, must have the same length as the number of coordinate frames
                              per frame. If not provided, all coordinate frames will be fully opaque
                              (opacity=1.0) by default.
            frame_size: Length of each coordinate axis in world units. Default is 0.1.
            coordinate_frame_names: Optional names for each coordinate frame. If not provided,
                                  names will be automatically generated as "coordinate_frame_0", etc.

        Raises:
            ValueError: If origins and orientations don't have matching structures.

        Note:
            - All frames must contain the same number of coordinate frames
            - Origins and orientations must have matching array shapes within each frame
            - Uses ScenePic's native coordinate axes with standard RGB coloring
            - Automatically updates the scene bounding box
        """
        (
            self.coordinate_frame_origins,
            self.coordinate_frame_orientations,
            self.num_coordinate_frames_per_frame,
        ) = self._validate_input_pair_are_list_of_lists(
            coordinate_frame_origins,
            "Coordinate frame origins",
            coordinate_frame_orientations,
            "Coordinate frame orientations",
        )

        self.coordinate_frame_names = (
            coordinate_frame_names
            if coordinate_frame_names is not None
            else [
                f"coordinate_frame_{i}"
                for i in range(self.num_coordinate_frames_per_frame)
            ]
        )

        self.coordinate_frame_opacity = (
            coordinate_frame_opacity
            if coordinate_frame_opacity is not None
            else [1.0] * self.num_coordinate_frames_per_frame
        )

        # Get the bounding box for the added coordinate frames.
        max_coordinate_frame_corner = -np.inf * np.ones(3)
        min_coordinate_frame_corner = np.inf * np.ones(3)
        for frame_idx in range(len(self.coordinate_frame_origins)):
            for coordinate_frame_idx in range(self.num_coordinate_frames_per_frame):
                origin = self.coordinate_frame_origins[frame_idx][coordinate_frame_idx]
                # Consider the extent of the coordinate frame axes based on frame_size
                max_coordinate_frame_corner = np.maximum(
                    max_coordinate_frame_corner, np.max(origin + frame_size, axis=0)
                )
                min_coordinate_frame_corner = np.minimum(
                    min_coordinate_frame_corner, np.min(origin - frame_size, axis=0)
                )

        # Update the scene bounding box according to the bounding box of all the added coordinate frames.
        self.scene_bbox[0] = np.maximum(self.scene_bbox[0], max_coordinate_frame_corner)
        self.scene_bbox[1] = np.minimum(self.scene_bbox[1], min_coordinate_frame_corner)

        self.frame_size = frame_size

    def set_shading(
        self,
        background_color: MeshMonoColors | None = None,
        ambient_light_color: MeshMonoColors | None = None,
        directional_light_color: MeshMonoColors | None = None,
        directional_light_dir: np.ndarray[tuple[int], np.dtype[np.float32]]
        | None = None,
    ) -> None:
        """
        Configure the lighting and shading properties of the scene.

        Sets up the background color, ambient lighting, and directional lighting
        for the 3D scene. These properties affect how all objects in the scene
        are illuminated and rendered.

        Args:
            background_color: RGB color for the scene background. Default is black [0,0,0].
                            Can be specified as float32 values in range [0.0, 1.0] or
                            uint8 values in range [0, 255].
            ambient_light_color: RGB color for ambient lighting that illuminates all
                               objects uniformly. Default is light gray [0.7, 0.7, 0.7].
            directional_light_color: RGB color for directional lighting that creates
                                   shadows and highlights. Default is dark gray [0.3, 0.3, 0.3].
            directional_light_dir: Direction vector for the directional light source.
                                 Default is [2, 1, 2], pointing diagonally down and forward.

        Note:
            - Colors will be automatically normalized to the appropriate format
            - The lighting setup affects all objects in the scene
            - Changes take effect when the scene is next rendered
        """
        self.shading = sp.Shading(
            bg_color=_normalize_mesh_colors([background_color], [_WHITE])[0],
            ambient_light_color=_normalize_mesh_colors(
                [ambient_light_color], [_LIGHT_GRAY]
            )[0],
            directional_light_color=_normalize_mesh_colors(
                [directional_light_color], [_DARK_GRAY]
            )[0],
            directional_light_dir=directional_light_dir
            if directional_light_dir is not None
            else np.array([2, 1, 2]),
        )

    def show(self, view_height: int = 600, view_width: int = 600) -> None:
        """
        Display the 3D scene in the current Jupyter notebook cell.

        Renders the scene with all added assets (meshes, point clouds, line sets,
        coordinate frames) and displays it as an interactive HTML widget that can
        be manipulated with mouse controls.

        Args:
            view_height: Height of the rendered view in pixels. Default is 600.
            view_width: Width of the rendered view in pixels. Default is 600.

        Note:
            - This method only works in Jupyter notebook environments
            - The rendered scene supports mouse navigation (rotation, zoom, pan)
            - If multiple frames are present, the scene will include timeline controls
            - The camera will be automatically positioned unless manually set
        """
        self._prepare_rendering(width=view_width, height=view_height)
        html_string = self._generate_html_string()
        html_object = IPython.display.HTML(html_string)
        IPython.display.display(html_object)

    def set_camera(
        self,
        camera_location: np.ndarray[tuple[int], np.dtype[np.float32]],
        camera_look_at: np.ndarray[tuple[int], np.dtype[np.float32]],
        fov_degrees: float,
    ) -> None:
        """
        Manually set the camera position and orientation for the scene.

        This overrides the automatic camera positioning based on scene bounding box.
        The camera will be fixed at the specified location and orientation for all frames.

        Args:
            camera_location: The 3D position of the camera in world coordinates.
            camera_look_at: The 3D point the camera should focus on in world coordinates.
            fov_degrees: The vertical field of view angle in degrees. Controls zoom level.

        Note:
            - Once manually set, the camera position won't be updated automatically
            - A small offset (1e-8) is added to prevent numerical issues when
              camera location and look-at point align with coordinate axes
            - Aspect ratio is automatically set to 1.0 for square viewports
        """
        self.camera = sp.Camera(
            center=camera_location,
            look_at=camera_look_at
            + 1e-8,  # Added 1e-8 to avoid nan when center and look_at align with coordinate axis.
            fov_y_degrees=fov_degrees,
            aspect_ratio=1.0,
            far_crop_distance=20.0,
            near_crop_distance=0.01,
        )
        self.camera_is_manually_set = True

    def render_to_html_string(
        self, view_height: int = 600, view_width: int = 600
    ) -> str:
        """
        Generate an HTML string representing the 3D scene.

        Renders the scene with all added assets (meshes, point clouds, line sets,
        coordinate frames) and returns the generated HTML string. This can be used
        to embed the scene in other HTML documents or to save it to a file.

        Args:
            view_height: Height of the rendered view in pixels. Default is 600.
            view_width: Width of the rendered view in pixels. Default is 600.

        Returns:
            str: The generated HTML string representing the 3D scene.
        """
        self._prepare_rendering(width=view_width, height=view_height)
        return self._generate_html_string()

    def render_to_html_string2(
        self, view_height: int = 600, view_width: int = 600
    ) -> str:
        """
        Generate an HTML string representing the 3D scene.

        Renders the scene with all added assets (meshes, point clouds, line sets,
        coordinate frames) and returns the generated HTML string. This can be used
        to embed the scene in other HTML documents or to save it to a file.

        Args:
            view_height: Height of the rendered view in pixels. Default is 600.
            view_width: Width of the rendered view in pixels. Default is 600.

        Returns:
            str: The generated HTML string representing the 3D scene.
        """
        self._prepare_rendering(width=view_width, height=view_height)
        return self._generate_html_string2()

    def save_to_html(
        self, html_file_path: str, view_height: int = 600, view_width: int = 600
    ) -> None:
        """
        Save the 3D scene as a standalone HTML file.

        Creates a complete HTML document containing the ScenePic visualization and
        saves it to the specified file path. The HTML file is self-contained and
        can be opened in any modern web browser.

        Args:
            html_file_path: Path where the HTML file should be saved. Should have
                           a .html extension.
            view_height: Height of the rendered view in pixels. Default is 600.
            view_width: Width of the rendered view in pixels. Default is 600.

        Note:
            - The generated HTML file is completely standalone
            - No external dependencies or internet connection required
            - Compatible with modern web browsers supporting WebGL
            - Includes mouse navigation controls (rotation, zoom, pan)
            - Timeline controls are included if multiple frames are present
        """
        self._prepare_rendering(width=view_width, height=view_height)
        html_string = self._generate_html_string()
        with open(html_file_path, "w") as f:
            f.write(html_string)

    def _validate_input_is_list_of_lists(
        self,
        input_data: list[Any],
        data_type: str,
    ) -> tuple[list[Any], int]:
        """
        Validate the input data is in list of lists format.

        Args:
            input_data: The input data to validate
            data_type: Description of the data type for error messages

        Returns:
            tuple: (normalized_data, items_per_frame)

        Raises:
            ValueError: If validation fails
        """
        # Check if input is already list of lists format
        for frame_data in input_data:
            if not isinstance(frame_data, list):
                raise ValueError(
                    f"{data_type} must be a list of lists, where each inner list represents a frame of {data_type.lower()}."
                )
        return input_data, len(input_data[0])

    def _setup_mesh_texture_and_vertex_color(
        self,
        mesh_colors: list[MeshMonoColors | MeshPervertexColors | None] | None = None,
        mesh_vertex_uvs: list[np.ndarray[tuple[int, int], np.dtype[np.float32]] | None]
        | None = None,
        mesh_texture_images: list[
            np.ndarray[tuple[int, int, int], np.dtype[np.uint8]] | None
        ]
        | None = None,
    ) -> None:
        """
        Set up mesh texture and vertex colors for rendering.

        Args:
            mesh_colors: Optional colors for each mesh. Can be per-mesh uniform colors,
                        per-vertex colors, or None for individual meshes in mixed rendering.
            mesh_vertex_uvs: Optional UV coordinate arrays for per-vertex texture mapping.
                            Individual elements can be None for non-textured meshes.
            mesh_texture_images: Optional texture image arrays. Individual elements
                                can be None for non-textured meshes.

        Raises:
            ValueError: If validation fails
        """
        # Validate and take mesh vertex colors. Generate mesh vertex colors if not provided.
        # Mesh vertex colors are not used if texture images are provided.
        # Generate default colors for all meshes
        color_indices = np.linspace(0, 0.85, self.num_meshes_per_frame)
        rainbow_cmap = plt.cm.rainbow_r
        self.mesh_colors = rainbow_cmap(color_indices)[:, :3]
        if mesh_colors is not None:
            # Standard color validation for non-mixed rendering
            _validate_mesh_colors(mesh_colors, self.num_meshes_per_frame)
            self.mesh_colors = _normalize_mesh_colors(mesh_colors, self.mesh_colors)

        # If texture images are provided, validate UV coordinates and texture images.
        if mesh_texture_images is not None:
            if (
                mesh_vertex_uvs is None
            ):  # When texture images are provided, UV coordinates must also be provided.
                raise ValueError(
                    "When texture images are provided, Pervertex UV coordinates must also be provided."
                )
            if len(mesh_vertex_uvs) != len(
                mesh_texture_images
            ):  # UV coordinates and texture images must match.
                raise ValueError(
                    "The length of UV coordinates and texture images must match."
                )
            if (
                len(mesh_vertex_uvs) != self.num_meshes_per_frame
            ):  # UV coordinates and meshes must match.
                raise ValueError(
                    "The length of UV coordinates and meshes must match the length of meshes."
                )
            for i, (uv, tex) in enumerate(zip(mesh_vertex_uvs, mesh_texture_images)):
                if (uv is None) ^ (
                    tex is None
                ):  # UV coordinates and texture images must be paired.
                    raise ValueError(
                        "UV coordinates and texture images must be paired. The corresponding elements should both or neither be None."
                    )
                elif (
                    uv is not None and tex is not None
                ):  # When UV coordinates are provided, it must be per-vertex UV, i.e. with a shape of (V x 2).
                    if (
                        uv.shape[0] != self.meshes[0][i].vertices.shape[0]
                        or uv.shape[1] != 2
                    ):
                        raise ValueError(
                            "UV coordinates must be 2D and have the same length as the number of vertices in the corresponding mesh."
                        )
                    # Assert the tex image is in RGB format with shape (H, W, 3) and uint8 values [0, 255].
                    if (
                        len(tex.shape) != 3
                        or tex.shape[2] != 3
                        or tex.dtype != np.uint8
                    ):
                        raise ValueError(
                            "Texture image must be in RGB format with shape (H, W, 3) and uint8 values [0, 255]."
                        )
            self.mesh_vertex_uvs = mesh_vertex_uvs
            self.mesh_texture_images = mesh_texture_images

    def _validate_input_pair_are_list_of_lists(
        self,
        input_data: list[Any],
        data_type: str,
        paired_data: list[Any],
        paired_data_type: str,
    ) -> tuple[list[Any], list[Any], int]:
        """
        Validate the input data pairs are in list of lists format and the sizes match.

        Args:
            input_data: The input data to validate
            data_type: Description of the data type for error messages
            paired_data: The paired data to validate
            paired_data_type: Description of the paired data type for error messages

        Returns:
            tuple: (normalized_data, items_per_frame)

        Raises:
            ValueError: If validation fails
        """
        for frame_idx, paired_frame_data in enumerate(paired_data):
            # Validate paired data is list of lists format.
            if not isinstance(paired_frame_data, list):
                raise ValueError(
                    f"{paired_data_type} must be a list of lists, where each inner list represents a frame of {paired_data_type.lower()}."
                )
            # Validate paired data and input data have the same number of items per frame.
            if len(paired_frame_data) != len(input_data[frame_idx]):
                raise ValueError(
                    f"{data_type} and {paired_data_type} must have the same number of items per frame. "
                    f"In frame {frame_idx}: got {len(paired_frame_data)} {paired_data_type.lower()} "
                    f"and {len(input_data[frame_idx])} {data_type.lower()}."
                )

        # Validate frame count consistency.
        if len(input_data) != len(paired_data):
            raise ValueError(
                f"{data_type} and {paired_data_type} must have the same number of frames. "
                f"Got {len(input_data)} {data_type.lower()} frames and {len(paired_data)} {paired_data_type.lower() if paired_data_type else 'unknown'} frames."
            )

        return input_data, paired_data, len(input_data[0])

    def _prepare_rendering(self, height: int = 800, width: int = 800) -> None:
        """
        Prepare the ScenePic scene for rendering.

        Sets up the 3D canvas, camera positioning, and all scene assets for rendering.
        This is called internally before displaying or saving the scene.

        Args:
            height: Height of the rendered canvas in pixels. Default is 800.
            width: Width of the rendered canvas in pixels. Default is 800.
        """
        canvas3d = self.scene.create_canvas_3d(
            height=height, width=width, shading=self.shading
        )

        self._setup_camera_if_needed()
        layer_settings = self._create_layer_settings()
        canvas3d.set_layer_settings(layer_settings)

        num_frames = self._setup_scene_assets()
        self._create_animation_frames(canvas3d, num_frames)

    def _setup_camera_if_needed(self) -> None:
        """Set up automatic camera positioning if not manually configured."""
        if not self.camera_is_manually_set:
            scene_center = (self.scene_bbox[0] + self.scene_bbox[1]) / 2.0
            scene_scale = np.max(np.abs(self.scene_bbox[1] - self.scene_bbox[0]))
            self.camera = sp.Camera(
                center=scene_center + np.array([0.0, 0.0, 2 * scene_scale]),
                look_at=scene_center + 1e-8,  # Avoid numerical issues
                fov_y_degrees=30.0,
                aspect_ratio=1.0,
                far_crop_distance=20.0,
                near_crop_distance=0.01,
            )

    def _create_layer_settings(self) -> dict[str, dict[str, Any]]:
        """Create layer settings for asset opacity control."""
        layer_settings: dict[str, dict[str, Any]] = {}

        # Configure mesh layer settings
        if self.mesh_names:
            for i, mesh_name in enumerate(self.mesh_names):
                opacity = self.mesh_opacity[i]
                layer_settings[mesh_name] = {
                    "filled": False if opacity == 0 else True,
                    "wireframe": False,
                    "opacity": 1.0 if opacity == 0 else opacity,
                }

        # Configure point cloud layer settings
        if self.point_cloud_names:
            for i, pc_name in enumerate(self.point_cloud_names):
                opacity = self.point_cloud_opacity[i]
                layer_settings[pc_name] = {
                    "filled": False if opacity == 0 else True,
                    "wireframe": False,
                    "opacity": 1.0 if opacity == 0 else opacity,
                }
                if self.point_labels:
                    # Point label opacity is the minimum of label opacity and point cloud opacity
                    effective_label_opacity = min(
                        self.point_label_opacity[i], self.point_cloud_opacity[i]
                    )
                    layer_settings[f"Labels_{pc_name}"] = {
                        "filled": False if effective_label_opacity == 0 else True,
                        "wireframe": False,
                        "opacity": 1.0
                        if effective_label_opacity == 0
                        else effective_label_opacity,
                    }

        # Configure line set layer settings
        if self.line_set_names:
            for i, line_name in enumerate(self.line_set_names):
                opacity = self.line_set_opacity[i]
                layer_settings[line_name] = {
                    "filled": False if opacity == 0 else True,
                    "wireframe": False,
                    "opacity": 1.0 if opacity == 0 else opacity,
                }

        # Configure coordinate frame layer settings
        if self.coordinate_frame_names:
            for i, frame_name in enumerate(self.coordinate_frame_names):
                opacity = self.coordinate_frame_opacity[i]
                layer_settings[frame_name] = {
                    "filled": False if opacity == 0 else True,
                    "wireframe": False,
                    "opacity": 1.0 if opacity == 0 else opacity,
                }

        return layer_settings

    def _setup_scene_assets(self) -> int:
        """Set up all scene assets and return the number of animation frames."""
        num_frames = 0

        if self.mesh_names:
            num_frames = self._add_meshes_to_scene()

        if self.point_cloud_names:
            num_frames = self._add_point_clouds_to_scene(num_frames)

        if self.line_set_names:
            num_frames = self._add_line_sets_to_scene(num_frames)

        return num_frames

    def _create_animation_frames(self, canvas3d: sp.Canvas3D, num_frames: int) -> None:
        """Create all animation frames with their respective assets."""
        scene_center = (self.scene_bbox[0] + self.scene_bbox[1]) / 2.0
        for frame_index in range(num_frames):
            frame = canvas3d.create_frame(
                frame_id=f"{frame_index}",
                camera=self.camera,
                # We put the focus_point at the scene_center so that when dragging the mouse
                # on the visualization, the scene rotates around its center rather than
                # around the camera center. This gives better experience when examine the 3D
                # objects.
                focus_point=sp.FocusPoint(scene_center, np.zeros(3)),
            )

            self._add_frame_assets(frame, frame_index)

    def _add_frame_assets(self, frame: sp.Frame3D, frame_index: int) -> None:
        """Add all assets to a specific animation frame."""
        if self.mesh_names:
            self._update_meshes_in_frame(frame, frame_index)

        if self.point_cloud_names:
            self._update_point_clouds_in_frame(frame, frame_index)
        # The scene.update_** methods reduce the size of the scene compared to scene.create_** methods.
        # But updating lines and coordinate frames seem not to be supported by scenepic.
        if self.line_set_names:
            self._add_line_sets_to_frame(frame, frame_index)

        if self.coordinate_frame_names:
            self._add_coordinate_frames_to_frame(frame, frame_index)

    def _update_meshes_in_frame(self, frame: sp.Frame3D, frame_index: int) -> None:
        """Update meshes in a specific frame."""
        updated_meshes = []
        frame_meshes = self.meshes[frame_index]
        for mesh_index, mesh in enumerate(frame_meshes):
            updated_meshes.append(
                self.scene.update_mesh_positions(
                    self.mesh_names[mesh_index], mesh.vertices
                )
            )
        frame.add_meshes(updated_meshes)

    def _update_point_clouds_in_frame(
        self, frame: sp.Frame3D, frame_index: int
    ) -> None:
        """Update point clouds and labels to a specific frame."""
        updated_point_clouds = []
        labels = []
        label_positions = []

        frame_point_clouds = self.point_clouds[frame_index]
        for pc_index, point_cloud in enumerate(frame_point_clouds):
            updated_point_clouds.append(
                self.scene.update_instanced_mesh(
                    self.point_cloud_names[pc_index], point_cloud
                )
            )

            if self.point_labels:
                self._create_point_labels(labels, pc_index, point_cloud)
                label_positions.append(point_cloud)

        frame.add_meshes(updated_point_clouds)

        if self.point_labels and labels:
            self._add_labels_to_frame(frame, labels, label_positions)

    def _create_point_labels(
        self,
        labels: list[sp.Label],
        pc_index: int,
        point_cloud: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ) -> None:
        """Create labels for points in a point cloud."""
        for point_id in range(point_cloud.shape[0]):
            label = self.scene.create_label(
                text=self.point_labels[pc_index][point_id],
                color=self.label_color,
                layer_id=f"Labels_{self.point_cloud_names[pc_index]}",
                size_in_pixels=self.label_size_in_pixel,
                offset_distance=0.0,
            )
            labels.append(label)

    def _add_labels_to_frame(
        self,
        frame: sp.Frame3D,
        labels: list[sp.Label],
        label_positions: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]],
    ) -> None:
        """Add all labels to a frame at their respective positions."""
        all_positions = np.vstack(label_positions)
        for label, position in zip(labels, all_positions):
            frame.add_label(
                label=label,
                position=position + self.point_label_offset,
            )

    def _add_line_sets_to_frame(self, frame: sp.Frame3D, frame_index: int) -> None:
        """Add line sets to a specific frame."""
        updated_line_sets = []
        frame_start_points = self.start_points[frame_index]
        frame_end_points = self.end_points[frame_index]

        for line_set_index in range(self.num_line_sets_per_frame):
            start_points = frame_start_points[line_set_index]
            end_points = frame_end_points[line_set_index]
            line_set_mesh = self.scene.create_mesh(
                mesh_id=self.line_set_names[line_set_index],
                layer_id=self.line_set_names[line_set_index],
            )

            self._add_lines_to_mesh(
                line_set_mesh, start_points, end_points, line_set_index
            )
            updated_line_sets.append(line_set_mesh)

        frame.add_meshes(updated_line_sets)

    def _add_lines_to_mesh(
        self,
        mesh: sp.Mesh,
        start_points: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        end_points: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        line_set_index: int,
    ) -> None:
        """Add lines to a mesh based on line type."""
        line_set_color = self.line_set_colors[line_set_index]

        if self.line_type == "line":
            mesh.add_lines(
                start_points=start_points,
                end_points=end_points,
                color=line_set_color,
            )
        elif self.line_type == "thickline":
            for start_point, end_point in zip(start_points, end_points):
                mesh.add_thickline(
                    color=line_set_color,
                    start_point=start_point,
                    end_point=end_point,
                    start_thickness=self.line_start_thickness,
                    end_thickness=0.001,
                )

    def _add_coordinate_frames_to_frame(
        self, frame: sp.Frame3D, frame_index: int
    ) -> None:
        """Add coordinate frames to a specific frame."""
        updated_coordinate_frames = []
        frame_origins = self.coordinate_frame_origins[frame_index]
        frame_orientations = self.coordinate_frame_orientations[frame_index]

        for coord_frame_index in range(self.num_coordinate_frames_per_frame):
            origins = frame_origins[coord_frame_index]
            orientations = frame_orientations[coord_frame_index]
            coordinate_frame_mesh = self.scene.create_mesh(
                mesh_id=self.coordinate_frame_names[coord_frame_index],
                layer_id=self.coordinate_frame_names[coord_frame_index],
            )

            self._add_coordinate_axes_to_mesh(
                coordinate_frame_mesh, origins, orientations
            )
            updated_coordinate_frames.append(coordinate_frame_mesh)

        frame.add_meshes(updated_coordinate_frames)

    def _add_coordinate_axes_to_mesh(
        self,
        mesh: sp.Mesh,
        origins: np.ndarray[tuple[int, int], np.dtype[np.float32]],
        orientations: np.ndarray[tuple[int, int], np.dtype[np.float32]],
    ) -> None:
        """Add coordinate axes to a mesh using transform matrices."""
        # Create transform matrices
        transform_matrices = np.zeros((orientations.shape[0], 4, 4), dtype=np.float32)
        transform_matrices[..., :3, :3] = orientations
        transform_matrices[..., :3, 3] = origins
        transform_matrices[..., 3, 3] = 1.0

        # Add coordinate axes using ScenePic's built-in function
        for transform_matrix in transform_matrices:
            mesh.add_coordinate_axes(
                length=self.frame_size,
                thickness=self.frame_size * 0.1,
                transform=transform_matrix,
            )

    def _add_meshes_to_scene(self) -> int:
        """
        Add meshes to the scene and set up their properties.

        Supports mixed rendering: some meshes can use textures while others use vertex colors.

        Returns:
            int: The number of frames
        """
        # Create texture images for meshes that have textures.
        texture_ids: list[str | None] = [None for _ in range(self.num_meshes_per_frame)]
        # This is to prevent pyre from complaining about "Undefined attribute [16]:
        # Optional type has no attribute `__getitem__`" if otherwise we use
        # enumerate(self.mesh_texture_images) below. Assign to local variable for type narrowing
        mesh_texture_images = self.mesh_texture_images
        mesh_vertex_uvs = self.mesh_vertex_uvs
        if mesh_texture_images is not None:
            for mesh_index, texture_image in enumerate(mesh_texture_images):
                if texture_image is not None:
                    texture_id = f"texture_{mesh_index}"
                    scene_image = self.scene.create_image(image_id=texture_id)
                    scene_image.from_numpy(texture_image)
                    texture_ids[mesh_index] = texture_id

        # Create scene meshes with per-mesh vertex colors.
        scene_meshes = []
        for mesh_index, mesh_name in enumerate(self.mesh_names):
            if (
                self.mesh_texture_images is not None
                and self.mesh_texture_images[mesh_index] is not None
            ):  # If the mesh has texture, create a mesh with texture
                scene_mesh = self.scene.create_mesh(
                    mesh_id=mesh_name,
                    layer_id=mesh_name,
                    texture_id=texture_ids[mesh_index],
                )
                # Fix: use local mesh_vertex_uvs variable to avoid Optional type error
                scene_mesh.add_mesh_without_normals(
                    vertices=self.meshes[0][mesh_index].vertices,
                    triangles=self.meshes[0][mesh_index].faces,
                    uvs=mesh_vertex_uvs[mesh_index]
                    if mesh_vertex_uvs is not None
                    else None,
                )
            else:  # Otherwise, create a mesh with vertex colors
                scene_mesh = self.scene.create_mesh(
                    mesh_id=mesh_name, layer_id=mesh_name
                )
                per_vertex_color = _repeat_vertex_color(
                    self.mesh_colors[mesh_index],
                    self.meshes[0][mesh_index].vertices.shape[0],
                    mesh_index,
                )
                scene_mesh.add_mesh_without_normals(
                    vertices=self.meshes[0][mesh_index].vertices,
                    triangles=self.meshes[0][mesh_index].faces,
                    colors=per_vertex_color,
                )
            scene_meshes.append(scene_mesh)

        num_frames = len(self.meshes)
        return num_frames

    def _add_point_clouds_to_scene(self, num_frames: int) -> int:
        """
        Add point clouds to the scene and set up their properties.

        Args:
            num_frames: The current number of frames

        Returns:
            int: The validated number of frames
        """
        scene_point_clouds = [
            self.scene.create_mesh(mesh_id=point_cloud_name, layer_id=point_cloud_name)
            for point_cloud_name in self.point_cloud_names
        ]
        if num_frames == 0:
            num_frames = len(self.point_clouds)
        else:
            if num_frames != len(self.point_clouds):
                raise ValueError(
                    f"Invalid point clouds. Expected a list of point clouds, where the length matches the number of mesh frames or other 3D input assets. Got {len(self.point_clouds)} point clouds for {num_frames} frames."
                )
        for point_cloud_index, point_cloud in enumerate(self.point_clouds[0]):
            per_point_color = _repeat_vertex_color(
                self.point_cloud_colors[point_cloud_index],
                point_cloud.shape[0],
                point_cloud_index,
            )
            scene_point_clouds[point_cloud_index].add_sphere(
                sp.Colors.White, transform=sp.Transforms.Scale(self.point_size)
            )
            scene_point_clouds[point_cloud_index].enable_instancing(
                point_cloud, colors=per_point_color
            )
        return num_frames

    def _add_line_sets_to_scene(self, num_frames: int) -> int:
        """
        Add line sets to the scene and validate the number of frames.

        Args:
            num_frames: The current number of frames

        Returns:
            int: The validated number of frames
        """
        if num_frames == 0:
            num_frames = len(self.start_points)
        else:
            if num_frames != len(self.start_points):
                raise ValueError(
                    f"Invalid line sets. Expected a list of line sets, where the length matches the number of mesh frames or other 3D input assets. Got {len(self.start_points)} line set frames for {num_frames} frames."
                )
        return num_frames

    def _generate_html_string(self) -> str:
        """
        Generate an HTML string containing the complete ScenePic visualization.

        Creates a standalone HTML document with embedded JavaScript that contains
        the full 3D scene data and ScenePic viewer. The resulting HTML can be
        saved to a file or displayed in web browsers.

        Returns:
            str: Complete HTML document string with embedded ScenePic scene

        Note:
            - The HTML includes the full ScenePic JavaScript library
            - Scene data is embedded directly in the HTML for standalone viewing
            - The generated HTML is self-contained and doesn't require external resources
            - Compatible with modern web browsers supporting WebGL
        """
        SP_LIB = sp.js_lib_src()
        SP_SCRIPT = self.scene.get_script().replace(
            "window.onload = function()", "function scenepic_foo()"
        )
        HTML_string = (
            "<!DOCTYPE html>"
            '<html lang="en">'
            "<head>"
            '<meta charset="utf-8">'
            "<title>ScenePic </title>"
            f"<script>{SP_LIB}</script>"
            f"<script>{SP_SCRIPT} scenepic_foo();</script>"
            "</head>"
            '<body onload="scenepic_foo()">  </body>'
            "</html>"
        )
        return HTML_string

    def _generate_html_string2(self) -> str:
        """
        Generate an HTML string containing the complete ScenePic visualization.

        Creates a standalone HTML document with embedded JavaScript that contains
        the full 3D scene data and ScenePic viewer. The resulting HTML can be
        saved to a file or displayed in web browsers.

        Returns:
            str: Complete HTML document string with embedded ScenePic scene

        Note:
            - The HTML includes the full ScenePic JavaScript library
            - Scene data is embedded directly in the HTML for standalone viewing
            - The generated HTML is self-contained and doesn't require external resources
            - Compatible with modern web browsers supporting WebGL
        """
        SP_LIB = sp.js_lib_src()
        SP_SCRIPT = self.scene.get_script().replace(
            "window.onload = function()", "function scenepic_foo()"
        )
        HTML_div = (
            "<div id='scene-div'></div>"
            f"<script>{SP_LIB}</script>"
            f"<script>{SP_SCRIPT}</script>"
            "<script>scenepic_foo();</script>"
        )
        return HTML_div


def _validate_mesh_colors(
    colors: list[MeshMonoColors | MeshPervertexColors | None], num_meshes: int
) -> None:
    """
    Validate mesh colors configuration.

    Args:
        colors: The mesh colors array to validate
        num_meshes: Expected number of meshes

    Raises:
        ValueError: If the mesh colors are invalid

    Validates:
        - Shape is (num_meshes, (v x ) 4)
        - Data type is supported (uint8, int32, int64, float32, float64)
        - Value ranges are appropriate for the data type
    """
    if len(colors) != num_meshes:
        raise ValueError(
            f"Invalid mesh colors. Expected a list of mesh colors, where the length matches the number of meshes. Got {len(colors)} colors for {num_meshes} meshes."
        )

    # Check data type and value ranges
    for color in colors:
        if color is not None:
            if (
                color.dtype == np.uint8
                or color.dtype == np.int32
                or color.dtype == np.int64
            ):
                # Integer values should be in range [0, 255]
                if not np.all((color >= 0) & (color <= 255)):
                    raise ValueError(
                        "Invalid mesh colors. Integer color values are expected in range [0, 255]."
                    )
            elif color.dtype in [np.float32, np.float64]:
                # Float values should be in range [0.0, 1.0]
                if not np.all((color >= 0.0) & (color <= 1.0)):
                    raise ValueError(
                        "Invalid mesh colors. Float color values are expected in range [0.0, 1.0]."
                    )
            else:
                raise ValueError(
                    "Invalid mesh colors. Unsupported data type. Expected uint8, int32, int64, float32, or float64."
                )


def _normalize_mesh_colors(
    colors: list[MeshPervertexColors | MeshMonoColors | None],
    default_colors: list[MeshPervertexColors | MeshMonoColors],
) -> list[MeshPervertexColors | MeshMonoColors]:
    """
    Normalize mesh colors to uint8 values in range [0, 255].

    Args:
        colors: Input mesh colors in any supported format

    Returns:
        Normalized colors as a list of uint8 array in range [0, 255]
    """
    normalized_colors = []

    for i, color in enumerate(colors):
        if color is not None:
            if color.dtype in [np.uint8, np.int32, np.int64]:
                # Convert integer colors to float32 in range [0.0, 1.0]
                normalized_colors.append(color.astype(np.float32) / 255.0)
            elif color.dtype in [np.float32, np.float64]:
                # Already normalized float colors, ensure they are float32
                normalized_colors.append(color.astype(np.float32))
            else:
                raise ValueError(
                    "Invalid mesh colors. Unsupported data type. Expected uint8, int32, int64, float32, or float64."
                )
        else:
            normalized_colors.append(default_colors[i])

    return normalized_colors


def _repeat_vertex_color(
    mesh_color: MeshPervertexColors | MeshMonoColors, num_vertices: int, mesh_index: int
) -> MeshPervertexColors:
    """
    Repeat mesh color to match the number of vertices if needed.

    Args:
        mesh_color: The mesh color array
        num_vertices: Number of vertices in the mesh
        mesh_index: Index of the mesh for error reporting

    Returns:
        MeshMeshPervertexColors: Color array matching the number of vertices

    Raises:
        ValueError: If the mesh color format is invalid
    """
    if mesh_color.ndim == 1:
        # Single color (3,) -> expand to per-vertex colors (num_vertices, 3)
        mesh_color = np.repeat(mesh_color[np.newaxis, ...], num_vertices, axis=0)
    elif mesh_color.ndim == 2 and mesh_color.shape[0] == 1:
        # Single color (1, 3) -> expand to per-vertex colors (num_vertices, 3)
        mesh_color = np.repeat(mesh_color, num_vertices, axis=0)
    elif mesh_color.ndim == 2 and mesh_color.shape[0] == num_vertices:
        # Already per-vertex colors (num_vertices, 3) -> no change needed
        pass
    else:
        raise ValueError(
            f"Invalid mesh color. The shape for {mesh_index} mesh color has the shape of {mesh_color.shape}, which is neither of the following accepted format: \n (3) or (1, 3): a uniform color for the whole mesh. \n (v, 3): per-vertex color for the mesh."
        )

    return mesh_color


def visualize_posed_mhr_model(
    pose_parameters: np.ndarray,
    pca_coeffs: np.ndarray | None = None,
    face_expr_coeffs: np.ndarray | None = None,
    affected_joints_names: list[str] = [],
    output: str = "show",
) -> str | None:
    """Visualize a posed MHR model with joints, skeleton, and coordinate frames.

    Args:
        pose_parameters: Pose parameters as a float32 array of shape (1, N).
        pca_coeffs: Identity blendshape coefficients. Default zeros of shape (1, num_pca_comp).
        face_expr_coeffs: Facial expression coefficients. Default zeros of shape (1, 72).
        affected_joints_names: Names of joints affected by the current pose parameter.
        output: 'show' to display in notebook, 'html' to return HTML string.
    """
    if _mhr_model is None:
        raise RuntimeError("MHR model not loaded. Call load_model() first.")

    num_pca_comp = _mhr_model.get_num_identity_blendshapes()
    if pca_coeffs is None:
        pca_coeffs = np.zeros((1, num_pca_comp), dtype=np.float32)
    if face_expr_coeffs is None:
        face_expr_coeffs = np.zeros((1, 72), dtype=np.float32)

    params = torch.from_numpy(pose_parameters)
    identity_coeffs = torch.from_numpy(pca_coeffs)
    face_expr_coeffs_t = torch.from_numpy(face_expr_coeffs)

    body_vertices, skel_state = _mhr_model(
        identity_coeffs=identity_coeffs,
        model_parameters=params,
        face_expr_coeffs=face_expr_coeffs_t,
    )
    body_vertices = body_vertices.numpy()[0] / 100.0
    body_faces = _mhr_model.character_torch.mesh.faces.cpu().numpy()
    body_mesh = trimesh.Trimesh(body_vertices, body_faces, process=False)

    skel_state = skel_state.numpy()[0]
    joint_locations = skel_state[..., :3] / 100.0
    joint_names = _mhr_model.get_joint_names()

    joint_parents = _mhr_model.character_torch.skeleton.joint_parents
    joint_parents = np.clip(np.array(joint_parents), 0, np.inf).astype(np.int32)
    parent_joint_locations = joint_locations[joint_parents]

    kinematic_joints = joint_locations[np.unique(joint_parents)]
    kinematic_joints_names = [joint_names[i] for i in np.unique(joint_parents)]

    joint_orientations = skel_state[..., 3:7]
    joint_orientations = R.Rotation.from_quat(joint_orientations).as_matrix()
    kinematic_joint_orientations = joint_orientations[np.unique(joint_parents)]

    if len(affected_joints_names) == 1:
        affected_joints_names += affected_joints_names
    affected_joints_indices = [
        joint_names.index(joint_name) for joint_name in affected_joints_names
    ]
    if affected_joints_names:
        affected_joints_locations = joint_locations[affected_joints_indices]

    visualizer = ScenepicVisualization()
    visualizer.add_meshes(
        meshes=[[body_mesh]],
        mesh_names=["Body surface"],
        mesh_colors=[sp.Colors.Blue],
        mesh_opacity=[0.6],
    )
    if affected_joints_names:
        visualizer.add_point_clouds(
            point_clouds=[
                [joint_locations, kinematic_joints, affected_joints_locations]
            ],
            point_cloud_names=["Joints", "Kinematic joints", "Affected joints"],
            point_cloud_opacity=[0.0, 0.5, 1.0],
            point_size=0.02,
            point_labels=[joint_names, kinematic_joints_names, affected_joints_names],
            point_label_opacity=[0.0, 0.0, 1.0],
            label_size_in_pixel=30,
            label_offset=0.01,
            point_cloud_colors=[sp.Colors.Red, sp.Colors.Blue, sp.Colors.Purple],
        )
    else:
        visualizer.add_point_clouds(
            point_clouds=[[joint_locations, kinematic_joints]],
            point_cloud_names=["Joints", "Kinematic joints"],
            point_cloud_opacity=[0.0, 1.0],
            point_size=0.02,
            point_labels=[joint_names, kinematic_joints_names],
            label_size_in_pixel=30,
            label_offset=0.01,
            point_cloud_colors=[sp.Colors.Red, sp.Colors.Blue],
        )

    visualizer.add_line_sets(
        start_points=[[parent_joint_locations]],
        end_points=[[joint_locations]],
        line_set_colors=[sp.Colors.Green],
        line_set_names=["Skeleton"],
        line_start_thickness=0.015,
    )
    visualizer.add_coordinate_frames(
        coordinate_frame_origins=[[joint_locations, kinematic_joints]],
        coordinate_frame_orientations=[
            [joint_orientations, kinematic_joint_orientations]
        ],
        coordinate_frame_opacity=[0.0, 0.0],
        coordinate_frame_names=["Joints local frames", "Kinematic joint local frames"],
        frame_size=0.04,
    )

    if output == "html":
        return visualizer.render_to_html_string()
    else:
        visualizer.show()


def visualize_blendshape_space(
    pc_indices: list[int],
    model_params: torch.Tensor,
    mesh_faces: np.ndarray,
    num_frames: int = 48,
    num_pca_comp: int = 45,
    face_expr_dim: int = 72,
    is_expression: bool = False,
    face_mask: np.ndarray | None = None,
) -> list[list[trimesh.Trimesh]]:
    """Create an animation of blendshape deformation along specified principal components.

    Args:
        pc_indices: Indices of principal components to animate.
        model_params: Base model parameters (torch Tensor of shape (1, N)).
        mesh_faces: Face indices array for the mesh.
        num_frames: Total number of animation frames. Default 48.
        num_pca_comp: Number of identity blendshape components. Default 45.
        face_expr_dim: Number of facial expression dimensions. Default 72.
        is_expression: If True, animate expression components (range -1..1); otherwise identity (range -3..3).
        face_mask: Optional boolean mask for face triangles to isolate body parts.

    Returns:
        List of lists of trimesh.Trimesh objects, one list per frame.
    """
    if _mhr_model is None:
        raise RuntimeError("MHR model not loaded. Call load_model() first.")

    num_pc = len(pc_indices)
    left_most_frame_idx = num_frames // 4
    middle_frame_idx = num_frames // 2
    right_most_frame_idx = num_frames // 4 * 3
    meshes = [num_pc * [trimesh.Trimesh()] for i in range(num_frames)]

    start = -3.0
    end = 3.0
    if is_expression:
        start = -1.0
        end = 1.0

    for i_th_pc, pc_idx in enumerate(pc_indices):
        identity_coeffs = torch.zeros(1, num_pca_comp)
        face_expr_coeffs = torch.zeros(1, face_expr_dim)

        for i_th_frame, sigma in enumerate(np.linspace(start, end, num_frames // 2 + 1)):
            if is_expression:
                face_expr_coeffs[0, pc_idx] = sigma
            else:
                identity_coeffs[0, pc_idx] = sigma

            body_vertices, _ = _mhr_model(
                identity_coeffs=identity_coeffs,
                model_parameters=model_params,
                face_expr_coeffs=face_expr_coeffs,
            )
            body_vertices = body_vertices.numpy()[0] / 100.0

            mesh = trimesh.Trimesh(body_vertices, mesh_faces, process=False)
            if face_mask is not None:
                mesh.update_faces(face_mask)
                mesh.remove_unreferenced_vertices()
                mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)

            meshes[left_most_frame_idx + i_th_frame][i_th_pc] = mesh

        for i_th_frame in range(0, left_most_frame_idx):
            meshes[i_th_frame][i_th_pc] = meshes[middle_frame_idx - i_th_frame][i_th_pc]
        for i_th_frame in range(right_most_frame_idx + 1, num_frames):
            meshes[i_th_frame][i_th_pc] = meshes[
                2 * right_most_frame_idx - i_th_frame
            ][i_th_pc]

    return meshes


def get_head_hand_mask() -> tuple[np.ndarray, np.ndarray]:
    """Get soft masks for head/neck and hand vertices based on LBS weights.

    Returns:
        Tuple of (head_or_neck_softmask, right_hand_mask), both as numpy arrays
        with per-vertex soft weights.
    """
    if _mhr_model is None:
        raise RuntimeError("MHR model not loaded. Call load_model() first.")

    joint_names = _mhr_model.get_joint_names()
    sw_index, sw_weight = _mhr_model.get_lbsw()
    sw_index = sw_index.numpy()
    sw_weight = sw_weight.numpy()

    hand_joints = {
        side: [
            i
            for i, n in enumerate(joint_names)
            if (
                n.startswith(f"{side}_thumb")
                or n.startswith(f"{side}_index")
                or n.startswith(f"{side}_middle")
                or n.startswith(f"{side}_ring")
                or n.startswith(f"{side}_pinky")
                or n.startswith(f"{side}_wrist")
            )
        ]
        for side in ("l", "r")
    }

    hand_indices = {
        side: reduce(np.logical_or, [sw_index == i for i in joints])
        for side, joints in hand_joints.items()
    }
    hand_softmasks_orig = {
        side: np.multiply(hand_index, sw_weight).sum(axis=1)
        for side, hand_index in hand_indices.items()
    }
    right_hand_mask = hand_softmasks_orig["r"]

    head_or_neck_joints = [
        i for i, c in enumerate(joint_names) if (c.startswith("c") and not c.startswith("c_spine"))
    ]
    head_or_neck_index = reduce(
        np.logical_or, [sw_index == i for i in head_or_neck_joints]
    )
    head_or_neck_softmask = np.multiply(head_or_neck_index, sw_weight).sum(axis=1)

    return head_or_neck_softmask, right_hand_mask

