import argparse
import json
import os
import sys
import igl
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator
from concurrent.futures import ProcessPoolExecutor, as_completed


GLOBAL_AVERAGE_VERTEX_NORM = 5.215578427705153  # 5.712549 if not adjusting y-midpoint, 5.215578427705153 if adjusting y-midpoint


SPLIT_FILES = {"train": "training", "val": "validation", "test": "testing"}


def get_outer_crown_paths(data_path, max_cases=None):
    """Get outer crown mesh paths of the <root>/<split>/<id>/ layout, listed from the split JSONs"""
    data = []
    for split, split_file in SPLIT_FILES.items():
        ids = [str(case_id) for case_id in load_split(data_path, split_file)]
        if max_cases is not None:
            ids = ids[:max_cases]
        for case_id in ids:
            data.append(os.path.join(data_path, split, case_id, "outer_crown.ply"))
            if not os.path.exists(data[-1]):
                print("File does not exist: ", data[-1])
    return data


def load_split(data_path, split):
    with open(os.path.join(data_path, split + ".json"), "r") as f:
        data = json.load(f)
    return data


def create_output_dir(omages_output_dir, data_prefix="crown"):
    if not os.path.exists(omages_output_dir):
        print("Creating output directory: ", omages_output_dir)
        os.makedirs(omages_output_dir)
    # train_dir
    train_dir = os.path.join(omages_output_dir, f"{data_prefix}_train")
    if not os.path.exists(train_dir):
        print("Creating output directory: ", train_dir)
        os.makedirs(train_dir)
    # val_dir
    val_dir = os.path.join(omages_output_dir, f"{data_prefix}_val")
    if not os.path.exists(val_dir):
        print("Creating output directory: ", val_dir)
        os.makedirs(val_dir)
    # test_dir
    test_dir = os.path.join(omages_output_dir, f"{data_prefix}_test")
    if not os.path.exists(test_dir):
        print("Creating output directory: ", test_dir)
        os.makedirs(test_dir)

    return train_dir, val_dir, test_dir


def map_mesh_to_circle(outer_crown_path):

    # load the mesh
    vertices, facets = igl.read_triangle_mesh(outer_crown_path)
    vertex_normals = igl.per_vertex_normals(vertices, facets)

    # NORMALIZATION
    # Center vertices with their bounding box midpoint in the y-axis
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    y_mid = (y_min + y_max) / 2
    vertices[:, 1] -= y_mid
    vertices = vertices / GLOBAL_AVERAGE_VERTEX_NORM

    # use harmonic parameterization to map the vertices to the unit circle
    boundary = igl.boundary_loop(facets)
    boundary_uv_coordinates = igl.map_vertices_to_circle(vertices, boundary)
    uv_coordinates = igl.harmonic(
        vertices, facets, boundary, boundary_uv_coordinates, k=1
    )
    vertices_uv_coordinates = np.hstack(
        [uv_coordinates, np.zeros((uv_coordinates.shape[0], 1))]
    )

    return vertices_uv_coordinates, vertices, facets, vertex_normals


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def test_and_clamp(x, y, threshold=0.01):
    if -1 - threshold < x < -1 or 1 < x < 1 + threshold:
        x = clamp(x, -1, 1)
    if -1 - threshold < y < -1 or 1 < y < 1 + threshold:
        y = clamp(y, -1, 1)
    return x, y


def interpolate_mesh(
    vertices_uv_coordinates, vertices_xyz_coordinates, image_size=512, method="linear"
):
    u, v = vertices_uv_coordinates[:, 0], vertices_uv_coordinates[:, 1]
    # Triangulation function
    triFn = Triangulation(u, v)

    # Interpolator function
    if method == "linear":
        Interpolator = LinearTriInterpolator
    elif method == "cubic":
        Interpolator = CubicTriInterpolator
    else:
        raise ValueError("Unsupported interpolation method")

    rasterized_mesh = []

    for i in range(3):
        interpolator = Interpolator(triFn, vertices_xyz_coordinates[:, i])

        xCoords = np.linspace(u.min(), u.max(), image_size)
        yCoords = np.linspace(v.min(), v.max(), image_size)
        zCoords = np.zeros([yCoords.shape[0], xCoords.shape[0]])

        # Loop among each cell in the raster extension
        for indexX, x in np.ndenumerate(xCoords):
            for indexY, y in np.ndenumerate(yCoords):
                x, y = test_and_clamp(x, y)
                tempZ = interpolator(x, y)
                # Filtering masked values
                zCoords[indexY, indexX] = tempZ if tempZ == tempZ else np.nan

        rasterized_mesh.append(zCoords)

    return np.stack(rasterized_mesh, axis=-1)


def rasterize_uv_coordinates(
    vertices_uv_coordinates,
    vertices_orig_coordinates,
    vertex_normals=None,
    image_size=512,
    method="linear",
):
    # Example usage
    rasterized_mesh = interpolate_mesh(
        vertices_uv_coordinates,
        vertices_orig_coordinates,
        image_size=image_size,
        method=method,
    )

    rasterized_mesh = np.array(rasterized_mesh)
    rasterized_mesh = np.flipud(rasterized_mesh)

    if vertex_normals is not None:
        # Rasterize normals
        rasterized_normals = interpolate_mesh(
            vertices_uv_coordinates,
            vertex_normals,
            image_size=image_size,
            method=method,
        )
        # Normalize rasterized_normals to unit vectors
        norm = np.linalg.norm(rasterized_normals, axis=-1, keepdims=True)
        norm[norm == 0] = 1  # Avoid division by zero
        rasterized_normals = rasterized_normals / norm

        return rasterized_mesh, rasterized_normals
    else:
        return rasterized_mesh


def fisheye_smooth(uvs, inner_radius=0.6, expansion=1.2, smoothness=1):
    """
    Smooth radial remapping using hermite interpolation.
    """
    xy = uvs[:, :2]
    r = np.linalg.norm(xy, axis=1, keepdims=True)
    
    inner_target = min(inner_radius * expansion, 0.95)
    
    # Smooth transition zone
    t_start = inner_radius - smoothness / 2
    t_end = inner_radius + smoothness / 2
    
    # Smoothstep function
    def smoothstep(x):
        x = np.clip((x - t_start) / (t_end - t_start), 0, 1)
        return x * x * (3 - 2 * x)
    
    # Blend between inner (stretched) and outer (compressed) mappings
    inner_map = r * (inner_target / inner_radius)
    outer_map = inner_target + (r - inner_radius) * ((1 - inner_target) / (1 - inner_radius))
    
    blend = smoothstep(r)
    r_new = inner_map * (1 - blend) + outer_map * blend
    
    scale = r_new / (r + 1e-8)
    xy_warped = xy * scale
    return np.hstack([xy_warped, uvs[:, 2:3]])

def plot_fisheye_warping(vertices_uv_coordinates, facets, output_path=None):
    """
    Plot the fish eye lens warping effect showing both original and warped mesh.
    Includes mesh facet edges for better visualization.
    
    Args:
        vertices_uv_coordinates: UV coordinates of vertices
        facets: Triangulation facets
        output_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original UV coordinates
    u_orig = vertices_uv_coordinates[:, 0]
    v_orig = vertices_uv_coordinates[:, 1]
    
    # Apply fish eye warping
    vertices_warped = fisheye_smooth(vertices_uv_coordinates)
    u_warped = vertices_warped[:, 0]
    v_warped = vertices_warped[:, 1]
    
    # Plot original
    ax = axes[0]
    ax.set_aspect('equal')
    ax.set_title('Original UV Coordinates', fontsize=12, fontweight='bold')
    
    # Draw triangles (mesh edges)
    for face in facets:
        triangle = vertices_uv_coordinates[face]
        # Close the triangle by adding the first point at the end
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    
    # Draw vertices
    ax.scatter(u_orig, v_orig, c='red', s=10, alpha=0.5)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Plot warped
    ax = axes[1]
    ax.set_aspect('equal')
    ax.set_title('Fish Eye Lens Warped (γ=0.6)', fontsize=12, fontweight='bold')
    
    # Draw triangles (mesh edges) for warped
    for face in facets:
        triangle = vertices_warped[face]
        # Close the triangle by adding the first point at the end
        triangle_closed = np.vstack([triangle, triangle[0]])
        ax.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', alpha=0.3, linewidth=0.5)
    
    # Draw vertices
    ax.scatter(u_warped, v_warped, c='red', s=10, alpha=0.5)
    ax.set_xlabel('U')
    ax.set_ylabel('V')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved fish eye warping visualization to {output_path}")
    
    plt.show()


def process_outer_crown_path(i, outer_crown_path, visualize_fisheye=False, vis_output_dir=None):
    
    vertices_uv_coordinates, vertices_orig_coordinates, facets, vertex_normals = (
        map_mesh_to_circle(outer_crown_path)
    )
    
    # Visualize fish eye warping effect (for sanity check only, not every iteration)
    if visualize_fisheye and vis_output_dir is not None:
        vis_path = os.path.join(f"./fisheye_warping_{i}.png")
        plot_fisheye_warping(vertices_uv_coordinates, facets, output_path=vis_path)

    vertices_uv_coordinates_fish_eye = fisheye_smooth(vertices_uv_coordinates)
    
    rasterized_mesh, rasterized_normals = rasterize_uv_coordinates(
        vertices_uv_coordinates_fish_eye, vertices_orig_coordinates, vertex_normals
    )

    # Replace NaN values with 0
    rasterized_mesh = np.where(
        np.isnan(rasterized_mesh),
        0,
        rasterized_mesh
    )
    rasterized_normals = np.where(
        np.isnan(rasterized_normals),
        0,
        rasterized_normals
    )

    return rasterized_mesh, rasterized_normals


def get_output_location(outer_crown_path, train_dir, val_dir, test_dir):
    """Map <root>/<split>/<id>/outer_crown.ply to (case_id, output directory)"""
    case_dir = os.path.dirname(outer_crown_path)
    case_identifier = os.path.basename(case_dir)
    split = os.path.basename(os.path.dirname(case_dir))
    output_dirs = {"train": train_dir, "val": val_dir, "test": test_dir}
    if split not in output_dirs:
        print(f"Unknown split '{split}' for {outer_crown_path}")
        return None, None
    return case_identifier, output_dirs[split]


def process_paths(
    list_of_outer_crown_paths,
    train_dir,
    val_dir,
    test_dir,
    use_parallel=True,
):
    """Process outer crown paths either in parallel or sequentially"""
    not_found_count = 0
    error_count = 0
    already_processed_count = 0

    def is_already_processed(output_dir, case_identifier):
        """Check if the combined file already exists"""
        if output_dir is None or case_identifier is None:
            return False
        combined_file = f"{case_identifier}.npy"
        existing_files = os.listdir(output_dir)
        return combined_file in existing_files

    def save_result(output_dir, case_identifier, rasterized_mesh, rasterized_normals):
        """Save the processed mesh and normals in a single file"""
        data = {"mesh": rasterized_mesh, "normals": rasterized_normals}
        np.save(os.path.join(output_dir, f"{case_identifier}.npy"), data)

    # Pre-filter: only process files that need processing
    paths_to_process = []
    path_info = []

    for i, outer_crown_path in enumerate(list_of_outer_crown_paths):
        if not os.path.exists(outer_crown_path):
            not_found_count += 1
            continue

        case_identifier, output_location = get_output_location(
            outer_crown_path,
            train_dir,
            val_dir,
            test_dir,
        )

        if output_location is None:
            not_found_count += 1
            continue

        if is_already_processed(output_location, case_identifier):
            already_processed_count += 1
            continue

        paths_to_process.append((i, outer_crown_path))
        path_info.append((outer_crown_path, case_identifier, output_location))

    print(f"Total files: {len(list_of_outer_crown_paths)}")
    print(f"Missing files: {not_found_count}")
    print(f"Already processed: {already_processed_count}")
    print(f"Files to process: {len(paths_to_process)}")

    if use_parallel:
        with tqdm(
            total=len(paths_to_process),
            desc="Processing outer crown paths (parallel)",
        ) as pbar:
            with ProcessPoolExecutor() as executor:
                # Create a mapping from future to path info
                future_to_info = {
                    executor.submit(process_outer_crown_path, i, path): (path_info[idx])
                    for idx, (i, path) in enumerate(paths_to_process)
                }

                for future in as_completed(future_to_info):
                    outer_crown_path, case_identifier, output_location = future_to_info[
                        future
                    ]

                    try:
                        rasterized_mesh, rasterized_normals = future.result()
                        save_result(
                            output_location,
                            case_identifier,
                            rasterized_mesh,
                            rasterized_normals,
                        )
                    except Exception as e:
                        print(f"Error processing {outer_crown_path}: {e}")
                        error_count += 1

                    pbar.update(1)
    else:
        # Sequential processing
        with tqdm(
            total=len(paths_to_process),
            desc="Processing outer crown paths (sequential)",
        ) as pbar:
            for idx, (i, outer_crown_path) in enumerate(paths_to_process):
                outer_crown_path, case_identifier, output_location = path_info[idx]

                try:
                    rasterized_mesh, rasterized_normals = process_outer_crown_path(
                        i, outer_crown_path
                    )
                    save_result(
                        output_location,
                        case_identifier,
                        rasterized_mesh,
                        rasterized_normals,
                    )
                except Exception as e:
                    print(f"Error processing {outer_crown_path}: {e}")
                    error_count += 1

                pbar.update(1)

    return not_found_count, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Create crown omages from <data-path>/{train,val,test}/<id>/outer_crown.ply"
    )
    parser.add_argument("--data-path", required=True, help="Dataset root containing train/val/test and the split JSONs")
    parser.add_argument("--output-dir", default="./data/omages", help="Output directory for the omages")
    parser.add_argument("--no-parallel", action="store_true", help="Process sequentially (debugging)")
    parser.add_argument("--max-cases", type=int, default=None, help="Max cases per split (smoke tests)")
    args = parser.parse_args()

    data_path = args.data_path
    list_of_outer_crown_paths = get_outer_crown_paths(data_path, args.max_cases)

    print(f"Data path: {data_path}")
    print(f"Number of outer crown paths: {len(list_of_outer_crown_paths)}")

    train_dir, val_dir, test_dir = create_output_dir(
        args.output_dir, data_prefix="crown"
    )

    not_found_count, error_count = process_paths(
        list_of_outer_crown_paths=list_of_outer_crown_paths,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        use_parallel=not args.no_parallel,
    )

    print("Not found count: ", not_found_count)
    print("Error count: ", error_count)
    if error_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
