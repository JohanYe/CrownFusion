import os
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
import fpsample
from tqdm import tqdm


class BasePointCloudDataset(Dataset):
    """Base class with common functionality for point cloud datasets."""

    GLOBAL_AVERAGE_VERTEX_NORM = 5.215578427705153

    def _load_gt_samples(self, data_file="outer_crown.ply", num_points=2048):
        """Load ground truth point clouds for validation/testing."""
        gt_samples = {}

        for path in self.npy_files:
            basename = os.path.basename(path)
            gt_path = os.path.join(self.orig_data_path, self.split, basename.split('.')[0], data_file)

            assert os.path.exists(gt_path), f"GT file not found: {gt_path}"
            gt = o3d.io.read_point_cloud(gt_path)
            gt = np.asarray(gt.points)
            assert gt.shape[0] >= num_points, f"GT point cloud has only {gt.shape[0]} points (need {num_points}): {gt_path}"

            y_min, y_max = np.min(gt[:, 1]), np.max(gt[:, 1])
            y_mid = (y_min + y_max) / 2
            gt[:, 1] -= y_mid
            gt = gt / self.GLOBAL_AVERAGE_VERTEX_NORM

            gt_idx = fpsample.bucket_fps_kdline_sampling(gt, num_points, h=5)
            gt_samples[basename[:-4]] = torch.Tensor(gt[gt_idx])

        return gt_samples


class Only_Omages(BasePointCloudDataset):
    def __init__(
        self,
        root_dir,
        orig_data_path,
        data="",
        split=None,
        transform=None,
        debug=False,
    ):
        assert data in ["crown"]
        assert split in ["train", "val", "test"]
        assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist"

        split = split.lower()
        self.root_dir = root_dir
        self.orig_data_path = orig_data_path
        self.transform = transform
        self.debug = debug
        self.nan_count = 0
        self.data = data
        #  calculated in create_omage_dataset.py
        self.GLOBAL_AVERAGE_VERTEX_NORM = 5.215578427705153
        omages_path_list = [
            os.path.join(root_dir, data + "_" + split, f)
            for f in os.listdir(os.path.join(root_dir, data + "_" + split))
            if f.endswith(".npy")
        ]

        # Limit files for debug mode
        if self.debug:
            omages_path_list = omages_path_list[:100]
            print(f"DEBUG MODE: Limited to {len(omages_path_list)} samples")

        self.npy_files = omages_path_list
        self.split = split
        self.samples = self._load_samples()
        if split != "train":
            self.gt_samples = self._load_gt_samples()

    def __len__(self):
        return len(self.samples)

    def _load_samples(self):
        samples = []
        valid_files = []

        for npy_file in tqdm(self.npy_files):
            sample = np.load(npy_file, allow_pickle=True)
            sample = sample.item().get("mesh")
            if np.isnan(sample).any():
                self.nan_count += 1
                continue
            sample = torch.Tensor(np.transpose(sample, (2, 0, 1)))

            # A pixel is occupied if any coordinate is non-zero
            occupancy = (sample.abs().sum(dim=0, keepdim=True) > 0).float()  # Shape: (1, 512, 512)

            samples.append({"omage": sample, "occupancy": occupancy})
            valid_files.append(npy_file)

        self.npy_files = valid_files
        return samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)

        basename = os.path.basename(self.npy_files[idx])
        sample.update({"sample_id": basename[:-4]})

        # Add ground truth if available
        if hasattr(self, "gt_samples"):
            gt_sample = self.gt_samples.get(basename[:-4], None)
            if gt_sample is not None:
                sample["gt_pc"] = gt_sample

        return sample


class Conditional_Generation(Dataset):
    def __init__(
        self,
        root_dir,
        orig_data_path,
        split=None,
        transform=None,
        include_prep=True,
        include_neighbors=True,
        debug=False,
        conditions_in_memory=True,
    ):
        self.split = split.lower()
        self.root_dir = root_dir
        self.transform = transform
        self.include_prep = include_prep
        self.include_neighbors = include_neighbors
        self.debug = debug
        self.conditions_in_memory = conditions_in_memory
        self.GLOBAL_AVERAGE_VERTEX_NORM = (
            5.215578427705153  # calculated in create_omage_dataset.py
        )
        self.nan_count = 0
        self.crown_omage_folder = os.path.join(root_dir, "crown_" + self.split)
        self.crown_path_list = [
            os.path.join(root_dir, self.split, f)
            for f in os.listdir(self.crown_omage_folder)
            if f.endswith(".npy")
        ]

        # Limit files for debug mode
        if self.debug:
            self.crown_path_list = self.crown_path_list[:100]
            print(f"DEBUG MODE: Limited to {len(self.crown_path_list)} crown samples")

        self.orig_data_path = orig_data_path
        self.samples = self._load_samples()
        self.condition_point_count = 8192 * 2

    def _get_orig_data_path(self, basename, data_file_name="outer_crown.ply"):
        """Construct the original data path based on the basename."""
        # check if .npy is in basename
        if basename.endswith(".npy"):
            basename = basename[:-4]
        return os.path.join(self.orig_data_path, self.split, basename, data_file_name)

    def _load_samples(self):
        """Load samples and apply cutting operations on condition pc"""
        crown_basenames = {os.path.basename(path) for path in self.crown_path_list}

        all_sample_list = []  # Create a separate dictionary
        for i, sample_file_name in enumerate(
            tqdm(crown_basenames, desc="Loading samples")
        ):

            crown_path = os.path.join(self.crown_omage_folder, sample_file_name)
            crown_sample = np.load(crown_path, allow_pickle=True)
            crown_sample = crown_sample.item().get("mesh")

            # Check for NaNs, if found, skip the sample
            if np.isnan(crown_sample).any():
                self.nan_count += 1
                continue

            crown_sample = torch.Tensor(np.transpose(crown_sample, (2, 0, 1)))
            
            # Compute occupancy mask from the crown omage
            # A pixel is occupied if any coordinate is non-zero
            crown_occupancy = (crown_sample.abs().sum(dim=0, keepdim=True) > 0).float()  # Shape: (1, H, W)

            gt_crown_path = self._get_orig_data_path(
                sample_file_name, "outer_crown.ply"
            )
            gt_crown_pc = o3d.io.read_point_cloud(gt_crown_path)
            gt_crown_pc = np.asarray(gt_crown_pc.points)

            y_min, y_max = np.min(gt_crown_pc[:, 1]), np.max(gt_crown_pc[:, 1])
            y_mid = (y_min + y_max) / 2
            gt_crown_pc[:, 1] -= y_mid
            gt_crown_pc = gt_crown_pc / self.GLOBAL_AVERAGE_VERTEX_NORM

            if self.conditions_in_memory:
                with open(
                    self._get_orig_data_path(sample_file_name, "input_points.dat"),
                    "rb",
                ) as f:
                    input_pc = np.fromfile(f, dtype=np.float32).reshape(
                        -1, 6
                    )  # (x, y, z, nx, ny, nz)

                with open(
                    self._get_orig_data_path(sample_file_name, "input_properties.dat"),
                    "rb",
                ) as f:
                    # (diff to prep/antagonist, upper/lower jaw, jawSegmentKind)
                    # 0-dim: diff to prep/antagonist: 0 = antagonist / prep, 1 = unn4, -1 = unn2 or whatever on the other side
                    # 1-dim: upper/lower jaw: 0 = upper jaw, 1 = lower jaw
                    # 2-dim: jawSegmentKind: 0 = natural tooth, 1 = crown/prep
                    input_properties = np.fromfile(f, dtype=np.byte)
                    input_properties = input_properties.reshape(-1, 3)

                # Apply cutting operations
                antagonists = input_pc[
                    input_properties[:, 1] == 1, :
                ]  # Only natural teeth
                antagonists_ymax = np.percentile(antagonists[:, 1], 75)

                mask = (
                    antagonists[:, 1]
                    <= antagonists_ymax
                )
                antagonists_limited = antagonists[mask]

                if self.include_neighbors:
                    prep_neighbor1 = input_pc[
                        (input_properties[:, 0] == 1) & (input_properties[:, 1] == 0), :
                    ]  # Only preps and upper jaw
                    prep_neighbor2 = input_pc[
                        (input_properties[:, 0] == -1) & (input_properties[:, 1] == 0),
                        :,
                    ]  # Only preps and upper jaw

                if self.include_prep:
                    prep_mesh_path = self._get_orig_data_path(
                        sample_file_name, "prep_mesh.ply"
                    )
                    prep_mesh = o3d.io.read_point_cloud(prep_mesh_path)
                    prep_mesh.estimate_normals()
                    prep_mesh_points = np.asarray(prep_mesh.points)
                    prep_mesh_normals = np.asarray(prep_mesh.normals)
                    prep_mesh = np.concatenate(
                        [prep_mesh_points, prep_mesh_normals], axis=1
                    )

                    with open(
                        os.path.join(
                            os.path.dirname(gt_crown_path), "prep_distance2border.dat"
                        ),
                        "rb",
                    ) as f:
                        prep_vertex_distance2border = np.fromfile(f, dtype=np.double)
                    prep_mesh = prep_mesh[prep_vertex_distance2border < 1]

                # Construct final point cloud based on which components to include
                components = [antagonists_limited]
                labels = [np.zeros(len(antagonists_limited), dtype=np.int8)]

                # Add neighbors if required
                if self.include_neighbors:
                    components.append(prep_neighbor1)
                    labels.append(np.ones(len(prep_neighbor1), dtype=np.int8))

                    components.append(prep_neighbor2)
                    labels.append(np.ones(len(prep_neighbor2), dtype=np.int8) * 2)

                # Add prep if required
                if self.include_prep:
                    components.append(prep_mesh)
                    labels.append(np.ones(len(prep_mesh), dtype=np.int8) * 3)

                final_pc = np.concatenate(components, axis=0)
                final_pc_labels = np.concatenate(labels, axis=0)

                final_pc[:, 1] -= y_mid
                final_pc = final_pc / (self.GLOBAL_AVERAGE_VERTEX_NORM)

            out = {
                "crown_omage": crown_sample,
                "crown_occupancy": crown_occupancy,  # Add occupancy mask
                "crown_pc": gt_crown_pc,
                "sample_id": sample_file_name[:-4],
                "y_mid": y_mid,
            }

            if self.conditions_in_memory:
                out.update(
                    {
                        "condition_pc": final_pc,
                        "condition_antagonist_label": torch.Tensor(final_pc_labels),
                    }
                )

            all_sample_list.append(out)

        return all_sample_list  # Assign the populated dictionary to self.sample_dict

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        if self.transform:
            raise NotImplementedError

        if self.split != "train":
            gt_crown_pc = sample["crown_pc"]
            gt_crown_idx = fpsample.bucket_fps_kdline_sampling(gt_crown_pc, 2048, h=5)
            gt_crown_pc = torch.Tensor(gt_crown_pc[gt_crown_idx])
        else:
            gt_crown_pc = None

        condition_idx = fpsample.bucket_fps_kdline_sampling(
            sample["condition_pc"], self.condition_point_count, h=5
        )
        condition_antagonist_label = sample["condition_antagonist_label"][condition_idx]

        crown_omage = sample["crown_omage"]
        condition = sample["condition_pc"][condition_idx]

        result = {
                "crown_omage": crown_omage,
                "condition_pc": condition,
                "sample_id": sample["sample_id"],
                "condition_antagonist_label": condition_antagonist_label,
                "crown_y_mid": sample["y_mid"],
        }


        if self.split == "train":
            return result
        else:
            result.update(
                {
                    "crown_pc": gt_crown_pc,
                    "occupancy": sample["crown_occupancy"],  # Add occupancy for validation/test
                }
            )
            return result


class PreEncodedConditionalGeneration(Dataset):
    """
    Dataset class that loads pre-encoded crown omages instead of raw omages.
    This avoids the need to encode omages during training.
    """

    def __init__(
        self,
        data_path,
        encoded_path,
        orig_data_path,
        split="train",
        debug=False,
        conditions_in_memory=True,
    ):
        self.data_path = data_path
        self.encoded_path = encoded_path
        self.split = split
        self.debug = debug
        self.conditions_in_memory = conditions_in_memory
        self.GLOBAL_AVERAGE_VERTEX_NORM = (
            5.215578427705153  # calculated in create_omage_dataset.py
        )

        # Load the original dataset to get all other data (condition_pc, etc.)
        self.original_dataset = Conditional_Generation(
            data_path,
            orig_data_path,
            split=split,
            debug=debug,
            conditions_in_memory=conditions_in_memory,
        )
        self.orig_data_path = orig_data_path

        # Check that encoded files exist
        encoded_split_path = os.path.join(encoded_path, f"crown_{split}")
        if not os.path.exists(encoded_split_path):
            raise FileNotFoundError(f"Encoded data not found at {encoded_split_path}")
        self.encoded_path = encoded_split_path

        # Get list of available encoded files
        encoded_files_list = [
            f for f in os.listdir(encoded_split_path) if f.endswith(".npy")
        ]

        # Limit files for debug mode
        if self.debug:
            encoded_files_list = encoded_files_list[:100]
            print(f"DEBUG MODE: Limited to {len(encoded_files_list)} encoded samples")

        self.encoded_files = {
            f[:-4]: os.path.join(encoded_split_path, f) for f in encoded_files_list
        }

        # Filter original dataset to only include samples with encoded omages
        self.valid_samples = []
        for sample in self.original_dataset.samples:
            if sample["sample_id"] in self.encoded_files:
                self.valid_samples.append(sample)

        self.valid_samples = self._get_encoded_tooth_meshes()

        print(
            f"PreEncodedConditionalGeneration: {len(self.valid_samples)} samples with encoded omages"
        )

    def _get_encoded_tooth_meshes(self):
        for sample in self.valid_samples:
            sample_id = sample["sample_id"]
            if sample_id in self.encoded_files:
                encoded_crown_omage_path = self.encoded_files[sample_id]
                z_crown_omage = torch.from_numpy(np.load(encoded_crown_omage_path))
                sample["z_crown_omage"] = z_crown_omage
            
            # Load occupancy from original omage for validation/test
            if self.split != "train":
                crown_omage_path = os.path.join(
                    self.data_path, f"crown_{self.split}", sample_id + ".npy"
                )
                if os.path.exists(crown_omage_path):
                    crown_omage_data = np.load(crown_omage_path, allow_pickle=True).item().get("mesh")
                    crown_omage_tensor = torch.Tensor(np.transpose(crown_omage_data, (2, 0, 1)))
                    # Compute occupancy mask
                    crown_occupancy = (crown_omage_tensor.abs().sum(dim=0, keepdim=True) > 0).float()
                    sample["crown_occupancy"] = crown_occupancy
        return self.valid_samples

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        sample_id = sample["sample_id"]

        # Load pre-encoded crown omage
        z_crown_omage = sample["z_crown_omage"]

        orig_crown_path = self.original_dataset._get_orig_data_path(
            sample_id, "outer_crown.ply"
        )
        orig_dentition_mesh_path = self.original_dataset._get_orig_data_path(
            sample_id, "dentition_mesh.ply"
        )

        result = {
            "z_crown_omage": z_crown_omage,  # Pre-encoded latent
            "sample_id": sample_id,
            "crown_y_mid": sample["y_mid"],
            "orig_crown_path": orig_crown_path,
            "orig_dentition_mesh_path": orig_dentition_mesh_path,
        }
        if "condition_pc" in sample:
            condition_idx = fpsample.bucket_fps_kdline_sampling(
                sample["condition_pc"], self.original_dataset.condition_point_count, h=5
            )
            condition_pc = sample["condition_pc"][condition_idx].astype(dtype=np.float32)
            condition_antagonist_label = sample["condition_antagonist_label"][
                condition_idx
            ]
            result["condition_pc"] = condition_pc
            result["condition_antagonist_label"] = condition_antagonist_label


        if self.split != "train":
            gt_crown_pc = sample["crown_pc"]
            gt_crown_idx = fpsample.bucket_fps_kdline_sampling(gt_crown_pc, 2048, h=5)
            gt_crown_pc = torch.Tensor(gt_crown_pc[gt_crown_idx])
            result["crown_pc"] = gt_crown_pc
            # Add occupancy if available
            if "crown_occupancy" in sample:
                result["occupancy"] = sample["crown_occupancy"]
        return result
