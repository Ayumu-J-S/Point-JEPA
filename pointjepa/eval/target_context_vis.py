import numpy as np
import torch
import sys
from pathlib import Path

# sys.path.append(str(Path(__file__).resolve().parents[1]))
from pointjepa.modules.tokenizer import PointcloudTokenizer
from pointjepa.utils import transforms
from pointjepa.modules.point_sequencer import PointSequencer
from pointjepa.modules.target_sampler import TargetSampler
from pointjepa.modules.context_sampler import ContextSampler
from pytorch3d.ops import ball_query, knn_gather, knn_points, sample_farthest_points


TARGET_SAMPLER = TargetSampler(
    sample_method="contiguous",
    num_targets_per_sample=4,
    sample_ratio_range=(0.15, 0.2),
    device="cpu",
)

CONTEXT_SAMPLER = ContextSampler(
    sample_method="contiguous", sample_ratio_range=(0.4, 0.75), device="cpu"
)


def load_npy_as_torch(file_path: str):
    data = np.load(file_path)

    if data.size % 3 != 0:
        raise ValueError(
            "The total number of elements in the .npy file is not divisible by 3."
        )

    reshaped_data = data.reshape(-1, 3)

    torch_tensor = torch.from_numpy(reshaped_data)

    return torch_tensor


def get_reordered_pc_group(pc: torch.tensor):
    tokenizer = PointcloudTokenizer(
        num_groups=64,
        group_size=32,
        group_radius=None,
        token_dim=384,
    )
    grouped_pts, group_center = tokenizer(pc)

    point_sequencer = PointSequencer(method="iterative_nearest_min_start")
    point_sequencer.setup_device("cpu")
    tokens, group_center = point_sequencer.reorder(grouped_pts, centers=group_center)
    return tokens, group_center


def get_target_idx(group_center):
    dummpy_embed = torch.zeros(group_center.shape)
    _, target_patches = TARGET_SAMPLER.sample(dummpy_embed, group_center)

    return target_patches


def get_context_block_ctr(
    tokens: torch.Tensor, centers: torch.Tensor, target_patches: torch.Tensor
):
    _, context_centers = CONTEXT_SAMPLER.sample(tokens, centers, target_patches)

    return context_centers


def save_points_overlap_original(
    pc: torch.tensor,
    context_pc: torch.tensor,
    file_path: Path,
    over_lap_color: np.array = np.array([0, 10, 255]),
):
    # Only saves the pc at the idx as black points and the original pc as red points
    if len(pc.shape) == 3:
        pc = pc.reshape(-1, 3)
    if len(context_pc.shape) == 3:
        context_pc = context_pc.reshape(-1, 3)
    pc = pc.numpy()
    context_pc = context_pc.numpy()

    # Avoid overlapping points
    pc = pc[~np.isin(pc, context_pc).all(1)]

    context_pc_colors = over_lap_color
    context_pc = np.concatenate(
        (context_pc, np.tile(context_pc_colors, (context_pc.shape[0], 1))), axis=1
    )

    pc_colors = np.array([0, 0, 0])
    pc = np.concatenate((pc, np.tile(pc_colors, (pc.shape[0], 1))), axis=1)

    pc = np.concatenate((pc, context_pc), axis=0)

    np.save(file_path, pc)


def save_points_original(pc: torch.tensor, file_path: Path):
    # Only saves the pc at the idx as black points and the original pc as red points
    if len(pc.shape) == 3:
        pc = pc.reshape(-1, 3)
    pc = pc.numpy()

    # pc_colors = np.array([10, 10, 12])
    # pc = np.concatenate((pc, np.tile(pc_colors, (pc.shape[0], 1))), axis=1)

    np.save(file_path, pc)


def main():
    file_path = "./data/ShapeNet55/shapenet_pc/02691156-1e7dbf0057e067586e88b250ea6544d0.npy"  # Specify the path to your .npy file
    # file_path = "data/ShapeNet55/shapenet_pc/02958343-a34dc1b89a52e2c92b12ea83455b0f44.npy" # car
    # file_path = "/home/ayumu/Documents/SSL_3DClassification/data/ShapeNet55/shapenet_pc/02818832-5cdbef59581ba6b89a87002a4eeaf610.npy" # bed
    out_dir = Path("./output/")
    out_dir.mkdir(exist_ok=True, parents=True)

    pc = load_npy_as_torch(file_path)
    pc = pc.unsqueeze(0)
    pc = pc.float()

    pc = transforms.PointcloudSubsampling(1024)(pc)
    save_points_original(pc, out_dir / "original.npy")
    tokens, group_center = get_reordered_pc_group(pc)

    target_patches = get_target_idx(group_center)

    _, idx, _ = knn_points(
        group_center.float(),
        pc[:, :, :3].float(),
        K=32,
        return_sorted=False,
        return_nn=False,
    )  # (B, G, K)
    group_pc = knn_gather(pc, idx)

    for i, target_patch in enumerate(target_patches):
        save_points_overlap_original(
            pc, group_pc[0][target_patch], out_dir / ("target_patch_" + str(i) + ".npy")
        )

    context_ctr = get_context_block_ctr(tokens, group_center, target_patches)

    _, idx, _ = knn_points(
        context_ctr.float(),
        pc[:, :, :3].float(),
        K=32,
        return_sorted=False,
        return_nn=False,
    )  # (B, G, K)
    ctx_group = knn_gather(pc, idx)
    selected_points_ctx = ctx_group[0].reshape(-1, 3)

    save_points_overlap_original(
        pc,
        selected_points_ctx,
        out_dir / "context.npy",
        over_lap_color=np.array([200, 0, 0]),
    )    

if __name__ == "__main__":
    main()
