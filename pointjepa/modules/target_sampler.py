from typing import Literal, Tuple
import numpy as np
import torch
from pytorch3d.ops import knn_points


class TargetSampler:
    def __init__(
        self,
        sample_method: Literal["random", "contiguous"] = "contiguous",
        num_targets_per_sample=4,
        sample_ratio_range: Tuple[float, float] = (0.15, 0.2),
        device="cuda",
    ):
        self._sample_method = sample_method
        self._num_targets_per_sample = num_targets_per_sample
        self._sample_ratio_range = sample_ratio_range
        self._device = device
        
    def setup_device(self, device: str):
        self._device = device

    def select_embed_random(
        self, embed_global: torch.tensor, ratio: float
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Select random embeddings.

        Args:
            embed_global: Global embedding. Shape: (B, T, C)

        Returns:
            target_blocks: Target block. Shape: (M, B, k, C) where k = T * ratio
            target_embed_indices: Indices of selected embeddings. Shape: (B, M, k)
        """
        B, T, C = embed_global.shape

        num_to_select = max(1, int(T * ratio))
        selected_embed_list = []
        target_embed_indices = []

        for _ in range(self._num_targets_per_sample):
            # Randomly select a start index
            selected_indices = torch.randperm(T, device=self._device)[:num_to_select]
            target_embed_indices.append(selected_indices)

            expanded_indices = selected_indices.unsqueeze(-1).expand(B, -1, C) # [B, num_to_select, C]

            selected = embed_global.gather(1, expanded_indices)
            selected_embed_list.append(selected)

        # Stack the selected embeddings to get the desired shape [4, B, selected, C]
        target_blocks = torch.stack(selected_embed_list)
        target_embed_indices_tensor = torch.stack(target_embed_indices)  # Index of selected embeddings

        return (
            target_blocks,
            target_embed_indices_tensor,
        )

    def select_embed_contiguous(
        self, embed_global: torch.tensor, ratio: float
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Select random embeddings that are consecutive spatially.

        Args:
            embed_global: Global embedding. Shape: (B, T, C)

        Retruns:
            target_blocks: Target block. Shape: (M, B, k, C) where k = T * ratio
            target_embed_indices: Indices of  for target. Shape: (B, M, k)
        """
        B, T, C = embed_global.shape

        num_to_select = max(1, int(T * ratio))
        selected_embed_list = []
        target_embed_indices = []  # To keep track of the actual indices selected in target selection

        for _ in range(self._num_targets_per_sample):
            # Randomly select a start index
            max_start_index = T - num_to_select
            start_index = torch.randint(0, max_start_index + 1, (1,)).item()

            # Generate contiguous indices to select
            selection_indices = torch.arange(start_index, start_index + num_to_select, device=self._device).unsqueeze(0)
            selection_indices = selection_indices.expand(B, -1)
            target_embed_indices.append(selection_indices[0])  # Assuming uniform selection across the batch

            # Apply to the batch
            selected = torch.gather(embed_global, 1, selection_indices.unsqueeze(2).expand(-1, -1, C))
            selected_embed_list.append(selected)

        # Stack the selected embeddings to get the desired shape [4, B, selected, C]
        target_blocks = torch.stack(selected_embed_list)
        target_embed_indices_tensor = torch.stack(target_embed_indices)  # Index of selected embeddings
        return (
            target_blocks,
            target_embed_indices_tensor,
        )
    
    def sample(
        self, embed_global: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """Sample target blocks from the global embedding. Here global embedding is the embedding that came out
        of transformer encoder. Center is the corresponding center of each dimension of the global embedding in
        input space. More specifically the centers tensor is the tensor used to create positional encoding.

        Args:
            embed_global: Global embedding. Shape: (B, T, C)

        Returns:
            target_blocks: Target block. Shape: (M, B, k, C) where k = T * ratio
            target_embed_indices: Indices of selected embeddings. Shape: (B, M, k)
        """
        with torch.no_grad():
            ratio = torch.rand(1, requires_grad=False, device=self._device) * (
                self._sample_ratio_range[1] - self._sample_ratio_range[0]
            ) + self._sample_ratio_range[0]
            if self._sample_method == "contiguous":
                return self.select_embed_contiguous(embed_global, ratio)
            elif self._sample_method == "random":
                return self.select_embed_random(embed_global, ratio)
            else:
                raise NotImplementedError(f"Unknown sample method: {self._sample_method}")
