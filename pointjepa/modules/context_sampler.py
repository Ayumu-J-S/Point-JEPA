from typing import Optional, Literal, Tuple
import torch


class ContextSampler:
    def __init__(
        self,
        sample_method: Literal["contiguous", "random", "rest"] = "contiguous",
        sample_ratio_range: Optional[Tuple[float, float]] = (0.75, 1),
        device="cuda",
    ):
        self._sample_method = sample_method
        self._sample_ratio_range = sample_ratio_range
        self._device = device

        if self._sample_method == "random" or self._sample_method == "nearest":
            if self._sample_ratio_range is None:
                raise ValueError(
                    "sample_ratio_range should be not be None for random or rest sampling"
                )

        if sample_ratio_range is not None and sample_ratio_range[0] > sample_ratio_range[1]:
            raise ValueError(
                "sample_ratio_range should be a tuple of (min, max) where min <= max"
            )
    
    def setup_device(self, device: str):
        self._device = device

    def get_context_random_idx(self, valid_indices: torch.Tensor ,num_context_selct: int) -> torch.Tensor:
        """Select random context tokens.

        Args:
            tokens: All tokens available. Shape: (B, T, E)
            valid_indices: The valid indices. Shape: (N)
            num_context_selct: Number of context tokens to select.
        
        Returns:
            torch.Tensor: The selected indices. Shape: (num_context_selct) 
        """
        selected_indices = torch.randperm(len(valid_indices), device=self._device)[:num_context_selct]
        return selected_indices

    def get_context_contiguous_idx(self, valid_indices: torch.Tensor ,num_context_selct: int) -> torch.Tensor:
        """Select contiguous context tokens starting from random index.

        Args:
            tokens: All tokens available. Shape: (B, T, E)
            valid_indices: The valid indices. Shape: (N)
            num_context_selct: Number of context tokens to select.

        Returns:
            torch.Tensor: The selected indices. Shape: (num_context_selct) 
        """
        max_valid_start = len(valid_indices) - num_context_selct
        start_index = torch.randint(0, max_valid_start + 1, (1,), device=self._device)
        selected_indices = valid_indices[start_index:start_index + num_context_selct]
        return selected_indices
    
    def sample(
        self,
        tokens: torch.Tensor,
        centers: torch.Tensor,
        target_indices: torch.Tensor,
    ):
        _, T, _ = tokens.shape

        # Calculate new ratio within specified range or use default
        ratio = (torch.rand(1, requires_grad=False, device=self._device) *
                 (self._sample_ratio_range[1] - self._sample_ratio_range[0]) +
                 self._sample_ratio_range[0]) if self._sample_ratio_range else torch.tensor([1], device=self._device)

        # Determine available indices for selection, excluding those in target_indices
        valid_bool = torch.ones(T, device=self._device, dtype=torch.bool)
        for idx in target_indices.flatten():
            valid_bool[idx] = 0  # Mark indices covered by target_indices as invalid

        # bool -> index
        valid_indices = valid_bool.nonzero().squeeze()
        # Above line can put tensor into scaler if only one valid index is available
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.reshape(1)

        num_to_select = max(1, int(ratio.item() * len(valid_indices)))
        if num_to_select > len(valid_indices):
            print(
                f"Number of tokens to select ({num_to_select}) exceeds available valid indices ({valid_indices.shape[0]}). Selecting one illegal index per sample.")
            num_to_select = 1
            valid_indices = torch.ones(T, device=self._device, dtype=torch.bool).nonzero().squeeze()


        if self._sample_method == "random":
            selected_indices = self.get_context_random_idx(valid_indices, num_to_select)
        elif self._sample_method == "contiguous":
            selected_indices = self.get_context_contiguous_idx(valid_indices, num_to_select)
        elif self._sample_method == "rest":
            selected_indices = valid_indices
        else:
            raise NotImplementedError(f"Unknown sample method: {self._sample_method}")

        context_blocks_tokens = tokens[:, selected_indices]
        context_blocks_centers = centers[:, selected_indices]

        return context_blocks_tokens, context_blocks_centers
    
