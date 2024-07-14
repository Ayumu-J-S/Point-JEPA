import torch
import numpy as np

from typing import List
from pointjepa.modules.z_order import get_z_values

class PointSequencer:
    def __init__(self, method:str = "morton", device: str="cuda") -> None:
         self.device = device
         self.method = method
    
    def setup_device(self, device: str):
        self.device = device
    
    def sort_iterative_nearest(self, center: torch, min_start: bool = False) -> torch.tensor:
        """Orders the center points using iterative nearest neighbor search.
        Reference: https://github.com/CGuangyan-BIT/PointGPT/blob/V1.2/models/PointGPT.py
        
        Args:
            center: The center points. Shape: (B, N, C)
            min_start: If True, the starting point of each group is the one with the smallest in coordinates.
                If False, take the first index as the starting point.
        
        Returns:
            torch.tensor: The sorted indices. Shape: (B * N)
        """
        batch_size, num_group, _ = center.shape
        # Calculate a pairwise distance matrix for each center in the batch
        distances_batch = torch.cdist(center, center) # (B, N, N)
        # Set the diagonal to infinity to ignore itself as the closest point
        distances_batch[:, torch.eye(num_group).bool()] = float("inf") 
        # Starting index for each elem in the batch
        idx_base = torch.arange(
            0, batch_size, device=self.device) * num_group
        
        # This will keep the indices of the point in the sorted order
        sorted_indices_list: List[torch.tensor] = [] # [ (B), (B), (B), ....]
        idx_start = None
        if min_start:
            coord_sum = center.sum(dim=2)# (B, N)
            min_sum_idx = torch.argmin(coord_sum, dim=1) # Finds the minimum sum index

            idx_min = idx_base + min_sum_idx
            sorted_indices_list.append(idx_min)
            idx_start = idx_min
        else:
            idx_start = idx_base
            sorted_indices_list.append(idx_base)

        # Setting distances at idx_base to infinity to prevent selecting the starting point of each group as 
        # its own closest neighbor,
        distances_batch = distances_batch.view(batch_size, num_group, num_group).transpose(
            1, 2).contiguous().view(batch_size * num_group, num_group)
        distances_batch[idx_start] = float("inf")
        distances_batch = distances_batch.view(
            batch_size, num_group, num_group).transpose(1, 2).contiguous()
        
        for _ in range(num_group - 1):
            distances_batch = distances_batch.view(
                batch_size * num_group, num_group)
            
            # Find the closest point to the last point in the sorted list
            distances_to_last_batch = distances_batch[sorted_indices_list[-1]]
            closest_point_idx = torch.argmin(distances_to_last_batch, dim=-1)
            closest_point_idx = closest_point_idx + idx_base
            sorted_indices_list.append(closest_point_idx)

            # Set the chosen point to infinity to prevent it from being chosen again
            distances_batch = distances_batch.view(batch_size, num_group, num_group).transpose(
                1, 2).contiguous().view(batch_size * num_group, num_group)
            distances_batch[closest_point_idx] = float("inf")
            distances_batch = distances_batch.view(
                batch_size, num_group, num_group).transpose(1, 2).contiguous()
            
        sorted_indices = torch.stack(sorted_indices_list, dim=-1)
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices

    def sort_morton(self, center):
        batch_size, num_group, _ = center.shape
        all_indices = []
        for index in range(batch_size):
            points = center[index]
            z = get_z_values(points.cpu().numpy())
            idxs = np.zeros((num_group), dtype=np.int32)
            temp = np.arange(num_group)
            z_ind = np.argsort(z[temp])
            idxs = temp[z_ind]
            all_indices.append(idxs)
        
        # For performance, we convert the list of indices to a numpy array and then to a tensor
        all_indices = np.array(all_indices)
        all_indices = torch.tensor(all_indices, device=center.device)

        idx_base = torch.arange(0, batch_size, device=center.device).view(-1, 1) * num_group
        sorted_indices = all_indices + idx_base
        sorted_indices = sorted_indices.view(-1)
        return sorted_indices


    def reorder(self, tokens: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Reorders the tokens based on the sorted indices.
        
        Args:
            tokens: The tokens. Shape: (B, N, E)
            centers: The center points. Shape: (B, N, C)
        
        Returns:
            reordered_tokens. Shape: (B, N, E) and 
            The reordered centers. Shape: (B, N, C)
        """
        B, N, E = tokens.shape
        _, _, C = centers.shape

        if self.method == "iterative_nearest":
            sorted_indiices = self.sort_iterative_nearest(centers, min_start=False)
        elif self.method == "iterative_nearest_min_start":
            sorted_indiices = self.sort_iterative_nearest(centers, min_start=True)
        elif self.method == "morton":
            sorted_indiices = self.sort_morton(centers)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        tokens = tokens.view(B * N, E)[sorted_indiices, :]
        centers = centers.view(B * N, C)[sorted_indiices, :]

        return tokens.view(B, N, E).contiguous(), centers.view(B, N, C).contiguous()


         
         