"""
Orthogonality regularization loss for motion token embeddings.

This module encourages motion token embeddings in the LLM's embedding space
to be mutually orthogonal, which can improve the representational quality
and separability of different motion tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_motion_orthogonality_loss(
    embedding_weight: torch.Tensor,
    motion_ids_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Encourage motion token embeddings to be mutually orthogonal.

    Args:
        embedding_weight: (vocab_size, hidden_dim), model's embedding matrix.
        motion_ids_tensor: (num_motion_tokens,), indices of motion tokens in the vocab.
        device: torch device.

    Returns:
        A scalar tensor representing the orthogonality regularization loss.
    """
    if motion_ids_tensor is None:
        # No motion tokens provided → no regularization.
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    motion_ids = motion_ids_tensor.to(device)
    motion_embs = embedding_weight[motion_ids]  # (N, D)

    # If less than 2 motion tokens, orthogonality is not meaningful.
    if motion_embs.size(0) < 2:
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Check for zero or very small embeddings that could cause NaN after normalization
    norms = motion_embs.norm(dim=-1)
    if (norms < 1e-8).any():
        # Some embeddings have near-zero norm, skip orthogonality loss to avoid NaN
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Normalize each embedding to unit length.
    motion_embs = F.normalize(motion_embs, p=2, dim=-1)  # (N, D)
    
    # Check for NaN after normalization (safety check)
    if torch.isnan(motion_embs).any():
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    # Gram matrix of cosine similarities: G_ij = cos(e_i, e_j)
    gram = motion_embs @ motion_embs.t()  # (N, N)

    # We want gram ≈ I: diagonal ~ 1, off-diagonal ~ 0
    eye = torch.eye(gram.size(0), device=device, dtype=gram.dtype)
    diff = gram - eye  # diagonal goes to 0, off-diagonal is cos(e_i, e_j)

    # Only off-diagonal really matters, but diagonal is already zero here.
    # Normalize by N*(N-1) to make it scale invariant w.r.t. number of motion tokens.
    n = gram.size(0)
    loss = diff.pow(2).sum() / (n * (n - 1))
    
    # Final NaN check
    if torch.isnan(loss):
        return torch.zeros((), device=device, dtype=embedding_weight.dtype)

    return loss


class MotionOrthogonalityLoss(nn.Module):
    """
    A PyTorch module wrapper for orthogonality loss computation.
    
    This can be used as a regularization term during training to encourage
    motion token embeddings to be orthogonal to each other.
    
    NOTE: In MoT architecture, motion tokens have their own embedding layer
    (motion_und_head with vocab size = motion_codebook_size + special tokens).
    The indices in this embedding are 0..511 for codebook tokens.
    """
    
    def __init__(self, motion_codebook_size: int = 512, lambda_ortho: float = 0.1):
        """
        Args:
            motion_codebook_size: Number of motion tokens in the codebook (typically 512).
            lambda_ortho: Weight for the orthogonality loss term.
        """
        super().__init__()
        self.motion_codebook_size = motion_codebook_size
        self.lambda_ortho = lambda_ortho
        self._motion_ids = None
    
    def get_motion_token_ids(self, tokenizer=None) -> torch.Tensor:
        """
        Get the token IDs for all motion codebook tokens.
        
        In MoT architecture, motion embedding layer has indices:
        - 0..511: codebook tokens
        - 512+: special tokens (som, eom, mask, pad)
        
        Args:
            tokenizer: Unused, kept for API compatibility.
            
        Returns:
            Tensor of motion token IDs (0 to motion_codebook_size-1).
        """
        if self._motion_ids is None:
            # Motion embedding indices are simply 0..motion_codebook_size-1
            self._motion_ids = torch.arange(self.motion_codebook_size, dtype=torch.long)
                
        return self._motion_ids
    
    def forward(
        self,
        embedding_weight: torch.Tensor,
        tokenizer,
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute the weighted orthogonality loss.
        
        Args:
            embedding_weight: The motion embedding weight matrix from the model.
            tokenizer: Unused, kept for API compatibility.
            device: The device to compute on.
            
        Returns:
            Weighted orthogonality loss.
        """
        motion_ids = self.get_motion_token_ids(tokenizer)
        loss = compute_motion_orthogonality_loss(embedding_weight, motion_ids, device)
        return self.lambda_ortho * loss


def get_embedding_weight_from_model(model) -> torch.Tensor:
    """
    Extract the MOTION embedding weight matrix from MoT model architectures.
    
    In MoT architecture, motion tokens have their own embedding layer
    (pre_processors[1] / motion_und_head) separate from text embeddings.
    This function returns the motion embedding weights for orthogonality loss.
    
    Args:
        model: The language model (MoTGPT2LMHeadModel, MoTLlamaForCausalLM, etc.), 
               possibly wrapped by PEFT/LoRA
        
    Returns:
        The motion embedding weight tensor.
    """
    # Navigate through PEFT wrapper if present
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base_model = model.base_model.model
    else:
        base_model = model
    
    # MoT models have pre_processors where index 1 is motion embedding
    if hasattr(base_model, 'pre_processors') and len(base_model.pre_processors) > 1:
        motion_embedding = base_model.pre_processors[1]  # motion_und_head
        if hasattr(motion_embedding, 'weight'):
            return motion_embedding.weight
    
    # Fallback: Try to get motion embedding from modality_infos
    if hasattr(base_model, 'modality_infos') and len(base_model.modality_infos) > 1:
        motion_info = base_model.modality_infos[1]
        if hasattr(motion_info, 'pre_processor') and hasattr(motion_info.pre_processor, 'weight'):
            return motion_info.pre_processor.weight
    
    raise ValueError("Cannot find motion embedding weight in model. "
                    "Make sure the model has pre_processors[1] (motion_und_head)")
