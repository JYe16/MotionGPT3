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

    # Normalize each embedding to unit length.
    motion_embs = F.normalize(motion_embs, p=2, dim=-1)  # (N, D)

    # Gram matrix of cosine similarities: G_ij = cos(e_i, e_j)
    gram = motion_embs @ motion_embs.t()  # (N, N)

    # We want gram ≈ I: diagonal ~ 1, off-diagonal ~ 0
    eye = torch.eye(gram.size(0), device=device, dtype=gram.dtype)
    diff = gram - eye  # diagonal goes to 0, off-diagonal is cos(e_i, e_j)

    # Only off-diagonal really matters, but diagonal is already zero here.
    # Normalize by N*(N-1) to make it scale invariant w.r.t. number of motion tokens.
    n = gram.size(0)
    loss = diff.pow(2).sum() / (n * (n - 1))

    return loss


class MotionOrthogonalityLoss(nn.Module):
    """
    A PyTorch module wrapper for orthogonality loss computation.
    
    This can be used as a regularization term during training to encourage
    motion token embeddings to be orthogonal to each other.
    """
    
    def __init__(self, motion_codebook_size: int = 512, lambda_ortho: float = 0.1):
        """
        Args:
            motion_codebook_size: Number of motion tokens in the vocabulary.
            lambda_ortho: Weight for the orthogonality loss term.
        """
        super().__init__()
        self.motion_codebook_size = motion_codebook_size
        self.lambda_ortho = lambda_ortho
        self._motion_ids = None
    
    def get_motion_token_ids(self, tokenizer) -> torch.Tensor:
        """
        Get the token IDs for all motion tokens from the tokenizer.
        
        Args:
            tokenizer: The tokenizer containing motion tokens.
            
        Returns:
            Tensor of motion token IDs.
        """
        if self._motion_ids is None:
            motion_ids = []
            # Motion tokens are named '<motion_id_0>', '<motion_id_1>', etc.
            for i in range(self.motion_codebook_size):
                token_name = f'<motion_id_{i}>'
                token_id = tokenizer.convert_tokens_to_ids(token_name)
                if token_id != tokenizer.unk_token_id:
                    motion_ids.append(token_id)
            
            if len(motion_ids) > 0:
                self._motion_ids = torch.tensor(motion_ids, dtype=torch.long)
            else:
                self._motion_ids = None
                
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
            embedding_weight: The embedding weight matrix from the model.
            tokenizer: The tokenizer to get motion token IDs.
            device: The device to compute on.
            
        Returns:
            Weighted orthogonality loss.
        """
        motion_ids = self.get_motion_token_ids(tokenizer)
        loss = compute_motion_orthogonality_loss(embedding_weight, motion_ids, device)
        return self.lambda_ortho * loss


def get_embedding_weight_from_model(model) -> torch.Tensor:
    """
    Extract the embedding weight matrix from different model architectures.
    
    Args:
        model: The language model (GPT2, T5, LLaMA, etc.), possibly wrapped by PEFT/LoRA
        
    Returns:
        The embedding weight tensor.
    """
    # For PEFT-wrapped models (LoRA)
    if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
        base_model = model.base_model.model
        # GPT2
        if hasattr(base_model, 'transformer') and hasattr(base_model.transformer, 'wte'):
            return base_model.transformer.wte.weight
        # LLaMA (PEFT wrapped)
        if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
            return base_model.model.embed_tokens.weight
        # T5
        if hasattr(base_model, 'shared'):
            return base_model.shared.weight
    
    # For GPT2LMHeadModel (non-wrapped)
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte.weight
    
    # For LLaMA/MoTLlama (non-wrapped) - model.model.embed_tokens
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens.weight
    
    # For T5
    if hasattr(model, 'shared'):
        return model.shared.weight
    
    # For models with get_input_embeddings (fallback)
    if hasattr(model, 'get_input_embeddings'):
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, 'weight'):
            return emb.weight
    
    raise ValueError("Cannot find embedding weight in model")
