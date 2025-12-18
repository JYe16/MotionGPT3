"""
Training script with orthogonality regularization for motion token embeddings.

This script is identical to train.py but uses the MotGPTOrtho model variant
that adds orthogonality loss to encourage motion token embeddings to be
mutually orthogonal in the LLM's embedding space.

Key Features:
1. Loads pre-trained VQ-VAE codebook weights
2. Applies sparse projection to map codebook dimensions (512) to GPT2 hidden size (768)
3. Initializes motion token embeddings with projected codebook weights
4. Adds orthogonality loss during training

Usage:
    python train_ortho.py --cfg configs/m2t_custom_ortho.yaml

The orthogonality loss weight (lambda_ortho) can be configured in the YAML file
under model.params.lambda_ortho
"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf
from motGPT.callback import build_callbacks
from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

# PEFT LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_vqvae_codebook(codebook_path: str, logger=None) -> torch.Tensor:
    """
    Load VQ-VAE codebook weights from checkpoint.
    
    Args:
        codebook_path: Path to the codebook checkpoint file
        logger: Optional logger for printing info
        
    Returns:
        codebook_weight: Tensor of shape [num_codes, code_dim], e.g., [512, 512]
    """
    if logger:
        logger.info(f"Loading VQ-VAE codebook from: {codebook_path}")
    
    checkpoint = torch.load(codebook_path, map_location='cpu')
    
    # Extract codebook weights
    codebook_key = 'vqvae.quantizer.codebook.weight'
    if codebook_key in checkpoint:
        codebook_weight = checkpoint[codebook_key]
    else:
        raise KeyError(f"Codebook key '{codebook_key}' not found in checkpoint. "
                      f"Available keys: {list(checkpoint.keys())[:10]}...")
    
    if logger:
        logger.info(f"Codebook shape: {codebook_weight.shape}")  # Expected: [512, 512]
    
    return codebook_weight


def create_sparse_projection(input_dim: int, output_dim: int, sparsity: float = 0.9, 
                             seed: int = 42) -> torch.Tensor:
    """
    Create a sparse random projection matrix.
    
    This projects vectors from input_dim to output_dim using a sparse random matrix.
    Sparse projection can help preserve distances while being computationally efficient.
    
    Args:
        input_dim: Input dimension (e.g., 512 for codebook)
        output_dim: Output dimension (e.g., 768 for GPT2)
        sparsity: Fraction of zero entries (default 0.9 means 90% sparse)
        seed: Random seed for reproducibility
        
    Returns:
        projection_matrix: Tensor of shape [input_dim, output_dim]
    """
    torch.manual_seed(seed)
    
    # Create sparse random projection matrix
    # Following the approach in sparse random projections literature
    s = 1.0 / (1.0 - sparsity)  # scaling factor
    
    # Initialize with zeros
    projection = torch.zeros(input_dim, output_dim)
    
    # Fill with sparse entries: +sqrt(s), 0, -sqrt(s) with probabilities 1/2s, 1-1/s, 1/2s
    prob_nonzero = 1.0 / s
    mask = torch.rand(input_dim, output_dim) < prob_nonzero
    
    # Randomly assign +1 or -1 to non-zero entries
    signs = torch.randint(0, 2, (input_dim, output_dim)) * 2 - 1  # -1 or +1
    projection[mask] = signs[mask].float() * (s ** 0.5)
    
    # Normalize columns to have unit norm (optional, for better embedding initialization)
    # projection = projection / (projection.norm(dim=0, keepdim=True) + 1e-8)
    
    # Scale to preserve expected norms
    projection = projection / (input_dim ** 0.5)
    
    return projection


def initialize_motion_token_embeddings(model, codebook_weight: torch.Tensor, 
                                       original_vocab_size: int = 50257,
                                       logger=None):
    """
    Initialize motion token embeddings in GPT2 using projected VQ-VAE codebook weights.
    
    The GPT2 embedding layer has been resized to include motion tokens.
    Original vocab size is 50257, motion tokens are added after (indices 50257+).
    
    Motion tokens:
        - <motion_id_0> to <motion_id_511>: 512 codebook tokens
        - <motion_id_512>, <motion_id_513>, <motion_id_514>: 3 special tokens
    
    Args:
        model: The MotGPTOrtho model
        codebook_weight: Tensor of shape [512, 512] from VQ-VAE
        original_vocab_size: Original GPT2 vocab size (50257)
        logger: Optional logger
    """
    # Get the GPT2 embedding layer
    gpt2 = model.lm.language_model
    embedding_layer = gpt2.transformer.wte  # word token embeddings
    
    # Get dimensions
    num_codes, code_dim = codebook_weight.shape  # [512, 512]
    total_vocab_size, hidden_dim = embedding_layer.weight.shape  # [50772, 768]
    
    if logger:
        logger.info(f"=" * 60)
        logger.info(f"Initializing motion token embeddings from VQ-VAE codebook")
        logger.info(f"  Codebook: {num_codes} codes × {code_dim} dim")
        logger.info(f"  GPT2 embedding: {total_vocab_size} vocab × {hidden_dim} hidden")
        logger.info(f"  Original vocab size: {original_vocab_size}")
        logger.info(f"  Motion token indices: {original_vocab_size} to {original_vocab_size + num_codes - 1}")
    
    # Create sparse projection: [512, 768]
    projection_matrix = create_sparse_projection(
        input_dim=code_dim, 
        output_dim=hidden_dim,
        sparsity=0.9,
        seed=42
    )
    
    if logger:
        logger.info(f"  Sparse projection matrix: {projection_matrix.shape}")
        nonzero_ratio = (projection_matrix != 0).float().mean().item()
        logger.info(f"  Projection non-zero ratio: {nonzero_ratio:.2%}")
    
    # Project codebook to GPT2 hidden dimension: [512, 512] @ [512, 768] = [512, 768]
    projected_codebook = codebook_weight @ projection_matrix  # [512, 768]
    
    # Normalize projected embeddings to match the scale of existing GPT2 embeddings
    # Get the mean norm of existing GPT2 embeddings for reference
    with torch.no_grad():
        existing_norms = embedding_layer.weight[:original_vocab_size].norm(dim=1)
        mean_norm = existing_norms.mean().item()
        std_norm = existing_norms.std().item()
        
        if logger:
            logger.info(f"  Existing GPT2 embedding norms: mean={mean_norm:.4f}, std={std_norm:.4f}")
        
        # Normalize projected codebook to have similar norms
        projected_norms = projected_codebook.norm(dim=1, keepdim=True)
        projected_codebook = projected_codebook / (projected_norms + 1e-8) * mean_norm
        
        if logger:
            new_norms = projected_codebook.norm(dim=1)
            logger.info(f"  Projected embedding norms: mean={new_norms.mean().item():.4f}, std={new_norms.std().item():.4f}")
    
    # Initialize motion token embeddings (indices 50257 to 50257+511)
    with torch.no_grad():
        motion_start_idx = original_vocab_size
        motion_end_idx = original_vocab_size + num_codes
        
        # Copy projected codebook to embedding layer
        embedding_layer.weight[motion_start_idx:motion_end_idx] = projected_codebook.to(
            embedding_layer.weight.device
        )
        
        # Also initialize the LM head (tied weights in GPT2, but just to be safe)
        if hasattr(gpt2, 'lm_head') and gpt2.lm_head.weight is not embedding_layer.weight:
            gpt2.lm_head.weight[motion_start_idx:motion_end_idx] = projected_codebook.to(
                gpt2.lm_head.weight.device
            )
    
    if logger:
        logger.info(f"  Successfully initialized {num_codes} motion token embeddings")
        logger.info(f"  (3 special tokens <motion_id_512-514> remain randomly initialized)")
        logger.info(f"=" * 60)


def apply_lora_to_gpt2(model, lora_config: dict, original_vocab_size: int = 50257, logger=None):
    """
    Apply LoRA (Low-Rank Adaptation) to the GPT2 model inside MotGPT.
    
    This significantly reduces the number of trainable parameters while
    maintaining model performance. Importantly, this also ensures the 
    motion token embeddings remain trainable.
    
    Args:
        model: The MotGPTOrtho model containing self.lm.language_model (GPT2)
        lora_config: Dictionary with LoRA hyperparameters:
            - r: LoRA rank (default: 8)
            - lora_alpha: LoRA alpha scaling (default: 16)
            - lora_dropout: Dropout probability (default: 0.05)
            - target_modules: Which modules to apply LoRA to
            - train_embeddings: 'all' (full embedding), 'motion_only' (only motion tokens), or False
        original_vocab_size: GPT2 original vocab size (50257), used to identify motion tokens
        logger: Optional logger
    """
    if not PEFT_AVAILABLE:
        if logger:
            logger.warning("PEFT not installed. Run: pip install peft")
            logger.warning("Continuing without LoRA...")
        return model
    
    # Get LoRA hyperparameters with defaults
    r = lora_config.get('r', 8)
    lora_alpha = lora_config.get('lora_alpha', 16)
    lora_dropout = lora_config.get('lora_dropout', 0.05)
    target_modules = lora_config.get('target_modules', ['c_attn', 'c_proj'])
    train_embeddings = lora_config.get('train_embeddings', 'motion_only')  # 'all', 'motion_only', or False
    
    if logger:
        logger.info(f"=" * 60)
        logger.info(f"Applying LoRA to GPT2")
        logger.info(f"  r (rank): {r}")
        logger.info(f"  lora_alpha: {lora_alpha}")
        logger.info(f"  lora_dropout: {lora_dropout}")
        logger.info(f"  target_modules: {target_modules}")
        logger.info(f"  train_embeddings: {train_embeddings}")
    
    # Count parameters before LoRA
    gpt2 = model.lm.language_model
    total_params_before = sum(p.numel() for p in gpt2.parameters())
    trainable_params_before = sum(p.numel() for p in gpt2.parameters() if p.requires_grad)
    
    # Create LoRA config
    # Determine modules_to_save based on train_embeddings setting
    modules_to_save = None
    if train_embeddings == 'all':
        # Train entire embedding layer
        modules_to_save = ['wte', 'lm_head']
    # For 'motion_only' or False, we don't use modules_to_save
    # We'll handle motion-only training separately after LoRA is applied
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
    )
    
    # Apply LoRA to GPT2
    model.lm.language_model = get_peft_model(gpt2, peft_config)
    
    # If train_embeddings == 'motion_only', manually enable gradients for motion tokens only
    if train_embeddings == 'motion_only':
        if logger:
            logger.info(f"  Enabling gradients only for motion token embeddings (indices {original_vocab_size}+)")
        
        # Get the embedding layer - need to navigate through PEFT wrapper
        peft_model = model.lm.language_model
        base_model = peft_model.base_model.model  # Get underlying GPT2
        
        # Enable gradients for embedding layer but we'll use a hook to mask gradients
        wte = base_model.transformer.wte
        lm_head = base_model.lm_head
        
        # Make embedding layer require grad
        wte.weight.requires_grad = True
        if lm_head.weight is not wte.weight:  # If not tied
            lm_head.weight.requires_grad = True
        
        # Register hook to zero out gradients for non-motion tokens
        def create_embedding_grad_hook(vocab_size):
            def hook(grad):
                # Zero out gradients for original vocab tokens
                mask = torch.ones_like(grad)
                mask[:vocab_size] = 0
                return grad * mask
            return hook
        
        wte.weight.register_hook(create_embedding_grad_hook(original_vocab_size))
        if lm_head.weight is not wte.weight:
            lm_head.weight.register_hook(create_embedding_grad_hook(original_vocab_size))
        
        num_motion_tokens = wte.weight.shape[0] - original_vocab_size
        hidden_dim = wte.weight.shape[1]
        motion_params = num_motion_tokens * hidden_dim  # motion_tokens * hidden_dim
        if logger:
            logger.info(f"  Motion token embeddings: {num_motion_tokens} tokens × {hidden_dim} dim = {motion_params:,} params")
    
    # Count parameters after LoRA
    trainable_params_after = sum(
        p.numel() for p in model.lm.language_model.parameters() if p.requires_grad
    )
    
    # Count LoRA params specifically
    lora_params = sum(
        p.numel() for name, p in model.lm.language_model.named_parameters() 
        if p.requires_grad and 'lora' in name.lower()
    )
    
    if logger:
        logger.info(f"  Total parameters: {total_params_before:,}")
        logger.info(f"  Trainable before LoRA: {trainable_params_before:,}")
        logger.info(f"  Trainable after LoRA (technical): {trainable_params_after:,}")
        logger.info(f"  LoRA adapter params: {lora_params:,}")
        
        if train_embeddings == 'motion_only':
            # Calculate effective trainable params
            peft_model = model.lm.language_model
            base_model = peft_model.base_model.model
            total_vocab = base_model.transformer.wte.weight.shape[0]
            hidden_dim = base_model.transformer.wte.weight.shape[1]
            num_motion = total_vocab - original_vocab_size
            
            # Check if wte and lm_head are tied (GPT2 default is tied)
            wte_tied = base_model.transformer.wte.weight is base_model.lm_head.weight
            multiplier = 1 if wte_tied else 2
            
            motion_emb_params = num_motion * hidden_dim * multiplier
            effective_trainable = lora_params + motion_emb_params
            
            logger.info(f"  Motion embedding params: {motion_emb_params:,} ({num_motion} tokens × {hidden_dim} dim" + 
                       (f", tied)" if wte_tied else f" × 2)"))
            logger.info(f"  Effective trainable: {effective_trainable:,}")
        
        reduction = (1 - trainable_params_after / trainable_params_before) * 100
        logger.info(f"  Technical parameter reduction: {reduction:.1f}%")
        logger.info(f"=" * 60)
    
    # Print trainable parameters summary
    if logger:
        model.lm.language_model.print_trainable_parameters()
    
    return model


def main():
    # Configs
    cfg = parse_args(phase="train")  # parse config file

    # Logger
    logger = create_logger(cfg, phase="train")  # create logger
    logger.info(OmegaConf.to_yaml(cfg))  # print config file

    # Log orthogonality loss info if using MotGPTOrtho
    if 'ortho' in cfg.model.target.lower():
        lambda_ortho = cfg.model.params.get('lambda_ortho', 0.1)
        logger.info(f"=" * 50)
        logger.info(f"Using MotGPTOrtho with orthogonality regularization")
        logger.info(f"lambda_ortho = {lambda_ortho}")
        logger.info(f"=" * 50)

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Metric Logger
    pl_loggers = []
    for loggerName in cfg.LOGGER.TYPE:
        if loggerName == 'tenosrboard' or cfg.LOGGER.WANDB.params.project:
            pl_logger = instantiate_from_config(
                eval(f'cfg.LOGGER.{loggerName.upper()}'))
            pl_loggers.append(pl_logger)

    # Callbacks
    callbacks = build_callbacks(cfg, logger=logger, phase='train')
    logger.info("Callbacks initialized")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # Model
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # =====================================================
    # NEW: Initialize motion token embeddings from VQ-VAE codebook
    # =====================================================
    codebook_path = cfg.get('CODEBOOK_PATH', 'checkpoints/codebook_st_share.pt')
    if os.path.exists(codebook_path):
        try:
            # Load VQ-VAE codebook weights
            codebook_weight = load_vqvae_codebook(codebook_path, logger)
            
            # Initialize motion token embeddings with sparse projection
            initialize_motion_token_embeddings(
                model=model,
                codebook_weight=codebook_weight,
                original_vocab_size=50257,  # GPT2 original vocab size
                logger=logger
            )
        except Exception as e:
            logger.warning(f"Failed to initialize motion embeddings from codebook: {e}")
            logger.warning("Continuing with random initialization...")
    else:
        logger.warning(f"Codebook file not found at: {codebook_path}")
        logger.warning("Using default random initialization for motion token embeddings")

    # =====================================================
    # NEW: Apply LoRA to GPT2 for parameter-efficient training
    # =====================================================
    lora_config = cfg.get('LORA', None)
    if lora_config and lora_config.get('ENABLED', False):
        model = apply_lora_to_gpt2(
            model=model,
            lora_config=lora_config,
            logger=logger
        )
    else:
        logger.info("LoRA disabled. Training full GPT2 model.")

    # Seed
    pl.seed_everything(cfg.SEED_VALUE)

    # Lightning Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        max_epochs=cfg.TRAIN.END_EPOCH,
        logger=pl_loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=cfg.LOGGER.VAL_EVERY_STEPS,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
        num_nodes=cfg.NUM_NODES,
        strategy="ddp_find_unused_parameters_true"
        if len(cfg.DEVICE) > 1 else 'auto',
        benchmark=False,
        deterministic=False,
        accumulate_grad_batches=cfg.TRAIN.accumulate_grad_batches,
    )
    logger.info("Trainer initialized")

    # Strict load pretrained model
    if cfg.TRAIN.PRETRAINED:
        load_pretrained(cfg, model, logger)

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, logger)

    # Lightning Fitting
    if cfg.TRAIN.RESUME:
        trainer.fit(model,
                    datamodule=datamodule,
                    ckpt_path=cfg.TRAIN.PRETRAINED)
    else:
        trainer.fit(model, datamodule=datamodule)

    # Training ends
    logger.info(
        f"The outputs of this experiment are stored in {cfg.FOLDER_EXP}")
    logger.info("Training ends!")


if __name__ == "__main__":
    main()
