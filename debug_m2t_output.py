"""
Debug script to check M2T (Motion-to-Text) model outputs and CIDEr calculation.
Run this to see what your model is actually generating and compare with ground truth.
"""

import json
import os
import sys
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae
from nlgmetricverse import NLGMetricverse, load_metric


def main():
    # Parse config - use the same config as your test
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER

    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Dataset
    datamodule = build_data(cfg)
    datamodule.setup("test")
    
    # Model
    model = build_model(cfg, datamodule)
    
    # Load checkpoint
    if cfg.TRAIN.PRETRAINED_VAE:
        load_pretrained_vae(cfg, model, None)
    if cfg.TEST.CHECKPOINTS:
        load_pretrained(cfg, model, None, phase="test")
    
    model.eval()
    model.cuda()

    # Get test dataloader
    test_loader = datamodule.test_dataloader()
    
    # Collect some predictions and ground truths
    pred_texts = []
    gt_texts = []
    
    print("\n" + "="*80)
    print("Checking model outputs for M2T task")
    print("="*80)
    
    num_samples = 10  # Check first 10 samples
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= 1:  # Just check first batch
                break
                
            # Move batch to GPU
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
            
            # Forward pass
            rs_set = model.val_m2t_forward(batch)
            
            # Get predictions and ground truths
            preds = rs_set["t_pred"]
            gts = batch["all_captions"]
            
            # Print some examples
            for i in range(min(num_samples, len(preds))):
                print(f"\n--- Sample {i+1} ---")
                print(f"Ground Truth: {gts[i][0] if isinstance(gts[i], list) else gts[i]}")
                print(f"Prediction:   {preds[i]}")
                
                pred_texts.append(preds[i])
                if isinstance(gts[i], list):
                    gt_texts.append(gts[i])
                else:
                    gt_texts.append([gts[i]])
    
    print("\n" + "="*80)
    print("Testing CIDEr calculation")
    print("="*80)
    
    # Initialize metrics
    metrics_list = [
        load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
        load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
        load_metric("rouge"),
        load_metric("cider"),
    ]
    nlg_evaluator = NLGMetricverse(metrics_list)
    
    # Calculate metrics
    scores = nlg_evaluator(predictions=pred_texts, references=gt_texts)
    
    print(f"\nRaw metrics from nlgmetricverse:")
    print(f"  BLEU-1: {scores['bleu_1']['score']:.4f}")
    print(f"  BLEU-4: {scores['bleu_4']['score']:.4f}")
    print(f"  ROUGE-L: {scores['rouge']['rougeL']:.4f}")
    print(f"  CIDEr (raw): {scores['cider']['score']:.4f}")
    print(f"  CIDEr (×10): {scores['cider']['score'] * 10:.4f}")
    
    print("\n" + "="*80)
    print("Typical paper reporting conventions:")
    print("="*80)
    print(f"  BLEU-1: {scores['bleu_1']['score']*100:.2f} (×100)")
    print(f"  BLEU-4: {scores['bleu_4']['score']*100:.2f} (×100)")  
    print(f"  ROUGE-L: {scores['rouge']['rougeL']*100:.2f} (×100)")
    print(f"  CIDEr: {scores['cider']['score']*10:.2f} (nlgmetricverse returns [0-10] scale)")
    
    print("\n" + "="*80)
    print("If your CIDEr is still low, possible issues:")
    print("="*80)
    print("1. Model is not generating meaningful text (check the predictions above)")
    print("2. Model checkpoint is not properly loaded")
    print("3. Model needs more training")
    print("4. Different evaluation protocol (multiple references, etc.)")


if __name__ == "__main__":
    main()
