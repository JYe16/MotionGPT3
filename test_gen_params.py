"""
Quick test script to experiment with different generation parameters for M2T.

This script allows you to test different generation hyperparameters
(num_beams, temperature, etc.) to see their effect on R-precision.

Usage:
    python test_gen_params.py --cfg configs/2_m2t_custom.yaml --num_beams 5
    python test_gen_params.py --cfg configs/2_m2t_custom.yaml --temperature 0.7
"""

import os
import sys
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from motGPT.config import parse_args, instantiate_from_config
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.logger import create_logger


def patch_generate_params(model, num_beams=1, do_sample=False, temperature=1.0, 
                          top_p=1.0, max_length=40):
    """
    Monkey-patch the model's generate_direct method to use custom parameters.
    """
    original_generate_direct = model.lm.generate_direct
    
    def patched_generate_direct(texts, max_length=max_length, num_beams=num_beams,
                                do_sample=do_sample, bad_words_ids=None, **kwargs):
        # Get device
        model.lm.device = model.lm.language_model.device
        
        # Tokenize
        if model.lm.lm_type == 'dec':
            texts = [text + " \n " for text in texts]
        
        source_encoding = model.lm.tokenizer(texts,
                                             padding='max_length',
                                             max_length=model.lm.max_length,
                                             truncation=True,
                                             return_attention_mask=True,
                                             add_special_tokens=True,
                                             return_tensors="pt")
        
        source_input_ids = source_encoding.input_ids.to(model.lm.device)
        source_attention_mask = source_encoding.attention_mask.to(model.lm.device)
        
        # Generate with custom parameters
        if model.lm.lm_type == 'dec':
            gen_kwargs = {
                'input_ids': source_input_ids,
                'attention_mask': source_attention_mask,
                'pad_token_id': model.lm.tokenizer.pad_token_id,
                'max_new_tokens': max_length,
                'num_beams': num_beams,
                'do_sample': do_sample,
            }
            
            if do_sample:
                gen_kwargs['temperature'] = temperature
                gen_kwargs['top_p'] = top_p
            
            outputs = model.lm.language_model.generate(**gen_kwargs)
            model.lm.tokenizer.padding_side = 'left'
        else:
            outputs = model.lm.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        
        outputs_string = model.lm.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        outputs_tokens, cleaned_text = model.lm.motion_string_to_token(outputs_string)
        
        return outputs_tokens, cleaned_text
    
    model.lm.generate_direct = patched_generate_direct
    print(f"[GenParams] Patched generate_direct with: num_beams={num_beams}, "
          f"do_sample={do_sample}, temperature={temperature}, top_p={top_p}, max_length={max_length}")


def main():
    # Parse additional generation arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--max_length', type=int, default=40)
    args, remaining = parser.parse_known_args()
    
    # Reconstruct sys.argv for parse_args
    sys.argv = [sys.argv[0], '--cfg', args.cfg] + remaining
    
    # Configs
    cfg = parse_args(phase="test")
    
    # Logger
    logger = create_logger(cfg, phase="test")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Print generation parameters
    logger.info("=" * 50)
    logger.info("Generation Parameters:")
    logger.info(f"  num_beams: {args.num_beams}")
    logger.info(f"  do_sample: {args.do_sample}")
    logger.info(f"  temperature: {args.temperature}")
    logger.info(f"  top_p: {args.top_p}")
    logger.info(f"  max_length: {args.max_length}")
    logger.info("=" * 50)
    
    # Seed
    pl.seed_everything(cfg.SEED_VALUE)
    
    # Environment Variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module initialized")
    
    # Model
    model = build_model(cfg, datamodule)
    logger.info("model loaded")
    
    # Patch generation parameters
    patch_generate_params(
        model,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=cfg.FOLDER_EXP,
        accelerator=cfg.ACCELERATOR,
        devices=cfg.DEVICE,
    )
    
    # Test
    trainer.test(model, datamodule=datamodule)
    
    logger.info("Testing ends!")


if __name__ == "__main__":
    main()
