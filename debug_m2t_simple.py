"""
简单的调试脚本：直接打印模型生成的文本
"""
import os
import sys
import torch
from omegaconf import OmegaConf

# 添加项目路径
sys.path.insert(0, '/home/jackieye/code/MotionGPT3')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from motGPT.config import parse_args
from motGPT.data.build_data import build_data
from motGPT.models.build_model import build_model
from motGPT.utils.load_checkpoint import load_pretrained, load_pretrained_vae

def main():
    # 修改 sys.argv 来模拟命令行参数
    sys.argv = ['test.py', '--cfg', 'configs/m2t_custom_fixed.yaml', '--task', 'm2t']
    
    cfg = parse_args(phase="test")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.FOLDER_EXP = os.path.join(cfg.FOLDER, 'debug')
    os.makedirs(cfg.FOLDER_EXP, exist_ok=True)
    
    print("Loading datamodule...")
    datamodule = build_data(cfg)
    datamodule.setup("test")
    
    print("Building model...")
    model = build_model(cfg, datamodule)
    
    # Load checkpoint
    if cfg.TRAIN.PRETRAINED_VAE:
        print("Loading VAE...")
        load_pretrained_vae(cfg, model, None)
    if cfg.TEST.CHECKPOINTS:
        print(f"Loading checkpoint from {cfg.TEST.CHECKPOINTS}...")
        load_pretrained(cfg, model, None, phase="test")
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    # Get a single batch
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    
    # Move to GPU
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].cuda() if torch.cuda.is_available() else batch[key]
    
    print("\n" + "="*80)
    print("Testing M2T Generation")
    print("="*80)
    
    # 手动检查 LM 的生成
    print("\n--- Debug LM generation ---")
    
    # 获取 motion tokens
    m_tokens = batch.get("m_tokens")
    m_tokens_len = batch.get("m_tokens_len", batch["length"])
    
    if m_tokens is not None:
        print(f"Using m_tokens from batch: shape={m_tokens.shape}")
        # 处理 m_tokens_len
        if hasattr(m_tokens_len, 'tolist'):
            lengths_list = m_tokens_len.tolist()
        else:
            lengths_list = list(m_tokens_len)
        
        # 测试 LM 的直接生成
        lm = model.lm
        motion_strings = lm.motion_token_to_string(m_tokens.long(), lengths_list)
        print(f"Motion string (first sample): {motion_strings[0][:100]}...")
        
        # 构建输入
        tasks = [{
            'input': ['Generate text: <Motion_Placeholder>'],
            'output': ['']
        }] * len(lengths_list)
        texts = [''] * len(lengths_list)
        
        inputs, outputs = lm.template_fulfill(tasks, lengths_list, motion_strings, texts)
        print(f"\nTemplate fulfilled input (first sample): {inputs[0][:200]}...")
        
        # 直接调用生成
        print("\nCalling generate_direct...")
        outputs_tokens, cleaned_text = lm.generate_direct(
            inputs,
            max_length=40,
            num_beams=1,
            do_sample=False,
        )
        print(f"Cleaned text (first 5 samples):")
        for i in range(min(5, len(cleaned_text))):
            print(f"  {i}: {cleaned_text[i]}")
    else:
        print("No m_tokens in batch, using feats_ref...")
    
    with torch.no_grad():
        rs_set = model.val_m2t_forward(batch)
    
    pred_texts = rs_set["t_pred"]
    gt_texts = batch["all_captions"]
    
    print(f"\nNumber of samples in batch: {len(pred_texts)}")
    print("\n--- First 5 Samples ---")
    
    for i in range(min(5, len(pred_texts))):
        print(f"\n[Sample {i+1}]")
        gt = gt_texts[i][0] if isinstance(gt_texts[i], list) else gt_texts[i]
        print(f"  GT:   {gt}")
        print(f"  Pred: {pred_texts[i]}")
        print(f"  Pred length: {len(pred_texts[i])}")
        print(f"  Pred is empty: {len(pred_texts[i].strip()) == 0}")
    
    # 检查是否有空输出
    empty_count = sum(1 for p in pred_texts if len(p.strip()) == 0)
    print(f"\n{'='*80}")
    print(f"Empty predictions: {empty_count}/{len(pred_texts)}")
    
    # 检查常见问题模式
    print("\n--- Checking for common issues ---")
    
    # 检查是否所有输出都一样
    unique_preds = set(pred_texts)
    print(f"Unique predictions: {len(unique_preds)}/{len(pred_texts)}")
    
    if len(unique_preds) < 3:
        print("WARNING: Model might be outputting the same text for all inputs!")
        for p in unique_preds:
            print(f"  - '{p[:100]}...'")

if __name__ == "__main__":
    main()
