import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class Motion2TextDatasetTokenCustom(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        code_path="motion_tokens", # 默认 token 文件夹名
        w_vectorizer=None,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.mean = mean
        self.std = std
        self.split = split
        
        # 路径设置
        split_file = pjoin(data_root, split + '.txt')
        self.text_dir = pjoin(data_root, 'texts')
        self.motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        # 处理 Token 路径
        # 如果配置中传入了 code_path，使用配置的；否则默认 'motion_tokens'
        actual_code_path = kwargs.get('code_path', code_path)
        if actual_code_path is None:
            actual_code_path = "motion_tokens"
            
        if os.path.isabs(actual_code_path):
            self.motion_token_dir = actual_code_path
        else:
            self.motion_token_dir = pjoin(data_root, actual_code_path)

        print(f"[CustomDataset] Loading tokens from: {self.motion_token_dir}")

        # 加载 Split 文件
        self.id_list = []
        if os.path.exists(split_file):
            with cs.open(split_file, "r") as f:
                for line in f.readlines():
                    self.id_list.append(line.strip())
        else:
            print(f"[Error] Split file not found: {split_file}")

        # 数据过滤与加载逻辑
        new_name_list = []
        data_dict = {}
        
        print(f"Loading {split} dataset...")
        for name in tqdm(self.id_list):
            try:
                # 1. 检查 Token 文件
                token_file = pjoin(self.motion_token_dir, name + '.npy')
                if not os.path.exists(token_file):
                    continue
                
                # 2. 检查长度 (快速读取)
                tokens = np.load(token_file)
                if len(tokens.shape) > 1:
                    tokens = tokens.flatten()
                    
                if len(tokens) < self.min_motion_length or len(tokens) >= self.max_motion_length:
                    continue

                # 3. 加载文本
                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                    
                text_data = []
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        if len(line_split) < 2: continue
                        tokens_text = line_split[1].split(' ')
                        
                        text_dict = {
                            'caption': caption,
                            'tokens': tokens_text
                        }
                        text_data.append(text_dict)

                if len(text_data) > 0:
                    data_dict[name] = {
                        'token_path': token_file,
                        'text': text_data
                    }
                    new_name_list.append(name)
            except Exception as e:
                pass

        self.data_dict = data_dict
        self.name_list = new_name_list
        print(f"[CustomDataset] Loaded {len(self.name_list)} samples for {split}")

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        fname = self.name_list[item]
        data = self.data_dict[fname]
        
        # 1. 加载 Motion Tokens
        m_tokens = np.load(data['token_path'])
        if len(m_tokens.shape) > 1:
            m_tokens = m_tokens.flatten()
        
        # 截断处理
        m_length = len(m_tokens)
        if m_length > self.max_motion_length:
            m_tokens = m_tokens[:self.max_motion_length]
            m_length = self.max_motion_length

        # 2. 加载 Motion (用于评估)
        motion_file = pjoin(self.motion_dir, fname + '.npy')
        if os.path.exists(motion_file):
            motion = np.load(motion_file)
            motion = (motion - self.mean) / self.std
            motion_len = len(motion)
            # 简单截断以匹配 token (假设大致对应，或者评估时只用前一段)
            # 注意：这里没有严格对齐 token 和 motion 的裁剪，因为 token 可能是 VQ 后的
            # 但对于 R-Precision，只要是同一个动作即可
            if motion_len > self.max_motion_length:
                motion = motion[:self.max_motion_length]
                motion_len = self.max_motion_length
        else:
            # Fallback
            motion = np.zeros((m_length * 4, 263))
            motion_len = m_length * 4
            
        # 3. 加载 Text
        text_list = data['text']
        
        # 训练时随机选择一条文本，测试时通常也随机，或者在评估代码中处理多条
        text_data = random.choice(text_list)
        caption = text_data['caption']
        tokens = text_data['tokens']

        # Process text if w_vectorizer is available
        word_embeddings = None
        pos_one_hots = None
        sent_len = None
        
        if self.w_vectorizer is not None:
            max_text_len = 20
            if len(tokens) < max_text_len:
                tokens_padded = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sent_len = len(tokens_padded)
                tokens_padded = tokens_padded + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
            else:
                tokens_padded = tokens[:max_text_len]
                tokens_padded = ["sos/OTHER"] + tokens_padded + ["eos/OTHER"]
                sent_len = len(tokens_padded)
            
            pos_one_hots_list = []
            word_embeddings_list = []
            for token in tokens_padded:
                word_emb, pos_oh = self.w_vectorizer[token]
                pos_one_hots_list.append(pos_oh[None, :])
                word_embeddings_list.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots_list, axis=0)
            word_embeddings = np.concatenate(word_embeddings_list, axis=0)
        
        # 获取所有 caption (用于评估)
        all_captions = [x['caption'] for x in text_list]

        # 3. 构造返回值 (适配 MotionGPT3 collate_fn)
        # 即使是 M2T 任务，collate_fn 可能仍期望 motion tensor 的存在
        
        # 定义任务
        task = {
            "class": "m2t",
            "input": ["Generate text: <Motion_Placeholder>"],
            "output": ["<Caption_Placeholder>"]
        }

        # 返回元组顺序参考 dataset_t2m.py:
        # caption, m_tokens, m_tokens_len, motion, m_length, word_embs, pos_ohot, text_len, tokens, all_captions, tasks, fname
        
        return (
            caption,        # caption
            m_tokens,       # m_tokens (INPUT)
            m_length,       # m_tokens_len
            motion,         # motion (Real Motion)
            motion_len,     # length (Frame Length)
            word_embeddings,# word_embs
            pos_one_hots,   # pos_ohot
            sent_len,       # text_len
            None,           # tokens
            all_captions,   # all_captions
            task,           # tasks
            fname           # fname
        )