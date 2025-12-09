import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

class Text2MotionDatasetTokenCustom(data.Dataset):
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
        **kwargs,
    ):
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        self.mean = mean
        self.std = std
        self.split = split
        
        # 路径设置
        split_file = pjoin(data_root, split + '.txt')
        self.text_dir = pjoin(data_root, 'texts')
        
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
            
        # 2. 加载 Text
        text_list = data['text']
        
        # 训练时随机选择一条文本，测试时通常也随机，或者在评估代码中处理多条
        text_data = random.choice(text_list)
        caption = text_data['caption']
        
        # 获取所有 caption (用于评估)
        all_captions = [x['caption'] for x in text_list]

        # 3. 构造返回值 (适配 MotionGPT3 collate_fn)
        # 即使是 M2T 任务，collate_fn 可能仍期望 motion tensor 的存在
        # 我们创建一个 dummy motion，形状为 (m_length, 263)，内容为 0
        dummy_motion = np.zeros((m_length, 263)) 

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
            dummy_motion,   # motion (ignored for training but needed for shape)
            m_length,       # m_length
            None,           # word_embs
            None,           # pos_ohot
            None,           # text_len
            None,           # tokens
            all_captions,   # all_captions
            task,           # tasks
            fname           # fname
        )