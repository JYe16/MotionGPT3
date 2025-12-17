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
        # 注意：由于我们使用预量化的 token，只能使用 f_tag=0, to_tag=0 的完整 motion
        # f_tag/to_tag 裁剪需要实时重新编码 token，这里不支持
        new_name_list = []
        data_dict = {}
        
        print(f"Loading {split} dataset...")
        for name in tqdm(self.id_list):
            try:
                # 1. 检查 Token 文件
                token_file = pjoin(self.motion_token_dir, name + '.npy')
                if not os.path.exists(token_file):
                    continue
                
                # 2. 加载 Tokens
                tokens = np.load(token_file)
                if len(tokens.shape) > 1:
                    tokens = tokens.flatten()

                # 3. 基于 token 长度过滤 (10-49 tokens -> 3544 samples)
                token_len = len(tokens)
                min_token_len = 10  # 对应约 40 帧 motion (40//4=10)
                max_token_len = self.max_motion_length // self.unit_length  # 196//4=49
                if token_len < min_token_len or token_len > max_token_len:
                    continue

                # 4. 加载文本，只保留 f_tag=0, to_tag=0 的
                text_path = pjoin(self.text_dir, name + '.txt')
                if not os.path.exists(text_path):
                    continue
                    
                text_data = []
                with cs.open(text_path) as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        if len(line_split) < 4: 
                            continue
                        tokens_text = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        
                        # 只保留完整 motion 的文本标注
                        if f_tag == 0.0 and to_tag == 0.0:
                            text_dict = {
                                'caption': caption,
                                'tokens': tokens_text
                            }
                            text_data.append(text_dict)

                if len(text_data) > 0:
                    data_dict[name] = {
                        'tokens': tokens,
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
        
        # 1. 加载 Motion Tokens (已经在 __init__ 中裁剪好了)
        m_tokens = data['tokens']
        if len(m_tokens.shape) > 1:
            m_tokens = m_tokens.flatten()
        
        # 截断处理
        m_tokens_len = len(m_tokens)
        if m_tokens_len > self.max_motion_length:
            m_tokens = m_tokens[:self.max_motion_length]
            m_tokens_len = self.max_motion_length

        # 2. 加载 Text
        text_list = data['text']
        
        # 获取所有 caption (用于评估) - 与 eval_v3 相同的处理逻辑
        all_captions = [text_dic['caption'] for text_dic in text_list]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # 随机选择一条文本
        text_data = random.choice(text_list)
        caption = text_data['caption']
        tokens = text_data['tokens']
        
        # Text processing - 与 eval_v3 完全相同
        max_text_len = 20
        if len(tokens) < max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # 3. 加载 Motion (用于评估)
        motion_file = pjoin(self.motion_dir, fname + '.npy')
        if os.path.exists(motion_file):
            motion = np.load(motion_file)
            m_length = motion.shape[0]
            
            # 对齐到 unit_length 的倍数
            m_length = (m_length // self.unit_length) * self.unit_length
            motion = motion[:m_length]

            # Z Normalization
            motion = (motion - self.mean) / self.std
        else:
            # Fallback
            motion = np.zeros((m_tokens_len * 4, 263))
            m_length = m_tokens_len * 4

        # 返回元组顺序与 eval_v3 完全相同:
        # caption, m_tokens, m_tokens_len, motion, m_length, word_embs, pos_ohot, text_len, tokens, all_captions, tasks, fname
        # 注意: 这里返回 m_tokens 和 m_tokens_len 供 M2T 任务使用
        
        # M2T 任务定义
        task = {
            "class": "m2t",
            "input": ["Generate text: <Motion_Placeholder>"],
            "output": ["<Caption_Placeholder>"]
        }
        
        return caption, m_tokens, m_tokens_len, motion, m_length, word_embeddings, pos_one_hots, sent_len, "_".join(
            tokens), all_captions, task, fname