import random
import numpy as np
import os
from os.path import join as pjoin
from .dataset_t2m import Text2MotionDataset


class Motion2TextDatasetTokenCustom(Text2MotionDataset):
    """
    Custom dataset for M2T task with pre-tokenized motion.
    Inherits from Text2MotionDataset to reuse data loading logic.
    Functionality matches Text2MotionDatasetEvalV3, with custom token path support.
    """

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        w_vectorizer,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        code_path="motion_tokens",
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer
        
        # Setup motion token directory (custom addition)
        # code_path can come either as named param or from kwargs
        if code_path is None:
            code_path = "motion_tokens"
            
        if os.path.isabs(code_path):
            self.motion_token_dir = code_path
        else:
            self.motion_token_dir = pjoin(data_root, code_path)

        print(f"[Motion2TextDatasetTokenCustom] Loading tokens from: {self.motion_token_dir}")
        
        # Filter samples that don't have corresponding token files
        # We need to keep name_list and length_arr in sync, and rebuild sorted order
        original_count = len(self.name_list)
        
        # Build filtered lists while maintaining correspondence
        filtered_name_list = []
        filtered_length_list = []
        for i, name in enumerate(self.name_list):
            token_file = pjoin(self.motion_token_dir, name + '.npy')
            if os.path.exists(token_file):
                filtered_name_list.append(name)
                filtered_length_list.append(self.length_arr[i])
        
        # Sort by length (required for pointer mechanism)
        if filtered_name_list:
            sorted_pairs = sorted(zip(filtered_name_list, filtered_length_list), key=lambda x: x[1])
            self.name_list = [x[0] for x in sorted_pairs]
            self.length_arr = np.array([x[1] for x in sorted_pairs])
        else:
            self.name_list = []
            self.length_arr = np.array([])
        
        # Reset pointer for the new filtered dataset
        self.pointer = 0  # Reset first
        if len(self.length_arr) > 0:
            self.pointer = np.searchsorted(self.length_arr, self.max_length)
        
        filtered_count = len(self.name_list)
        effective_count = len(self.name_list) - self.pointer
        print(f"[Motion2TextDatasetTokenCustom] Filtered {original_count - filtered_count} samples without token files. Remaining: {filtered_count}")
        print(f"[Motion2TextDatasetTokenCustom] Effective samples (after pointer): {effective_count}")

    def __getitem__(self, item):
        # Get text data (same as Text2MotionDatasetEvalV3)
        idx = self.pointer + item
        fname = self.name_list[idx]
        data = self.data_dict[fname]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Get all captions (same as EvalV3)
        all_captions = [
            text_dic['caption']
            for text_dic in text_list
        ]

        if len(all_captions) > 3:
            all_captions = all_captions[:3]
        elif len(all_captions) == 2:
            all_captions = all_captions + all_captions[0:1]
        elif len(all_captions) == 1:
            all_captions = all_captions * 3

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        
        # Text processing (same as EvalV3)
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
        
        # Random crop (same as EvalV3)
        m_length = motion.shape[0]
        coin = np.random.choice([False, False, True])
        if coin:
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
        
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        # Load motion tokens (custom addition for M2T task)
        # Token file is guaranteed to exist (filtered in __init__)
        token_file = pjoin(self.motion_token_dir, fname + '.npy')
        m_tokens = np.load(token_file)
        if len(m_tokens.shape) > 1:
            m_tokens = m_tokens.flatten()
        m_tokens_len = len(m_tokens)

        # Create M2T task definition
        task = {
            'class': 'm2t',
            'input': ['<Motion_Placeholder>'],
            'output': [caption]
        }

        # Return format matches EvalV3:
        # text, m_tokens, m_tokens_len, motion, length, word_embs, pos_ohot, text_len, tokens, all_captions, tasks, fname
        return (
            caption,            # text
            m_tokens,           # m_tokens (for M2T input)
            m_tokens_len,       # m_tokens_len
            motion,             # motion
            m_length,           # length
            word_embeddings,    # word_embs
            pos_one_hots,       # pos_ohot
            sent_len,           # text_len
            "_".join(tokens),   # tokens (string, same as EvalV3)
            all_captions,       # all_captions
            task,               # tasks (M2T task definition)
            fname               # fname
        )