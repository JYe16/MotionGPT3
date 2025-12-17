from .HumanML3D import HumanML3DDataModule
from .humanml.dataset_m2t_token_custom import Motion2TextDatasetTokenCustom
from .humanml.dataset_t2m_eval_v3 import Text2MotionDatasetEvalV3

class HumanML3DDataModuleCustom(HumanML3DDataModule):
    def __init__(self, cfg, **kwargs):
        # 初始化父类，这会设置基本参数 (paths, mean/std 等)
        super().__init__(cfg, **kwargs)
        
        # 覆盖 Dataset 类选择逻辑
        # 支持 "token_custom" 和 "lm_token_custom" 两种 STAGE 名称
        if cfg.TRAIN.STAGE in ["token_custom", "lm_token_custom"]:
            print("[DataModule] Using Custom Token Dataset for M2T Training")
            self.Dataset = Motion2TextDatasetTokenCustom
            self.DatasetEval = Motion2TextDatasetTokenCustom

            # self.Dataset = Text2MotionDatasetEvalV3
            # self.DatasetEval = Text2MotionDatasetEvalV3
            
            # 确保一些参数正确传递
            self.hparams.code_path = cfg.DATASET.CODE_PATH
