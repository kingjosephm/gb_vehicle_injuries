import lightgbm as lgb
import pandas as pd
from typing import Dict


class Model:

    def __init__(self, data: pd.DataFrame, model_config: Dict):
        self.df = data
        self.model_config = model_config