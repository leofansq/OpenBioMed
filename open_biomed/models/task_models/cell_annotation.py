from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch

from open_biomed.data import Cell, Text
from open_biomed.models.base_model import BaseModel
from open_biomed.utils.collator import Collator
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class CellAnnotation(BaseModel, ABC):
    def __init__(self, model_cfg: Config) -> None:
        super().__init__(model_cfg)

    def _add_task(self) -> None:
        featurizer, collator = self.get_featurizer()
        self.supported_tasks["cell_annotation"] = {
            "forward_fn": self.forward,
            "predict_fn": self.predict,
            "featurizer": featurizer,
            "collator": collator,
        }
    
    @abstractmethod
    def forward(self) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def predict(self,
        cell: Featurized[Cell],
        class_texts: Featurized[Text],
        **kwargs,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_featurizer(self) -> Tuple[Featurizer, Collator]:
        raise NotImplementedError