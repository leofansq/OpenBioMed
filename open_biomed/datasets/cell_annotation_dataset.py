from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Cell, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized
from datasets import load_from_disk

class CellAnnotationDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.cells, self.labels = [], []
        self.class_texts = {}
        super(CellAnnotationDataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.cells)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        return {
            "cell": self.cells[index],
            "class_text": self.class_texts[index], 
            "label": self.labels[index],
        }

class CellAnnotation(CellAnnotationDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(CellAnnotation, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        dataset = load_from_disk(os.path.join(self.cfg.path, f"data.dataset"))
        class_texts = json.load(open(os.path.join(self.cfg.path, f"type2text.json"), "r"))
        
        for sample in dataset:
            self.cells.append(Cell.from_sequence(sample["input_ids"]))
            self.labels.append(Text.from_str(sample["celltype"]))
        self.class_texts = class_texts