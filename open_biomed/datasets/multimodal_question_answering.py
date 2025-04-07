from typing import Tuple, Union, Any, Dict, Optional, List
from typing_extensions import Self

import logging
import json
import os
from torch.utils.data import Dataset

from open_biomed.data import Molecule, Protein, Text
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class MultimodalQADataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.proteins, self.texts, self.labels = [], [], [], []
        super(MultimodalQADataset, self).__init__(cfg, featurizer)

    def __len__(self) -> int:
        return len(self.texts)

    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:

        text_prefix = ""
        molecules = []
        proteins = []

        if self.molecules[index]:
            mol = self.molecules[index]
            molecules.append(mol)
            text_prefix += f"<molecule><representation><moleculeHere></representation><SMILES>{mol.smiles}</SMILES></molecule> "

        if self.proteins[index]:
            proteins.append(self.proteins[index])
            text_prefix += "<protein><proteinHere></protein> "

        return {
            "molecule": molecules,
            "protein": proteins,
            "text": Text.from_str(text_prefix + self.texts[index].str), 
            "label": self.labels[index],
        }

class MultimodalQA(MultimodalQADataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(MultimodalQA, self).__init__(cfg, featurizer)

    def _load_data(self) -> None:
        self.split_indexes = {}
        cnt = 0
        for split in ["train", "val", "test"]:
            self.split_indexes[split] = []
            try:
                with open(os.path.join(self.cfg.path, f"{split}.json"), "r") as f:
                    cur = 0
                    sample_list = json.load(f)
                    for sample in sample_list:
                        # Getting the length of a list could be slow
                        if len(sample["smiles"]) > 1 or len(sample["sequence"]) > 1:
                            continue
                        self.split_indexes[split].append(cnt)
                        cnt += 1
                        cur += 1

                        if sample["smiles"][0]:
                            self.molecules.append(Molecule.from_smiles(sample["smiles"][0]))
                        else:
                            self.molecules.append(None)

                        if sample["sequence"][0]:
                            self.proteins.append(Protein.from_fasta(sample["sequence"][0]))
                        else:
                            self.proteins.append(None)
                        self.texts.append(Text.from_str(sample["question"]))
                        self.labels.append(Text.from_str(sample["answer"]))
                        if (split != "train" and cur >= 50 or cur >= 5000) and self.cfg.debug:
                            break
            except:
                warning_path = os.path.join(self.cfg.path, f"{split}.json")
                logging.warning(f"Can not open '{warning_path}', skip {split} !")
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        attrs = ["molecules", "proteins", "texts", "labels"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs), 
            self.get_subset(self.split_indexes["val"], attrs),
            self.get_subset(self.split_indexes["test"], attrs),
        )
        del self
        return ret
