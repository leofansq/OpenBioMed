import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Any, Dict, Optional, Tuple


from open_biomed.data import Molecule, Protein, Pocket
from open_biomed.datasets.base_dataset import BaseDataset, assign_split, featurize
from open_biomed.utils.config import Config
from open_biomed.utils.featurizer import Featurizer, Featurized

class SBDDDataset(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        self.molecules, self.proteins, self.pockets = [], [], []
        super(SBDDDataset, self).__init__(cfg, featurizer)
        
    def __len__(self) -> int:
        return len(self.molecules)
    
    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        if getattr(self.cfg, "pocket_only", False):
            return {
                "pocket": self.pockets[index],
                "label": self.molecules[index],
            }
        else:
            return {
                "protein": self.proteins[index],
                "label": self.molecules[index],
            }
        
# NOTE: This dataset is used for testing only!
class PoseBustersV2(BaseDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(PoseBustersV2, self).__init__(cfg, featurizer)
        
    def _load_data(self) -> None:
        self.molecules, self.proteins, self.pockets = [], [], []
        all_files = open(os.path.join(self.cfg.path, "posebusters_benchmark_set_ids.txt"), "r").readlines()
        for file in all_files:
            file = file.strip()
            self.proteins.append(Protein.from_pdb_file(os.path.join(self.cfg.path, "posebusters_benchmark_set", file, f"{file}_protein.pdb")))
            self.molecules.append(Molecule.from_sdf_file(os.path.join(self.cfg.path, "posebusters_benchmark_set", file, f"{file}_ligand.sdf")))
            self.pockets.append(Pocket.from_protein_ref_ligand(self.proteins[-1], self.molecules[-1]))
    
    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        if getattr(self.cfg, "pocket_only", False):
            return {
                "molecule": self.molecules[index],
                "pocket": self.pockets[index],
                "label": self.molecules[index],
            }
        else:
            return {
                "molecule": self.molecules[index],
                "protein": self.proteins[index],
                "label": self.molecules[index],
            }
        
    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        return (self, self, self)
    
    def __len__(self) -> int:
        return len(self.molecules)

class CrossDocked(SBDDDataset):
    def __init__(self, cfg: Config, featurizer: Featurizer) -> None:
        super(CrossDocked, self).__init__(cfg, featurizer)
        
    def _load_data(self) -> None:
        self.protein_ids = []
        self.split_indexes = {"train": [], "valid": [], "test": []}
        split_index = torch.load(open(os.path.join(
            self.cfg.path, 
            "split_by_name.pt"
        ), "rb"))
        cnt = 0
        for split in ["train", "test"]:
            for i, (protein_file, ligand_file) in enumerate(tqdm(split_index[split], desc=f"Loading {split} set")):
                if protein_file is None:
                    continue
                self.molecules.append(Molecule.from_sdf_file(os.path.join(self.cfg.path, "crossdocked_pocket10_with_protein", ligand_file)))
                self.pockets.append(Pocket.from_pdb_file(os.path.join(self.cfg.path, "crossdocked_pocket10_with_protein", protein_file), removeHs=getattr(self.cfg, "remove_hs", True)))
                self.pockets[-1].estimated_num_atoms = self.molecules[-1].get_num_atoms()
                assert self.pockets[-1].get_num_atoms() >= 1 and self.pockets[-1].estimated_num_atoms >= 1

                # Load the full protein
                orig_protein_file = protein_file.split("rec")[0] + "rec.pdb"
                if getattr(self.cfg, "preload_proteins", False):
                    self.proteins.append(Protein.from_pdb_file(os.path.join(self.cfg.path, "crossdocked_pocket10_with_protein", orig_protein_file)))
                else:
                    self.proteins.append(os.path.join(self.cfg.path, "crossdocked_pocket10_with_protein", orig_protein_file))
                protein_id = protein_file.split("/")[0] + "/" + "_".join(protein_file.split("/")[1].split("_")[:2])
                self.protein_ids.append(protein_id)
                if split == "train" and cnt <= 100:
                    self.split_indexes["valid"].append(cnt)
                else:
                    self.split_indexes[split].append(cnt)
                cnt += 1
                if split == "train" and cnt >= 200 and self.cfg.debug:
                    break

    @assign_split
    def split(self, split_cfg: Optional[Config] = None) -> Tuple[Any, Any, Any]:
        testfile2pocket = {}
        for i in self.split_indexes["test"]:
            testfile2pocket[self.protein_ids[i]] = self.pockets[i]
        attrs = ["molecules", "pockets"]
        ret = (
            self.get_subset(self.split_indexes["train"], attrs),
            self.get_subset(self.split_indexes["valid"], attrs),
            CrossDockedTestset(self.cfg, self.featurizer, testfile2pocket),
        )
        del self
        return ret
    
class CrossDockedTestset(Dataset):
    def __init__(self, cfg: Config, featurizer: Featurizer, testfile2pocket: Optional[Dict[str, Pocket]] = None) -> None:
        super(CrossDockedTestset, self).__init__()
        self.cfg = cfg
        self.featurizer = featurizer
        self.molecules, self.proteins, self.pockets = [], [], []
        self._load_data(testfile2pocket)

    def __len__(self) -> int:
        return len(self.molecules)
    
    @featurize
    def __getitem__(self, index) -> Dict[str, Featurized[Any]]:
        if getattr(self.cfg, "pocket_only", False):
            return {
                "pocket": self.pockets[index],
                "label": self.molecules[index],
            }
        else:
            return {
                "protein": self.proteins[index],
                "label": self.molecules[index],
            }
    
    def _load_data(self, testfile2pocket: Optional[Dict[str, Pocket]] = None) -> None:
        protein_files = []
        for dir in os.listdir(os.path.join(self.cfg.path, "test_set")):
            cur_samples = 0
            for file in sorted(os.listdir(os.path.join(self.cfg.path, "test_set", dir))):
                if file.endswith(".pdb"):
                    cur_samples += 1
                    self.proteins.append(Protein.from_pdb_file(os.path.join(self.cfg.path, "test_set", dir, file)))
                    protein_files.append(os.path.join(dir, file.rstrip("_rec.pdb")))
                elif file.endswith(".sdf"):
                    self.molecules.append(Molecule.from_sdf_file(os.path.join(self.cfg.path, "test_set", dir, file)))
            for i in reversed(range(cur_samples)):
                if testfile2pocket is None:
                    self.pockets.append(Pocket.from_protein_ref_ligand(self.proteins[-1-i], self.molecules[-1-i]))
                else:
                    self.pockets.append(testfile2pocket[protein_files[-1-i]])
                self.pockets[-1].estimated_num_atoms = self.molecules[-1-i].get_num_atoms()