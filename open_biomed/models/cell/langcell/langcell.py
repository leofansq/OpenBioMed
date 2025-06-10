import os
from typing import Any, Dict, List, Optional, Tuple
import torch, json
import torch.nn as nn, torch.nn.functional as F
from transformers import PreTrainedTokenizer, BertModel, BertTokenizer
from open_biomed.data import Cell, Text
from open_biomed.models.cell.langcell.langcell_utils import BertModel as MedBertModel, LangCellDataCollatorForCellClassification as DataCollatorForCellClassification
from open_biomed.utils.config import Config
from open_biomed.utils.collator import Collator
from open_biomed.utils.featurizer import Featurizer, Featurized
from open_biomed.models.task_models.cell_annotation import CellAnnotation

class LangCellFeaturizer(Featurizer):
    def __init__(self, 
                 text_tokenizer: PreTrainedTokenizer):
        self.text_tokenizer = text_tokenizer
    def __call__(self, 
                 cell: Cell, 
                 label: int, 
                 class_texts: List[Text]) -> Dict[str, Any]:
        class_texts = [text.str for text in class_texts]
        return {'cell': cell.sequence,
                'label': label,
                'class_texts': self.text_tokenizer(class_texts, padding=True, truncation=True, max_length=512, add_special_tokens=False, return_tensors='pt')}
    def get_attrs(self) -> List[str]:
        return ["cell", "label", "class_texts"]


class LangCell_Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output

class LangCell(CellAnnotation):
    def __init__(self, model_cfg: Config):
        super().__init__(model_cfg)

        self.cell_encoder = BertModel.from_pretrained(model_cfg.cell_model)
        self.cell_encoder.pooler = LangCell_Pooler(self.cell_encoder.config, pretrained_proj=model_cfg.cell_proj, proj_dim=256)
        proj = self.cell_encoder.pooler.proj

        self.tokenizer = BertTokenizer.from_pretrained(model_cfg.text_model)
        self.tokenizer.add_special_tokens({'bos_token':'[DEC]'})
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
        self.tokenizer.enc_token_id =  self.tokenizer.additional_special_tokens_ids[0] 

        self.text_encoder = MedBertModel.from_pretrained(model_cfg.text_model, add_pooling_layer=True)
        self.text_encoder.pooler = LangCell_Pooler(self.text_encoder.config, pretrained_proj=model_cfg.text_proj, proj_dim=256)

        self.ctm_head = nn.Linear(self.text_encoder.config.hidden_size, 2)
        self.ctm_head.load_state_dict(torch.load(model_cfg.ctm_head))

        for parent in reversed(type(self).__mro__[1:-1]):
            if hasattr(parent, '_add_task'):
                parent._add_task(self)

        self.text_input_ids_cache = None
        self.text_embs_cache = None

    def get_featurizer(self) -> Tuple[Featurizer, Collator]:
        return LangCellFeaturizer(self.tokenizer), DataCollatorForCellClassification()

    def encode_text(self, text):
        # print(text)
        text = self.text_encoder(**text).pooler_output
        # text = F.normalize(model.text_projector(text))
        return text

    def encode_cell(self, cell_input_ids):
        cell = self.cell_encoder(cell_input_ids)
        cell_last_h = cell.last_hidden_state
        cell_pooler = cell.pooler_output
        return cell_last_h, cell_pooler

    def ctm(self, text, cell_emb):
        output = self.text_encoder(**text,
                    encoder_hidden_states = cell_emb,
                    return_dict = True,
                    mode = 'multimodal',
                    )
        # print(output.last_hidden_state.shape)
        logits = self.ctm_head(output.last_hidden_state[:, 0, :])
        logits = F.softmax(logits, dim=-1)[..., 1] # [n]
        return logits
    
    def load_ckpt(self, ckpt):
        pass

    def forward(self):
        raise NotImplementedError

    def predict(self, 
                cell: Featurized[Cell], 
                class_texts: Featurized[Text],
                **kwargs) -> torch.Tensor:
        if self.text_input_ids_cache is not None and class_texts['input_ids'].equal(self.text_input_ids_cache):
            text_embs = self.text_embs_cache
        else:
            text_embs = self.encode_text(class_texts).T.to(cell.device)
            self.text_input_ids_cache = class_texts['input_ids']
            self.text_embs_cache = text_embs

        cell_last_h, cellemb = self.encode_cell(cell) # batchsize * 256
        sim = (cellemb @ text_embs) / 0.05 # batchsize * 161
        sim_logit = F.softmax(sim, dim=-1)
        # print(sim.shape)
        # ctm
        ctm_logit = torch.zeros_like(sim_logit)
        for text_idx in range(len(class_texts['input_ids'])):
            # text_list = [text] * sim_logit.shape[0]
            text_list = {k: v[text_idx].repeat(sim_logit.shape[0], 1) for k, v in class_texts.items()}
            # print(text_list['input_ids'].shape)
            ctm_logit[:, text_idx] = self.ctm(text_list, cell_last_h)
        ctm_logit = F.softmax(ctm_logit, dim=-1)

        logit = 0.1 * sim_logit + 0.9 * ctm_logit
        pred = logit.argmax(dim=-1)

        return pred