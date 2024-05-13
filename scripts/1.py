from transformers import DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LanguageModel(nn.Module):
    def __init__(self, model='../models/bge-micro-v2'):
        super(LanguageModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.model_name = model
        # Remove the CLIP vision tower
        if "clip" in self.model_name:
            self.model.vision_model = None
        # Freeze the pre-trained parameters (very important)
        for param in self.model.parameters():
            param.requires_grad = False

        # Make sure to set evaluation mode (also important)
        self.model.eval()

    def forward(self, text_batch):
        inputs = self.tokenizer(text_batch, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():  # Ensure no gradients are computed for this forward pass

            if "clip" in self.model_name:
                sentence_embedding = self.model.get_text_features(**inputs)
                return sentence_embedding

            outputs = self.model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        # Normalize embeddings
        sentence_embedding = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embedding

