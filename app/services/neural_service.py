import torch
from transformers import AutoModel, AutoTokenizer
from app.core.config import settings

class NeuralService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
    
    def load_model(self):
        try:
            if torch.backends.mps.is_available:
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise ValueError(e)
    
    def get_embedding(self, text: str):
        if self.model is None:
            self.load_model()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu().numpy()
    
neuro_service = NeuralService(settings.MODEL_NAME)
