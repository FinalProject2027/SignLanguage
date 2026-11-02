import torch
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class HFSignTokenizer:
    def __init__(self, vocab_size=5000, max_seq_len=50):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
    def train_from_csv(self, csv_path):
        print(f"Training Hugging Face BPE Tokenizer from {csv_path}...")
        df = pd.read_csv(csv_path, sep="\t")
        texts = df['SENTENCE'].dropna().astype(str).tolist()
        
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        )
        self.tokenizer.train_from_iterator(texts, trainer)
        
        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]"))
            ]
        )
        
        self.tokenizer.enable_padding(
            pad_id=self.tokenizer.token_to_id("[PAD]"), 
            pad_token="[PAD]", 
            length=self.max_seq_len
        )
        self.tokenizer.enable_truncation(max_length=self.max_seq_len)
        
        self.tokenizer.save("../models/how2sign_bpe.json")
        print(f"Tokenizer trained and saved. Final vocab size: {self.tokenizer.get_vocab_size()}")
        
    def encode(self, texts):
        encoded = self.tokenizer.encode_batch(texts)
        ids = [enc.ids for enc in encoded]
        return torch.tensor(ids, dtype=torch.long)
        
    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode_batch(token_ids, skip_special_tokens=True)