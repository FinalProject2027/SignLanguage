import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sacrebleu
import jiwer
from rouge_score import rouge_scorer

from train_dataset import How2SignDataset
from train_model import SignTranslationModel
from tokenizer import HFSignTokenizer

def train_and_export_ablation(config_name, input_features, vocab_size=5000):
    print(f"\n--- Starting Ablation: {config_name} ({input_features} features) ---")
    
    # 1. Initialize and train the Tokenizer
    tokenizer = HFSignTokenizer(vocab_size=vocab_size, max_seq_len=50)
    csv_path = "../data/raw/train/how2sign_train.csv"
    tokenizer.train_from_csv(csv_path)
    
    # 2. Setup Dataset and Model
    dataset = How2SignDataset(data_dir="../data/processed/train", config=config_name)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SignTranslationModel(input_dim=input_features, vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dynamically pull the correct padding index from the tokenizer to ignore during loss
    pad_idx = tokenizer.tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # --- TRAINING LOOP ---
    model.train()
    epochs = 5 
    for epoch in range(epochs): 
        epoch_loss = 0
        for batch_features, target_texts in dataloader:
            optimizer.zero_grad()
            
            # 1. Tokenization: Convert raw text to padded tensors
            target_tokens = tokenizer.encode(target_texts) 
            
            # 2. Forward pass: Feed the 126, 201, or 321 features to the LSTM
            outputs = model(batch_features) 
            
            # 3. Compute Loss: Flatten predictions and targets for CrossEntropy
            loss = criterion(outputs.view(-1, vocab_size), target_tokens.view(-1))
            
            # 4. Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    # --- EVALUATION METRICS BASELINE ---
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # MEMBER 2 TASK: Write validation evaluation loop here using tokenizer.decode(predictions)
    
    # --- EXPORT ---
    pth_path = f"../models/{config_name}_model.pth"
    torch.save(model.state_dict(), pth_path)
    
    dummy_input = torch.randn(1, 60, input_features)
    onnx_path = f"../models/{config_name}_model.onnx"
    
    torch.onnx.export(model, dummy_input, onnx_path, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Exported {config_name} to {onnx_path}")

if __name__ == "__main__":
    train_and_export_ablation("hands", 126)
    train_and_export_ablation("hands_pose", 201)
    train_and_export_ablation("full", 321)