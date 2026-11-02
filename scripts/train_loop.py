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
            
    # --- EVALUATION METRICS ON VALIDATION SET ---
    print(f"\nEvaluating {config_name} on Validation Set...")
    
    # Load the separate validation split
    val_dataset = How2SignDataset(data_dir="../data/processed/validation", config=config_name)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for val_features, val_targets in val_dataloader:
            # 1. Forward pass
            outputs = model(val_features) # Shape: (batch, seq_len, vocab_size)
            
            # 2. Get the most likely token IDs
            predicted_ids = torch.argmax(outputs, dim=-1) # Shape: (batch, seq_len)
            
            # 3. Decode token IDs back to English strings, stripping padding/EOS
            pred_strings = tokenizer.decode(predicted_ids)
            
            all_preds.extend(pred_strings)
            all_refs.extend(val_targets) # val_targets are already raw English strings
            
    # --- COMPUTE METRICS ---
    # BLEU expects references as a list of lists: [[ref1, ref2, ...]]
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    
    # WER and ROUGE evaluate sentence pairs directly
    wer = jiwer.wer(all_refs, all_preds)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(all_refs, all_preds)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    print(f"Validation Results for {config_name}:")
    print(f"BLEU: {bleu.score:.2f} | ROUGE-L: {avg_rouge:.4f} | WER: {wer:.4f}\n")
    
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