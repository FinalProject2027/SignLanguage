import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sacrebleu
import jiwer
from rouge_score import rouge_scorer
from pathlib import Path

from train_dataset import How2SignDataset
from train_model import SignTranslationModel
from tokenizer import HFSignTokenizer

ROOT = Path(__file__).resolve().parent.parent

def train_and_export_ablation(config_name, input_features, vocab_size=5000):
    print(f"\n--- Starting Ablation: {config_name} ({input_features} features) ---")
    
    # 1. Initialize and train the Tokenizer
    tokenizer = HFSignTokenizer(vocab_size=vocab_size, max_seq_len=50)
    csv_path = ROOT / "data" / "raw" / "train" / "how2sign_train.csv"
    tokenizer.train_from_csv(csv_path)
    
    # 2. Setup Dataset and Model
    train_dir = ROOT / "data" / "processed" / "train"
    dataset = How2SignDataset(data_dir=train_dir, config=config_name)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = SignTranslationModel(input_dim=input_features, vocab_size=vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    pad_idx = tokenizer.tokenizer.token_to_id("[PAD]")
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # --- TRAINING LOOP ---
    model.train()
    epochs = 5 
    for epoch in range(epochs): 
        epoch_loss = 0
        for batch_features, target_texts in dataloader:
            optimizer.zero_grad()
            target_tokens = tokenizer.encode(target_texts) 
            outputs = model(batch_features) 
            loss = criterion(outputs.view(-1, vocab_size), target_tokens.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
            
    # --- EVALUATION METRICS ON VALIDATION SET ---
    print(f"\nEvaluating {config_name} on Validation Set...")
    val_dir = ROOT / "data" / "processed" / "validation"
    val_dataset = How2SignDataset(data_dir=val_dir, config=config_name)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for val_features, val_targets in val_dataloader:
            outputs = model(val_features)
            predicted_ids = torch.argmax(outputs, dim=-1)
            pred_strings = tokenizer.decode(predicted_ids)
            all_preds.extend(pred_strings)
            all_refs.extend(val_targets)
            
            
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs])
    wer = jiwer.wer(all_refs, all_preds)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(all_refs, all_preds)]
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0
    
    print(f"Validation Results for {config_name}:")
    print(f"BLEU: {bleu.score:.2f} | ROUGE-L: {avg_rouge:.4f} | WER: {wer:.4f}\n")
    
    # --- EXPORT ---
    pth_path = ROOT / "models" / f"{config_name}_model.pth"
    torch.save(model.state_dict(), pth_path)
    
    dummy_input = torch.randn(1, 60, input_features)
    onnx_path = ROOT / "models" / f"{config_name}_model.onnx"
    
    torch.onnx.export(model, dummy_input, str(onnx_path), 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print(f"Exported {config_name} to {onnx_path}")

if __name__ == "__main__":
    train_and_export_ablation("hands", 126)
    train_and_export_ablation("hands_pose", 201)
    train_and_export_ablation("full", 321)