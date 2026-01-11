import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import json
import csv

from model import CustomTransformer
from tokenizer import SimpleTokenizer
from config import Config

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.tokens) - self.seq_len
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def read_file_content(file_path):
    """Read content from various file types"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Text-based files
        if ext in ['.txt', '.md', '.log', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.c', '.h', '.yaml', '.yml', '.ini', '.cfg']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # JSON files
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        
        # CSV files
        elif ext == '.csv':
            text = ""
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                for row in reader:
                    text += ' '.join(row) + '\n'
            return text
        
        # PDF files
        elif ext == '.pdf':
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    for page in pdf.pages:
                        text += page.extract_text() + '\n'
                return text
            except ImportError:
                print(f"  ⚠ PyPDF2 not installed. Skipping PDF: {os.path.basename(file_path)}")
                return ""
            except Exception as e:
                print(f"  ✗ Error reading PDF {os.path.basename(file_path)}: {e}")
                return ""
        
        # DOCX files
        elif ext == '.docx':
            try:
                import docx
                doc = docx.Document(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            except ImportError:
                print(f"  ⚠ python-docx not installed. Skipping DOCX: {os.path.basename(file_path)}")
                return ""
            except Exception as e:
                print(f"  ✗ Error reading DOCX {os.path.basename(file_path)}: {e}")
                return ""
        
        # Default: try reading as text
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    
    except Exception as e:
        print(f"  ✗ Error reading {os.path.basename(file_path)}: {e}")
        return ""


def load_all_files(data_path):
    """Load ALL files from a directory or single file"""
    all_text = ""
    skip_extensions = ['.pt', '.pkl', '.pyc', '.pth', '.bin', '.exe', '.dll', '.so', '.zip', '.tar', '.gz']
    
    if os.path.isdir(data_path):
        print(f"\n{'='*60}")
        print(f"LOADING NEW DATA FROM: {data_path}")
        print(f"{'='*60}")
        
        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                if ext not in skip_extensions:
                    all_files.append(file_path)
        
        if not all_files:
            print(f"ERROR: No readable files found in {data_path}")
            return None
        
        print(f"\nFound {len(all_files)} file(s) to process:\n")
        
        total_chars = 0
        for file_path in all_files:
            content = read_file_content(file_path)
            if content:
                all_text += content + "\n\n"
                char_count = len(content)
                total_chars += char_count
                
                rel_path = os.path.relpath(file_path, data_path)
                print(f"  ✓ {rel_path}: {char_count:,} characters")
            else:
                rel_path = os.path.relpath(file_path, data_path)
                print(f"  ○ {rel_path}: Skipped (empty or error)")
        
        print(f"\n{'='*60}")
        print(f"Total new data: {total_chars:,} characters")
        print(f"{'='*60}")
    
    elif os.path.isfile(data_path):
        print(f"\nLoading single file: {data_path}")
        all_text = read_file_content(data_path)
        print(f"✓ Loaded {len(all_text):,} characters")
    
    else:
        print(f"ERROR: Path not found: {data_path}")
        return None
    
    return all_text


def continue_training(model, tokenizer, train_loader, config, start_epoch=0):
    """Continue training an existing model"""
    device = config.DEVICE
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"CONTINUING TRAINING (Starting from epoch {start_epoch})")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {start_epoch + epoch + 1}')
        
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            mask = create_causal_mask(x.size(1), device)
            
            optimizer.zero_grad()
            output = model(x, mask)
            
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            
            optimizer.step()
            total_loss += loss.item()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'\nEpoch {start_epoch + epoch + 1} completed | Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_continued_epoch_{start_epoch + epoch + 1}.pt')
            torch.save({
                'epoch': start_epoch + epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.get_model_config()
            }, checkpoint_path)
            print(f'✓ Checkpoint saved: {checkpoint_path}')
    
    # Save updated model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.get_model_config(),
    }, config.MODEL_SAVE_PATH)
    
    # Save tokenizer (with updated vocabulary if new words were added)
    tokenizer_path = config.MODEL_SAVE_PATH.replace('.pt', '_tokenizer.pkl')
    tokenizer.save(tokenizer_path)
    
    print(f'\n{"="*60}')
    print(f'✓ Continued training completed!')
    print(f'✓ Updated model saved: {config.MODEL_SAVE_PATH}')
    print(f'✓ Updated tokenizer saved: {tokenizer_path}')
    print(f'{"="*60}\n')


def main():
    config = Config()
    
    print(f"\n{'='*60}")
    print(f"CONTINUE TRAINING - ADD NEW KNOWLEDGE")
    print(f"{'='*60}\n")
    
    # Check if model exists
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: No trained model found at {config.MODEL_SAVE_PATH}")
        print("Please train a model first using train.py")
        return
    
    # Load existing model
    print(f"Loading existing model from: {config.MODEL_SAVE_PATH}")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    
    model = CustomTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded successfully")
    
    # Load existing tokenizer
    tokenizer_path = config.MODEL_SAVE_PATH.replace('.pt', '_tokenizer.pkl')
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})")
    
    # Ask user for new data path
    print("\n" + "="*60)
    new_data_path = input("Enter path to new data folder (e.g., 'list2/'): ").strip()
    
    if not new_data_path:
        print("No path provided. Exiting.")
        return
    
    # Load new training data
    text = load_all_files(new_data_path)
    
    if text is None or len(text) < 100:
        print("\nERROR: Not enough new data to train!")
        return
    
    print(f"\n✓ Total new training data: {len(text):,} characters")
    
    # Update tokenizer with new words (if any)
    print("\nUpdating tokenizer with new vocabulary...")
    old_vocab_size = tokenizer.vocab_size
    
    # Build new vocabulary from combined old + new data
    # Note: This keeps old words and adds new ones
    tokenizer.build_vocab(text, max_vocab_size=config.VOCAB_SIZE)
    
    new_vocab_size = tokenizer.vocab_size
    print(f"✓ Vocabulary updated: {old_vocab_size:,} → {new_vocab_size:,} words")
    
    # Tokenize new data
    print("Tokenizing new data...")
    tokens = tokenizer.encode(text)
    print(f"✓ Total tokens: {len(tokens):,}")
    
    # Create dataset
    dataset = TextDataset(tokens, seq_len=config.SEQ_LENGTH)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    print(f"✓ Created {len(dataset):,} new training samples")
    print(f"✓ Batches per epoch: {len(train_loader):,}")
    
    # Display model info
    print(f"\nModel Info:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Size: {model.get_size_mb():.2f} MB")
    
    # Continue training
    continue_training(model, tokenizer, train_loader, config, start_epoch=0)
    
    print("\n✓ Your model now knows the new data!")
    print("✓ You can chat with it using: python chat.py")


if __name__ == "__main__":
    main()