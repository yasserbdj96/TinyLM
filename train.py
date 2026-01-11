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
        if ext in ['.txt', '.md', '.log', '.json', '.xml', '.html', '.css', '.js', '.py', '.java', '.cpp', '.c', '.h']:
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
        
        # PDF files (requires PyPDF2)
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
        
        # DOCX files (requires python-docx)
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
    
    # Skip these file types
    skip_extensions = ['.pt', '.pkl', '.pyc', '.pth', '.bin', '.exe', '.dll', '.so']
    
    # Check if it's a directory or file
    if os.path.isdir(data_path):
        print(f"\n{'='*60}")
        print(f"LOADING ALL FILES FROM: {data_path}")
        print(f"{'='*60}")
        
        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[1].lower()
                
                # Skip binary/model files
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
                
                # Show relative path
                rel_path = os.path.relpath(file_path, data_path)
                print(f"  ✓ {rel_path}: {char_count:,} characters")
            else:
                rel_path = os.path.relpath(file_path, data_path)
                print(f"  ○ {rel_path}: Skipped (empty or error)")
        
        print(f"\n{'='*60}")
        print(f"Total characters loaded: {total_chars:,}")
        print(f"{'='*60}")
    
    elif os.path.isfile(data_path):
        print(f"\nLoading single file: {data_path}")
        all_text = read_file_content(data_path)
        print(f"✓ Loaded {len(all_text):,} characters")
    
    else:
        print(f"ERROR: Path not found: {data_path}")
        return None
    
    return all_text


def train_model(model, train_loader, config, tokenizer):
    device = config.DEVICE
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRAINING STARTED")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}')
        
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
        print(f'\nEpoch {epoch+1} completed | Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config.get_model_config()
            }, checkpoint_path)
            print(f'✓ Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.get_model_config(),
    }, config.MODEL_SAVE_PATH)
    
    # Save tokenizer
    tokenizer_path = config.MODEL_SAVE_PATH.replace('.pt', '_tokenizer.pkl')
    tokenizer.save(tokenizer_path)
    
    print(f'\n{"="*60}')
    print(f'✓ Training completed!')
    print(f'✓ Model saved: {config.MODEL_SAVE_PATH}')
    print(f'✓ Tokenizer saved: {tokenizer_path}')
    print(f'{"="*60}\n')


def main():
    # Load configuration
    config = Config()
    config.display()
    
    # Load training data (supports ALL file types)
    text = load_all_files(config.TRAIN_DATA_PATH)
    
    if text is None or len(text) < 100:
        print("\n" + "="*60)
        print("ERROR: Not enough data to train!")
        print("="*60)
        print("\nPlease add files to the data folder.")
        print("Supported formats: .txt, .md, .json, .csv, .pdf, .docx, .py, .js, etc.")
        print("\nOptional: Install these for more formats:")
        print("  pip install PyPDF2        # For PDF files")
        print("  pip install python-docx   # For Word documents")
        return
    
    print(f"\n✓ Total training data: {len(text):,} characters")
    
    # Build tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(text, max_vocab_size=config.VOCAB_SIZE)
    print(f"✓ Vocabulary size: {tokenizer.vocab_size:,}")
    
    # Tokenize data
    print("Tokenizing data...")
    tokens = tokenizer.encode(text)
    print(f"✓ Total tokens: {len(tokens):,}")
    
    # Create dataset
    dataset = TextDataset(tokens, seq_len=config.SEQ_LENGTH)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    print(f"✓ Created {len(dataset):,} training samples")
    print(f"✓ Batches per epoch: {len(train_loader):,}")
    
    # Create model
    print("\nInitializing model...")
    model = CustomTransformer(**config.get_model_config())
    
    print(f"✓ Model parameters: {model.get_num_params():,}")
    print(f"✓ Model size: {model.get_size_mb():.2f} MB")
    
    # Train
    train_model(model, train_loader, config, tokenizer)


if __name__ == "__main__":
    main()