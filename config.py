import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()
        
        # Model architecture
        self.VOCAB_SIZE = int(os.getenv('VOCAB_SIZE', 10000))
        self.D_MODEL = int(os.getenv('D_MODEL', 512))
        self.NUM_HEADS = int(os.getenv('NUM_HEADS', 8))
        self.NUM_LAYERS = int(os.getenv('NUM_LAYERS', 12))
        self.D_FF = int(os.getenv('D_FF', 2048))
        self.MAX_SEQ_LEN = int(os.getenv('MAX_SEQ_LEN', 512))
        self.DROPOUT = float(os.getenv('DROPOUT', 0.1))
        
        # Training settings
        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.0003))
        self.NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 10))
        self.GRADIENT_CLIP = float(os.getenv('GRADIENT_CLIP', 1.0))
        
        # Data settings
        self.TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/train.txt')
        self.SEQ_LENGTH = int(os.getenv('SEQ_LENGTH', 128))
        
        # Model save/load
        self.MODEL_SAVE_PATH = os.getenv('MODEL_SAVE_PATH', 'models/my_model.pt')
        self.CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', 'checkpoints/')
        
        # Generation settings
        self.TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
        self.MAX_GENERATION_LENGTH = int(os.getenv('MAX_GENERATION_LENGTH', 200))
        self.TOP_K = int(os.getenv('TOP_K', 50))
        self.TOP_P = float(os.getenv('TOP_P', 0.9))
        
        # Device settings
        device_setting = os.getenv('DEVICE', 'auto')
        if device_setting == 'auto':
            import torch
            self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.DEVICE = device_setting
    
    def get_model_config(self):
        return {
            'vocab_size': self.VOCAB_SIZE,
            'd_model': self.D_MODEL,
            'num_heads': self.NUM_HEADS,
            'num_layers': self.NUM_LAYERS,
            'd_ff': self.D_FF,
            'max_seq_len': self.MAX_SEQ_LEN,
            'dropout': self.DROPOUT
        }
    
    def display(self):
        print("="*60)
        print("MODEL CONFIGURATION")
        print("="*60)
        print(f"Vocabulary Size: {self.VOCAB_SIZE:,}")
        print(f"Embedding Dimension: {self.D_MODEL}")
        print(f"Attention Heads: {self.NUM_HEADS}")
        print(f"Transformer Layers: {self.NUM_LAYERS}")
        print(f"Feed-Forward Dimension: {self.D_FF}")
        print(f"Max Sequence Length: {self.MAX_SEQ_LEN}")
        print(f"Dropout: {self.DROPOUT}")
        print()
        print("TRAINING CONFIGURATION")
        print("="*60)
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Epochs: {self.NUM_EPOCHS}")
        print(f"Gradient Clip: {self.GRADIENT_CLIP}")
        print(f"Device: {self.DEVICE}")
        print("="*60)