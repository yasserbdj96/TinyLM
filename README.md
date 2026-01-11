# Tiny Language Model

A lightweight, customizable transformer-based language model built from scratch using PyTorch. Train your own AI assistant on any text data in minutes!

## 🌟 Features

- **Fast Training**: Optimized configuration trains in minutes, not hours
- **Multi-Format Support**: Train on `.txt`, `.md`, `.json`, `.csv`, `.pdf`, `.docx`, `.py`, `.js`, and more
- **Incremental Learning**: Add new knowledge to existing models without retraining from scratch
- **Interactive Chat**: Chat with your trained model in real-time
- **Customizable Architecture**: Easily adjust model size, layers, attention heads, and more
- **GPU Acceleration**: Automatic CUDA support for faster training

## 📋 Requirements

### Core Dependencies
```bash
torch>=2.0.0
tqdm>=4.65.0
python-dotenv>=1.0.0
```

### Optional (for additional file formats)
```bash
PyPDF2>=3.0.0          # For PDF files
python-docx>=0.8.11    # For Word documents (.docx)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the repository
git clone https://github.com/yasserbdj96/TinyLM.git
cd TinyLM

# Install dependencies
pip install -r requirements.txt

# For CUDA support (recommended):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision torchaudio
```

### 2. Prepare Your Data

Create a `data/` folder and add your training files:

```bash
mkdir data
# Add your .txt, .pdf, .docx, .json, .csv, or other files to data/
```

**Supported file formats:**
- Text: `.txt`, `.md`, `.log`
- Code: `.py`, `.js`, `.java`, `.cpp`, `.c`, `.h`
- Documents: `.pdf`, `.docx`, `.json`, `.csv`
- Web: `.html`, `.xml`, `.css`
- Config: `.yaml`, `.yml`, `.ini`, `.cfg`

### 3. Train Your Model

```bash
python train.py
```

The script will:
- Load all files from the `data/` folder
- Build a tokenizer from your data
- Train a transformer model
- Save the model and tokenizer to `models/`

**Training Progress:**
```
Loading ALL FILES FROM: data/
Found 5 file(s) to process:

  ✓ document1.txt: 15,432 characters
  ✓ notes.md: 8,921 characters
  ✓ data.json: 12,045 characters
  
Total characters loaded: 36,398

Building tokenizer...
✓ Vocabulary size: 8,000
✓ Total tokens: 9,234
✓ Created 9,106 training samples

Epoch 1/3: 100%|████████████████████████| 285/285 [00:45<00:00, 6.31it/s, loss=2.3421]
```

### 4. Chat with Your Model

```bash
python chat.py
```

Example conversation:
```
You: Hello! How are you?
AI: I'm doing great! How can I help you today?

You: What do you know about machine learning?
AI: Machine learning is a subset of artificial intelligence...
```

### 5. Add New Knowledge (Optional)

To teach your model new information without starting from scratch:

```bash
python continue_training.py
```

Then enter the path to your new data:
```
Enter path to new data folder (e.g., 'list2/'): new_documents/
```

## ⚙️ Configuration

Edit `.env` file to customize your model:

### Model Architecture
```bash
D_MODEL=384           # Embedding dimension
NUM_LAYERS=6          # Number of transformer layers
D_FF=1536            # Feed-forward dimension
NUM_HEADS=8          # Number of attention heads
VOCAB_SIZE=8000      # Vocabulary size
MAX_SEQ_LEN=512      # Maximum sequence length
```

### Training Settings
```bash
BATCH_SIZE=32        # Larger = faster training (if you have GPU memory)
LEARNING_RATE=0.0003 # Higher = faster convergence
NUM_EPOCHS=3         # Number of training epochs
SEQ_LENGTH=128       # Training sequence length (shorter = faster)
```

### Performance Presets

**Fast & Small** (minutes to train):
```bash
D_MODEL=256
NUM_LAYERS=4
BATCH_SIZE=64
SEQ_LENGTH=64
NUM_EPOCHS=2
```

**Balanced** (default):
```bash
D_MODEL=384
NUM_LAYERS=6
BATCH_SIZE=32
SEQ_LENGTH=128
NUM_EPOCHS=3
```

**Large & High Quality** (slower, better results):
```bash
D_MODEL=512
NUM_LAYERS=12
BATCH_SIZE=16
SEQ_LENGTH=256
NUM_EPOCHS=5
```

## 📁 Project Structure

```
TinyLM/
│
├── train.py              # Main training script
├── chat.py               # Interactive chat interface
├── continue_training.py  # Add new knowledge to existing model
├── model.py             # Transformer architecture
├── tokenizer.py         # Custom tokenizer
├── config.py            # Configuration loader
├── .env                 # Configuration file
├── requirements.txt     # Dependencies
│
├── data/                # Training data folder
│   ├── file1.txt
│   ├── file2.pdf
│   └── ...
│
├── models/              # Saved models (auto-created)
│   ├── personal_assistant.pt
│   └── personal_assistant_tokenizer.pkl
│
└── checkpoints/         # Training checkpoints (auto-created)
    └── checkpoint_epoch_1.pt
```

## 🛠️ Advanced Usage

### Custom Training Data Path

Edit `.env`:
```bash
TRAIN_DATA_PATH=my_custom_data/
```

Or specify a single file:
```bash
TRAIN_DATA_PATH=my_document.txt
```

### Adjusting Generation Quality

Edit `.env`:
```bash
TEMPERATURE=0.7      # Lower = more focused, Higher = more creative
MAX_GENERATION_LENGTH=150  # Maximum response length
TOP_K=40            # Top-k sampling
TOP_P=0.9           # Nucleus sampling threshold
```

### Using Different Model Sizes

**Tiny Model** (~50MB):
```bash
D_MODEL=256
NUM_LAYERS=4
D_FF=1024
```

**Small Model** (~150MB, default):
```bash
D_MODEL=384
NUM_LAYERS=6
D_FF=1536
```

**Medium Model** (~300MB):
```bash
D_MODEL=512
NUM_LAYERS=8
D_FF=2048
```

**Large Model** (~600MB):
```bash
D_MODEL=768
NUM_LAYERS=12
D_FF=3072
```

## 🔧 Troubleshooting

### Out of Memory Error

Reduce these settings in `.env`:
```bash
BATCH_SIZE=8         # Reduce batch size
SEQ_LENGTH=64        # Reduce sequence length
D_MODEL=256          # Use smaller model
```

### Training Too Slow

Increase these settings:
```bash
BATCH_SIZE=64        # Larger batches
SEQ_LENGTH=64        # Shorter sequences
NUM_EPOCHS=2         # Fewer epochs
```

### Poor Quality Responses

Increase these settings:
```bash
NUM_EPOCHS=5         # More training
D_MODEL=512          # Larger model
NUM_LAYERS=8         # More layers
VOCAB_SIZE=15000     # Larger vocabulary
```

### Model Not Learning

- Ensure you have enough training data (at least 10,000 characters)
- Increase `NUM_EPOCHS`
- Try lowering `LEARNING_RATE` to 0.0001
- Check that your data is being loaded correctly

## 📊 Performance Benchmarks

Approximate training times (on RTX 3080):

| Config | Model Size | 10k chars | 100k chars | 1M chars |
|--------|-----------|-----------|------------|----------|
| Fast   | ~100MB    | 2 min     | 10 min     | 90 min   |
| Balanced | ~150MB  | 5 min     | 20 min     | 3 hours  |
| Large  | ~600MB    | 15 min    | 60 min     | 10 hours |

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

Built with:
- PyTorch
- Transformer architecture (Vaswani et al., 2017)
- Modern deep learning best practices

## 💡 Tips for Best Results

1. **Data Quality Matters**: Clean, well-formatted text produces better results
2. **More Data = Better Model**: Aim for at least 100,000 characters
3. **Domain-Specific Training**: Train on data similar to what you want to chat about
4. **Experiment with Settings**: Try different configurations to find what works best
5. **Use GPU**: Training on GPU is 10-50x faster than CPU
6. **Start Small**: Test with a small model first, then scale up

## 📧 Support

If you encounter issues or have questions, please open an issue on GitHub.

---

**Happy Training! 🚀**