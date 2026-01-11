import torch
from model import CustomTransformer
from tokenizer import SimpleTokenizer
from config import Config
import os

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def generate_text(model, tokenizer, prompt, config, device):
    model.eval()
    tokens = tokenizer.encode(prompt)
    
    generated_tokens = tokens.copy()
    
    with torch.no_grad():
        for _ in range(config.MAX_GENERATION_LENGTH):
            # Get last MAX_SEQ_LEN tokens
            input_tokens = generated_tokens[-config.MAX_SEQ_LEN:]
            x = torch.tensor([input_tokens], dtype=torch.long, device=device)
            mask = create_causal_mask(x.size(1), device)
            
            output = model(x, mask)
            logits = output[0, -1, :] / config.TEMPERATURE
            
            # Top-k filtering
            if config.TOP_K > 0:
                top_k_values, top_k_indices = torch.topk(logits, config.TOP_K)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_values
            
            # Top-p (nucleus) filtering
            if config.TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > config.TOP_P
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            generated_tokens.append(next_token)
            
            # Stop if EOS token (index 2)
            if next_token == 2:
                break
    
    return tokenizer.decode(generated_tokens)


def main():
    config = Config()
    device = config.DEVICE
    
    print(f"\n{'='*60}")
    print(f"AI CHAT INTERFACE")
    print(f"{'='*60}\n")
    
    # Load model
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first using train.py")
        return
    
    print(f"Loading model from: {config.MODEL_SAVE_PATH}")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device)
    
    model = CustomTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    
    # Load tokenizer
    tokenizer_path = config.MODEL_SAVE_PATH.replace('.pt', '_tokenizer.pkl')
    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer not found at {tokenizer_path}")
        return
    
    tokenizer = SimpleTokenizer()
    tokenizer.load(tokenizer_path)
    print(f"✓ Tokenizer loaded successfully")
    print(f"✓ Using device: {device}")
    
    print(f"\n{'='*60}")
    print("Type your message and press Enter. Type 'quit' to exit.")
    print(f"{'='*60}\n")
    
    conversation_history = ""
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        # Add to conversation history
        conversation_history += f"User: {user_input}\nAI:"
        
        # Generate response
        print("AI: ", end="", flush=True)
        response = generate_text(model, tokenizer, conversation_history, config, device)
        
        # Extract only the AI's response
        ai_response = response[len(conversation_history):].split("User:")[0].strip()
        
        print(ai_response)
        
        # Update conversation history
        conversation_history += f" {ai_response}\n"
        
        # Keep conversation history manageable
        if len(conversation_history) > 2000:
            lines = conversation_history.split('\n')
            conversation_history = '\n'.join(lines[-10:])


if __name__ == "__main__":
    main()