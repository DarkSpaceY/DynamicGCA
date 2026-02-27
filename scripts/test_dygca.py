
import torch
from torch import nn
from dygca_model import DyGCAPlugin, DyGCAConfig
from transformers import AutoConfig, AutoModelForCausalLM

def test_dygca_flow():
    print("🚀 Starting DyGCA Flow Test...")
    
    # 1. Setup minimal config and model
    # Use a tiny model for fast testing
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"📦 Loading config for {model_id}...")
    base_config = AutoConfig.from_pretrained(model_id)
    
    # Create a dummy base model to avoid downloading/loading full weights if possible
    # But since DyGCAPlugin needs to call base_model, we'll initialize a small one on CPU
    print("🏗️ Initializing base model (CPU)...")
    base_model = AutoModelForCausalLM.from_config(base_config)
    
    dygca_config = DyGCAConfig(
        vocab_size=base_config.vocab_size,
        hidden_size=base_config.hidden_size,
        k_focuses=4,      # Small number for testing
        m_selection=2,
        distribution="beta"
    )
    
    print("🔌 Initializing DyGCA Plugin...")
    model = DyGCAPlugin(base_model, dygca_config)
    model.eval()
    
    # 2. Prepare dummy batch
    bsz = 2
    seq_len = 16
    input_ids = torch.randint(0, base_config.vocab_size, (bsz, seq_len))
    attention_mask = torch.ones((bsz, seq_len))
    labels = input_ids.clone()
    
    print(f"input_ids shape: {input_ids.shape}")
    
    # 3. Run forward pass
    print("🏃 Running Forward Pass...")
    try:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        print("✅ Forward Pass Successful!")
        print(f"Output keys: {outputs.keys()}")
        print(f"Total Loss: {outputs['loss'].item():.4f}")
        print(f"LM Loss: {outputs['lm_loss'].item():.4f}")
        print(f"Diversity Loss: {outputs['diversity_loss'].item():.4f}")
        print(f"Logits shape: {outputs['logits'].shape}")
        
    except Exception as e:
        print(f"❌ Forward Pass Failed!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dygca_flow()
