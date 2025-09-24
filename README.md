# 🏥 Dental Claim Parser - Educational Demo

## Learn to Use AI Models Locally

This demo shows how to use a small AI model to read dental claim tables extracted from PDF documents and extract structured information. Different insurance companies use different formats and field names - the AI learns to handle this variety.

## 🎯 What You'll Learn

- How to run AI models on your computer (no internet needed!)
- How to use prompts to tell AI what data to extract
- How AI can read extracted PDF tables with different formats

## 📋 What You Need

- Python installed on your computer
- About 1GB of free memory

## 🚀 How to Run

### 1. Install Python Libraries
```bash
pip install torch transformers accelerate safetensors
```

### 2. Download the AI Model
```bash
# Install git-lfs if you don't have it
git lfs install

# Clone the model from Hugging Face
git clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct models/qwen2-0.5b

# Or download manually from: https://huggingface.co/Qwen/Qwen2-0.5B-Instruct

```

### 3. Run the Demo
```bash
python basic_parser.py
```

## 📊 Expected Output

```
🏥 Dental Claim Parser - Educational Demo
Using Local Qwen2-0.5B Language Model
==================================================
🤖 Loading Qwen2-0.5B model locally...
✅ Model loaded successfully in 1.92 seconds
💾 Memory usage: ~942MB (Float16 optimization)
📋 Sample dental claim data loaded
✅ Prompts prepared for model
🧠 Generating response from local model...
✅ Response generated in about 10 seconds

🎉 SUCCESS! Extracted structured data:
----------------------------------------
{
  "claim_number": "202524532740000",
  "provider_name": "John Doe",
  "patient_name": "Xwcfs",
  "patient_dob": "11/11/1111",
  "procedures": [
    {
      "item_number": "1",
      "submitted_code": "D7240",
      "description": "Removal of impacted tooth-completely bony",
      "submitted_amount": "$370.99",
      "plan_pay": "$370.99"
    },
    {
      "item_number": "2", 
      "submitted_code": "D7240",
      "description": "Removal of impacted tooth-completely bony",
      "submitted_amount": "$370.99",
      "plan_pay": "$370.99"
    }
  ]
}
----------------------------------------
📊 Extracted 2 dental procedures
✅ Demo completed successfully!
```

## 🧠 How It Works (Simple Explanation)

### 1. Load the AI Model
The computer loads a small AI model (like loading a program).

### 2. Create Smart Prompts
We use special prompts that work with different insurance formats:
- Tell AI exactly what output format we want
- Extract key data regardless of table layout and variable names

### 3. AI Reads the Table Data
The AI processes the raw table and finds:
- Patient information
- Provider/doctor details
- Claim numbers and references  
- Procedure codes and descriptions
- Financial amounts

### 4. Get Clean Results
The AI gives us back organized, clean data in a JSON file.

## ⏱️ Performance (on 12GB MacBook Air)

- **Model loading**: ~2 seconds
- **Processing one table**: ~14 seconds  
- **Memory used**: ~1GB
- **Accuracy**: Excellent on dental claims from various insurers

*Note: Performance varies by computer specs*

## 🔧 Technical Details

### Memory Optimization

- **Float16 Precision**: Reduces memory usage by 50%
- **Efficient Loading**: `low_cpu_mem_usage=True`
- **Token Caching**: `use_cache=True` for faster generation

### Generation Parameters

- **Greedy Decoding**: `do_sample=False` for consistent outputs
- **Token Limits**: `max_new_tokens=400` to prevent runaway generation
- **Context Length**: `max_length=2048` for input processing


## 🔗 Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [Qwen Models](https://huggingface.co/Qwen)

## 📝 License

This educational demo is provided for learning purposes and is licensed under the MIT License. Anyone is free to use, modify, and distribute this project for any purpose. 
