# 🏥 Dental Claim Parser - Educational Demo

## Learn to Use AI Models Locally on Apple Silicon

This demo shows how to use a local AI model (Qwen3-4B) to read dental claim tables extracted from PDF documents and extract structured information. Different insurance companies use different formats and field names - the AI learns to handle this variety.

Key highlights:
- **100% local processing** - No cloud APIs, no internet required
- **Apple Silicon optimized** - Uses MLX framework for fast inference on M1/M2/M3
- **Small memory footprint** - Only ~2-3GB RAM with 4-bit quantization
- **Prompt engineering insights** - Learn what works for structured data extraction

## 🎯 What You'll Learn

- How to run 4B parameter AI models efficiently on Apple Silicon
- How to design effective prompts for structured data extraction
- How AI can parse complex tables with varying formats from different insurers
- MLX framework basics for local LLM deployment

## 📋 What You Need

- **Apple Silicon Mac** (M1/M2/M3) - Required for MLX framework
- **Python 3.8+** installed on your computer
- **~3-4GB of free RAM** (2-3GB for model, 1GB for processing)
- **~3GB disk space** for the model files

## 🚀 How to Run

### 1. Install Python Libraries
```bash
pip install mlx mlx-lm
```

### 2. Download the AI Model

**Option A: Using Hugging Face CLI (Recommended)**
```bash
# Install Hugging Face CLI if you don't have it
pip install huggingface-hub

# Download the MLX-optimized Qwen3-4B model (4-bit quantized)
huggingface-cli download mlx-community/Qwen3-4B-Instruct-4bit \
    --local-dir models/qwen3-4b-instruct-4bit
```

**Option B: Using LM Studio**
1. Download LM Studio from https://lmstudio.ai
2. Search for "Qwen3-4B-Instruct" and download the MLX 4-bit version
3. Copy the model files to `models/qwen3-4b-instruct-4bit/` in this project

### 3. Run the Demo
```bash
python basic_parser.py
```

## 📊 Expected Output

```
🏥 Dental Claim Parser - Educational Demo
Using MLX Framework with Qwen3-4B Language Model
==================================================
🤖 Loading Qwen3-4B model with MLX...
✅ Model loaded successfully in 1.54 seconds
💾 Memory usage: ~2-3GB (4-bit quantized)
📋 Sample dental claim data loaded
✅ Prompts prepared for model
🧠 Generating response with MLX...
✅ Response generated in 9.60 seconds
✅ JSON parsed successfully

🎉 SUCCESS! Extracted procedure-level data:
----------------------------------------
📊 Extracted 2 dental procedures
📝 Writing procedure data to CSV: procedure_level_claims.csv
✅ CSV file created with 2 procedure records
✅ Demo completed successfully!
```

**Generated CSV Output:**
```csv
subscriber_id,claim_number,procedure_code,service_date,tooth_number,submitted_amount,plan_paid_amount
H61010799,202524532740000,D7240,08/29/25,16,370.99,370.99
H61010799,202524532740000,D7240,08/29/25,17,370.99,370.99
```

## 🧠 How It Works

### 1. Load the AI Model with MLX
MLX is Apple's framework for running machine learning models on Apple Silicon:
- Uses unified memory architecture (no data copying between CPU/GPU)
- 4-bit quantization reduces model size from ~8GB to ~2GB
- Metal GPU acceleration for fast inference

### 2. Create Structured Prompts
The key to success is prompt design:
- **System prompt**: Defines role and strict output constraints (JSON only)
- **User prompt**: Shows input data and expected output structure
- **Section markers** (### ... ###): Clearly separate input from instructions
- **Concise approach**: For 4B models, less verbose = better results

### 3. AI Processes the Table
The model intelligently:
- Parses metadata strings to extract member/claim numbers
- Maps column headers (which vary by insurance company)
- Identifies actual procedure rows vs summary/total rows
- Extracts all required fields into structured JSON

### 4. Output Clean CSV
Converts JSON to CSV format ready for:
- Database import
- Analytics and reporting
- Downstream processing

## ⏱️ Performance (on Apple Silicon)

Tested on M1/M2/M3 Macs:

- **Model loading**: ~1-2 seconds
- **Processing one claim**: ~8-10 seconds  
- **Memory used**: ~2-3GB
- **Accuracy**: 100% on test claims with proper prompt engineering

**Performance Comparison:**
- 0.5B model: Fast (2s) but low accuracy (failed metadata extraction)
- 2.5B-3B model: Good balance (6-8s) but occasionally includes totals
- **Qwen3-4B: Best results** (9-10s) with perfect accuracy

*Note: Performance may vary based on Mac model and available memory*

## 🔧 Technical Details

### MLX Framework Benefits

- **Unified Memory**: No CPU-GPU data transfers on Apple Silicon
- **4-bit Quantization**: Reduces model from 8GB to 2-3GB with minimal accuracy loss
- **Metal Backend**: Native GPU acceleration without CUDA or external dependencies
- **Simple API**: Much easier than PyTorch/TensorFlow for inference

### Model Configuration

- **Architecture**: Qwen3-4B-Instruct (Transformer-based)
- **Quantization**: 4-bit for memory efficiency
- **Context Window**: 2048 tokens (sufficient for most claims)
- **Generation Limit**: 5000 tokens (handles ~25 procedures per claim)

### Prompt Engineering Insights

**What Works:**
- Clear role definition with strict output constraints
- Section markers (###) to separate input and instructions
- Showing target structure with field descriptions
- Concise instructions (verbose = worse for 4B models)

**What Doesn't Work:**
- Step-by-step detailed instructions (triggers verbose mode)
- Too short prompts without structure (misses metadata)
- Examples with actual values (gives away answers)


## 🔗 Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [Qwen3 Model Card](https://huggingface.co/Qwen)
- [MLX Community Models](https://huggingface.co/mlx-community)

## 🚧 Extending This Project

Ideas for further development:

1. **Batch Processing**: Process multiple claims from a folder
2. **PDF Integration**: Add pdfplumber/camelot to extract tables directly from PDFs
3. **Error Handling**: Retry logic, validation rules, data quality checks
4. **Web Interface**: Build a simple UI with Streamlit or Flask
5. **Database Integration**: Automatically insert results into PostgreSQL/MySQL
6. **Different Claim Types**: Expand to medical, vision, or other insurance types

## 📝 License

This educational demo is provided for learning purposes and is licensed under the MIT License. Anyone is free to use, modify, and distribute this project for any purpose. 
