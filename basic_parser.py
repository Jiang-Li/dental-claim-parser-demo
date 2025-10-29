#!/usr/bin/env python3
"""
Dental Claim Parser using MLX Framework
=======================================
This educational demo shows how to use Apple's MLX framework with a local 
Qwen3-4B model to parse dental claim tables and extract structured information.

Key Learning Points:
- Loading and using local LLMs (4B parameters, 4-bit quantized) with MLX
- Apple Silicon optimization with MLX framework for fast inference
- Prompt engineering for structured data extraction from complex tables
- JSON parsing and validation for production-ready output

Model Requirements:
- Qwen3-4B-Instruct (4-bit quantized) or similar MLX-optimized model
- ~2-3GB RAM for model weights
- Apple Silicon (M1/M2/M3) recommended for optimal performance
"""

import json
import csv
import time
from mlx_lm import load, generate

def load_model():
    """
    Load the Qwen3-4B model with MLX optimizations for Apple Silicon
    
    The model can be downloaded from Hugging Face using:
        huggingface-cli download mlx-community/Qwen3-4B-Instruct-4bit \
            --local-dir models/qwen3-4b-instruct-4bit
    
    Or use LM Studio to download and copy to the models folder.
    
    Returns:
        tuple: (model, tokenizer) - Ready to use for inference
        
    Key optimizations:
    - MLX framework: Unified memory architecture on Apple Silicon
    - 4-bit quantization: Reduces memory from ~8GB to ~2-3GB
    - Metal acceleration: Native GPU support without external dependencies
    """
    print("🤖 Loading Qwen3-4B model with MLX...")
    start_time = time.time()
    
    # Path to local Qwen3-4B MLX model (4-bit quantized)
    # Alternative: Use "mlx-community/Qwen3-4B-Instruct-4bit" to auto-download
    model_path = "models/qwen3-4b-instruct-4bit"
    
    # Load model and tokenizer using MLX
    # tokenizer_config sets the end-of-sequence token for Qwen models
    model, tokenizer = load(
        model_path,
        tokenizer_config={"eos_token": "<|im_end|>"}
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    print(f"💾 Memory usage: ~2-3GB (4-bit quantized)")
    
    return model, tokenizer

def get_test_data():
    """
    Sample dental claim table data for demonstration
    
    This represents extracted table data from a dental claim PDF with:
    - Raw metadata string (row 0): Contains patient, provider, claim info
    - Column headers (row 1): Field names that may vary by insurance company
    - Data rows (rows 2-3): Individual dental procedures
    - Summary row (row 4): Totals (should be filtered out during extraction)
    
    Real-world usage: Replace this with actual PDF table extraction output
    using libraries like pdfplumber, camelot, or tabula-py.
    
    Returns:
        dict: Structured table data with metadata and raw table rows
    """
    return {
        "table_metadata": {
            "page_number": 3,
            "table_index": 0,
            "table_type": "claim_detail",
            "rows": 5,
            "cols": 17
        },
        "raw_data": [
            [
                "Patient Name: Xwcfs, Dytts Provider Name: John Doe Office Reference #:R8T204SQP Claim #: 202524532740000\nMember #: H61010799 Provider NPI: 1699047043 Group: OH Humana Medicaid Auth #: 202523034841400\nMember Type: Subscriber Location NPI: Sub-Group: OHIO Humana Medicaid Adult Referral #:\nDOB: 11/11/1111 Place of Service: Office Product: Medicaid- Adult Referral Date:\nService Address: 7370 Sawmill Rd",
                None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
            ],
            [
                "Item", "Submitted\nCode", "Paid\nCode", "Tooth", "Description", "Date of\nService",
                "Submitted", "Approved", "Allowed", "Other\nInsurance", "Copay", "Plan %",
                "Deductible", "Patient Pay", "Writeoff", "Plan Pay", "Processing\nPolicies"
            ],
            [
                "1", "D7240", "D7240", "16", "removal of impacted\ntooth-completely bony",
                "08/29/25", "$370.99", "$370.99", "$370.99", "$0.00", "$0.00", "100%",
                "$0.00", "$0.00", "$0.00", "$370.99", ""
            ],
            [
                "2", "D7240", "D7240", "17", "removal of impacted\ntooth-completely bony",
                "08/29/25", "$370.99", "$370.99", "$370.99", "$0.00", "$0.00", "100%",
                "$0.00", "$0.00", "$0.00", "$370.99", ""
            ],
            [
                "", None, None, None, None, "Total:", "$741.98", "$741.98", "$741.98",
                "$0.00", "$0.00", "", "$0.00", "$0.00", "$0.00", "$741.98", ""
            ]
        ]
    }

def create_prompt(json_data):
    """
    Create prompts for the language model to extract structured data
    
    This is the heart of prompt engineering. The approach used here is:
    - Simple, clear instructions without over-explanation
    - Strict output format constraints (JSON only, no conversational text)
    - Explicit target structure showing field names and constraints
    - Section markers (### ... ###) to clearly separate input and output
    
    Key insight: For 4B parameter models, concise prompts with clear structure
    work better than verbose step-by-step instructions. The model can infer
    extraction logic from the data structure and target format.
    
    Args:
        json_data (dict): The raw table data to process
        
    Returns:
        tuple: (system_prompt, user_prompt) for the conversation
    """
    
    # System prompt: Define the model's role and enforce JSON-only output
    # The **bold** emphasis and STRICT RULE help prevent conversational responses
    system_prompt = """You are a **data extraction assistant** with the sole task of converting unstructured data into a structured JSON format.

**STRICT RULE:** BEGIN AND END RESPONSE WITH THE JSON OBJECT ONLY. DO NOT ADD ANY INTRODUCTORY, CONCLUDING, OR CONVERSATIONAL TEXT."""
    
    # User prompt: Provide input data and expected output structure
    # This minimal approach works well for Qwen3-4B, which has strong reasoning
    user_prompt = f"""Your task is to extract specific information from the provided JSON Input, which represents a dental claim.

**If a piece of information is not present, the value MUST be `null`.**

### JSON INPUT ###
{json.dumps(json_data, indent=2)}
### END JSON INPUT ###

### TARGET JSON STRUCTURE ###
Your response MUST strictly adhere to this format and constraints:
[
  {{
    "member_number": "[value]" (must be the Member #),
    "claim_number": "[value]",
    "procedure_code": "[value]",
    "service_date": "[value]" (format MM/DD/YY),
    "tooth_number": "[value]",
    "submitted_amount": "[value]",
    "plan_paid_amount": "[value]"
  }}
]
### END TARGET JSON STRUCTURE ###"""
    
    return system_prompt, user_prompt

def generate_response(model, tokenizer, system_prompt, user_prompt):
    """
    Use the MLX language model to generate a structured response
    
    This function handles the core LLM inference process using MLX:
    1. Format the conversation properly
    2. Apply chat template
    3. Generate response using MLX (handles tokenization automatically)
    
    Args:
        model: The MLX language model for generation
        tokenizer: MLX tokenizer for formatting
        system_prompt (str): Instructions for the model
        user_prompt (str): The actual task and data
        
    Returns:
        str: Generated response from the model
    """
    print("🧠 Generating response with MLX...")
    start_time = time.time()
    
    # Step 1: Format as a conversation (like ChatGPT format)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Step 2: Apply the model's chat template (Qwen-specific formatting)
    # This converts the message format into the proper token sequence
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,              # Get string output, not token IDs
        add_generation_prompt=True   # Add special tokens for response start
    )
    
    # Step 3: Generate response using MLX (much simpler than PyTorch!)
    # MLX handles tokenization, generation, and decoding automatically
    response = generate(
        model, 
        tokenizer, 
        prompt=text,
        max_tokens=5000,         # High limit for tables with many procedures (~25+)
        verbose=False            # Suppress token-by-token output
    )
    
    generation_time = time.time() - start_time
    print(f"✅ Response generated in {generation_time:.2f} seconds")
    
    return response

def write_to_csv(procedures_data, output_file="procedure_level_claims.csv"):
    """
    Write procedure-level data to CSV file
    
    Converts the JSON array output from the LLM into a CSV file suitable
    for database import, analytics, or further processing.
    
    Args:
        procedures_data (list): List of procedure dictionaries from LLM
        output_file (str): Output CSV filename (default: procedure_level_claims.csv)
    """
    print(f"📝 Writing procedure data to CSV: {output_file}")
    
    # Define CSV column headers (normalized field names)
    fieldnames = [
        'subscriber_id',       # Patient/member identifier
        'claim_number',        # Unique claim identifier
        'procedure_code',      # Dental procedure code (e.g., D7240)
        'service_date',        # Date procedure was performed
        'tooth_number',        # Tooth number or surface
        'submitted_amount',    # Amount billed by provider
        'plan_paid_amount'     # Amount paid by insurance plan
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header row
        writer.writeheader()
        
        # Write each procedure as a row
        for procedure in procedures_data:
            # Map LLM output field names to standardized CSV column names
            # Handles variations in field naming (member_number vs subscriber_id)
            mapped_procedure = {
                'subscriber_id': procedure.get('member_number') or procedure.get('subscriber_id'),
                'claim_number': procedure.get('claim_number'),
                'procedure_code': procedure.get('procedure_code'),
                'service_date': procedure.get('service_date'),
                'tooth_number': procedure.get('tooth_number'),
                'submitted_amount': procedure.get('submitted_amount'),
                'plan_paid_amount': procedure.get('plan_paid_amount')
            }
            writer.writerow(mapped_procedure)
    
    print(f"✅ CSV file created with {len(procedures_data)} procedure records")


def main():
    """
    Main demonstration function
    
    Workflow:
    1. Load MLX-optimized language model from local storage
    2. Prepare sample dental claim data (simulates PDF extraction output)
    3. Create structured prompts for data extraction
    4. Generate response using local LLM (no API calls, runs on-device)
    5. Parse and validate JSON response
    6. Save results to CSV file for downstream processing
    
    Performance on Apple Silicon (M1/M2/M3):
    - Model loading: ~1-2 seconds
    - Inference: ~8-10 seconds per claim
    - Memory: ~2-3GB
    """
    print("🏥 Dental Claim Parser - Educational Demo")
    print("Using MLX Framework with Qwen3-4B Language Model")
    print("=" * 50)
    
    # Step 1: Load the local language model
    model, tokenizer = load_model()
    
    # Step 2: Get sample dental claim data
    # In production, this would come from PDF extraction
    test_data = get_test_data()
    print("📋 Sample dental claim data loaded")
    
    # Step 3: Create structured prompts for the model
    # This is where prompt engineering happens
    system_prompt, user_prompt = create_prompt(test_data)
    print("✅ Prompts prepared for model")
    
    # Step 4: Generate response using the local model
    # All processing happens on-device using Apple Silicon GPU
    response = generate_response(model, tokenizer, system_prompt, user_prompt)
    
    # Step 5: Parse the JSON response
    # Our prompt design ensures clean JSON output (no markdown, no explanations)
    try:
        procedures = json.loads(response)
        print("✅ JSON parsed successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Model didn't return valid JSON: {e}")
        print(f"Raw response: {response[:200]}...")  # Show first 200 chars for debugging
        procedures = None
    
    # Step 6: Display and save results
    if procedures:
        print("\n🎉 SUCCESS! Extracted procedure-level data:")
        print("-" * 40)
        print(f"📊 Extracted {len(procedures)} dental procedures")
        
        # Save to CSV file for database import or analytics
        write_to_csv(procedures)
        print("✅ Demo completed successfully!")
    else:
        print("\n❌ Failed to extract structured data")
        print("💡 Check the raw response above for issues")

if __name__ == "__main__":
    main()
