#!/usr/bin/env python3
"""
Dental Claim Parser using MLX Framework
=======================================
This educational demo shows how to use Apple's MLX framework with a local 
Qwen2-0.5B model to parse dental claim tables and extract structured information.

Key Learning Points:
- Loading and using small local LLMs (500M parameters) with MLX
- Apple Silicon optimization with MLX framework
- Prompt engineering for structured data extraction
- JSON parsing and validation
"""

import json
import time
from mlx_lm import load, generate

def load_model():
    """
    Load the Qwen2-0.5B model with MLX optimizations for Apple Silicon
    
    Returns:
        tuple: (model, tokenizer) - Ready to use for inference
        
    Key optimizations:
    - MLX framework: Optimized for Apple Silicon (M1/M2/M3)
    - Automatic memory management: No manual dtype/memory settings needed
    - Native Apple Silicon acceleration: Better performance than PyTorch
    """
    print("🤖 Loading Qwen2-0.5B model with MLX...")
    start_time = time.time()
    
    # Use the pre-converted MLX model from Hugging Face
    model_path = "Qwen/Qwen2-0.5B-Instruct-MLX"
    
    # Load model and tokenizer using MLX
    # MLX automatically handles memory optimization for Apple Silicon
    model, tokenizer = load(
        model_path,
        tokenizer_config={"eos_token": "<|im_end|>"}
    )
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    print(f"💾 Memory usage: Optimized automatically by MLX for Apple Silicon")
    
    return model, tokenizer

def get_test_data():
    """
    Sample dental claim table data for demonstration
    
    This represents a real dental claim with:
    - Patient and provider information
    - Two dental procedures (tooth extractions)
    - Financial details (amounts, payments, etc.)
    
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
    
    This is the heart of prompt engineering - we tell the model:
    1. What its role is (system prompt)
    2. What data to process and how to format output (user prompt)
    
    Args:
        json_data (dict): The raw table data to process
        
    Returns:
        tuple: (system_prompt, user_prompt) for the conversation
    """
    
    # System prompt: Define the model's role and behavior
    system_prompt = """You are a data extraction assistant. Your task is to extract specific information from a JSON object representing a dental claim and its procedures. Do not generate any conversational text. Your response should be a single JSON object that strictly adheres to the specified structure. If a piece of information is not present, use a null value."""
    
    # User prompt: Provide the data and specify exact output format
    user_prompt = f"""JSON Input:
{json.dumps(json_data, indent=2)}

Output JSON Structure:
{{
  "claim_number": "[value]",
  "provider_name": "[value]",
  "patient_name": "[value]",
  "patient_dob": "[value]",
  "procedures": [
    {{
      "item_number": "[value]",
      "submitted_code": "[value]",
      "description": "[value]",
      "submitted_amount": "[value]",
      "plan_pay": "[value]"
    }}
  ]
}}"""
    
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
    
    # Step 2: Apply the model's chat template (Qwen2 specific formatting)
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Step 3: Generate response using MLX (much simpler than PyTorch!)
    # MLX handles tokenization, generation, and decoding automatically
    response = generate(
        model, 
        tokenizer, 
        prompt=text,
        max_tokens=400,          # Limit output length
        verbose=False            # Don't print generation progress
    )
    
    generation_time = time.time() - start_time
    print(f"✅ Response generated in {generation_time:.2f} seconds")
    
    return response


def main():
    """
    Main demonstration function
    
    This shows the complete workflow:
    1. Load a local small language model
    2. Prepare dental claim data
    3. Create structured prompts
    4. Generate and parse the response
    5. Display the extracted information
    """
    print("🏥 Dental Claim Parser - Educational Demo")
    print("Using MLX Framework with Qwen2-0.5B Language Model")
    print("=" * 50)
    
    # Step 1: Load the local language model
    model, tokenizer = load_model()
    
    # Step 2: Get sample dental claim data
    test_data = get_test_data()
    print("📋 Sample dental claim data loaded")
    
    # Step 3: Create structured prompts for the model
    system_prompt, user_prompt = create_prompt(test_data)
    print("✅ Prompts prepared for model")
    
    # Step 4: Generate response using the local model
    response = generate_response(model, tokenizer, system_prompt, user_prompt)
    
    # Step 5: Parse the JSON response (our prompt ensures it's clean JSON)
    try:
        result = json.loads(response)
        print("✅ JSON parsed successfully")
    except json.JSONDecodeError as e:
        print(f"❌ Model didn't return valid JSON: {e}")
        result = None
    
    # Step 6: Display and save results
    if result:
        print("\n🎉 SUCCESS! Extracted structured data:")
        print("-" * 40)
        print(json.dumps(result, indent=2))
        print("-" * 40)
        print(f"📊 Extracted {len(result.get('procedures', []))} dental procedures")
        
        # Save to output file
        output_file = "extracted_claim_data.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Results saved to: {output_file}")
        print("✅ Demo completed successfully!")
    else:
        print("\n❌ Failed to extract structured data")
        print("💡 This can happen with small models - try running again!")

if __name__ == "__main__":
    main()
