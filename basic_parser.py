#!/usr/bin/env python3
"""
Dental Claim Parser using Local Small Language Model
====================================================
This educational demo shows how to use a local Qwen2-0.5B model to parse
dental claim tables and extract structured information.

Key Learning Points:
- Loading and using small local LLMs (500M parameters)
- Memory optimization with Float16 precision
- Prompt engineering for structured data extraction
- JSON parsing and validation
"""

import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """
    Load the Qwen2-0.5B model with optimizations for local inference
    
    Returns:
        tuple: (tokenizer, model) - Ready to use for inference
        
    Key optimizations:
    - Float16: Reduces memory usage by 50% (942MB vs 1885MB)
    - low_cpu_mem_usage: Efficient loading for limited RAM
    - use_cache: Speeds up sequential token generation
    """
    print("🤖 Loading Qwen2-0.5B model locally...")
    start_time = time.time()
    
    # Path to the local model files
    model_path = "./models/qwen2-0.5b"
    
    # Load tokenizer (converts text to numbers the model understands)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure padding token exists (needed for batch processing)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load the language model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,        # Half precision: 2 bytes per parameter instead of 4
        low_cpu_mem_usage=True,     # Load efficiently to save RAM
        use_cache=True,             # Cache previous computations for speed
    )
    
    # Set to evaluation mode (no training, just inference)
    model.eval()
    
    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
    print(f"💾 Memory usage: ~942MB (Float16 optimization)")
    
    return tokenizer, model

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

def generate_response(tokenizer, model, system_prompt, user_prompt):
    """
    Use the language model to generate a structured response
    
    This function handles the core LLM inference process:
    1. Format the conversation properly
    2. Convert text to tokens (numbers)
    3. Generate new tokens with the model
    4. Convert tokens back to text
    
    Args:
        tokenizer: Converts between text and tokens
        model: The language model for generation
        system_prompt (str): Instructions for the model
        user_prompt (str): The actual task and data
        
    Returns:
        str: Generated response from the model
    """
    print("🧠 Generating response from local model...")
    start_time = time.time()
    
    # Step 1: Format as a conversation (like ChatGPT format)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Step 2: Apply the model's chat template (Qwen2 specific formatting)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Step 3: Convert text to tokens (numbers the model understands)
    model_inputs = tokenizer(
        [text], 
        return_tensors="pt",     # Return PyTorch tensors
        padding=True,            # Pad to consistent length
        truncation=True,         # Cut off if too long
        max_length=2048          # Maximum input length
    )
    
    # Step 4: Generate new tokens using the model
    with torch.no_grad():  # Don't compute gradients (we're not training)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=400,           # Limit output length
            do_sample=False,              # Use greedy decoding (most likely tokens)
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Step 5: Extract only the newly generated tokens (remove input)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # Step 6: Convert tokens back to readable text
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
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
    print("Using Local Qwen2-0.5B Language Model")
    print("=" * 50)
    
    # Step 1: Load the local language model
    tokenizer, model = load_model()
    
    # Step 2: Get sample dental claim data
    test_data = get_test_data()
    print("📋 Sample dental claim data loaded")
    
    # Step 3: Create structured prompts for the model
    system_prompt, user_prompt = create_prompt(test_data)
    print("✅ Prompts prepared for model")
    
    # Step 4: Generate response using the local model
    response = generate_response(tokenizer, model, system_prompt, user_prompt)
    
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
