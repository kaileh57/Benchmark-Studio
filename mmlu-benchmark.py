import argparse
import os
import json
import time
import requests
import zipfile
import io
from tqdm import tqdm
from pathlib import Path
import random
import csv

def format_prompt(question, choices, subject):
    """Format the prompt for the model."""
    prompt = f"The following is a multiple choice question about {subject}.\n\n"
    prompt += f"Question: {question}\n"
    
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    
    prompt += "\nAnswer:"
    return prompt

def extract_answer(response):
    """Extract the answer letter (A, B, C, D) from the model response."""
    response = response.strip()
    
    # Look for the first occurrence of A, B, C, or D
    for char in response:
        if char in "ABCD":
            return char
    
    # If no direct letter found, check for common patterns
    response_lower = response.lower()
    if "a" in response_lower and "b" not in response_lower and "c" not in response_lower and "d" not in response_lower:
        return "A"
    elif "b" in response_lower and "a" not in response_lower and "c" not in response_lower and "d" not in response_lower:
        return "B"
    elif "c" in response_lower and "a" not in response_lower and "b" not in response_lower and "d" not in response_lower:
        return "C"
    elif "d" in response_lower and "a" not in response_lower and "b" not in response_lower and "c" not in response_lower:
        return "D"
    
    return ""

def load_mmlu_samples(data_dir, num_samples=100):
    """Load samples from the MMLU dataset."""
    # Define the subjects we want to test
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
        "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
        "college_medicine", "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts",
        "high_school_biology", "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting", "professional_law",
        "professional_medicine", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy", "virology",
        "world_religions"
    ]
    
    samples = []
    for subject in subjects:
        subject_path = Path(data_dir) / f"{subject}_dev.csv"
        if not subject_path.exists():
            print(f"Warning: Could not find data for {subject}")
            continue
        
        # Read the CSV file
        with open(subject_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            subject_samples = list(reader)
        
        # Select random samples from this subject
        if subject_samples:
            subject_count = max(1, num_samples // len(subjects))
            selected = random.sample(subject_samples, min(subject_count, len(subject_samples)))
            
            for row in selected:
                if len(row) >= 5:  # Ensure we have question, 4 choices, and answer
                    question = row[0]
                    choices = row[1:5]
                    # Answer is the last element (0-3 index)
                    answer_idx = int(row[-1])
                    answer = chr(65 + answer_idx)  # Convert to A, B, C, D
                    
                    samples.append({
                        "subject": subject,
                        "question": question,
                        "choices": choices,
                        "answer": answer
                    })
    
    # If we have more samples than requested, select a random subset
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)
    
    return samples

def evaluate_model(model_path, data_dir, num_samples=100):
    """Evaluate a GGUF model on MMLU."""
    # Import here to avoid requiring the package unless needed
    from llama_cpp import Llama
    
    print(f"Loading model from {model_path}...")
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_batch=1,
        verbose=False
    )
    print("Model loaded successfully.")
    
    # Load MMLU samples
    print(f"Loading MMLU samples (up to {num_samples})...")
    samples = load_mmlu_samples(data_dir, num_samples)
    if not samples:
        print("Error: No MMLU samples loaded. Exiting.")
        return None
    
    print(f"Loaded {len(samples)} samples across {len(set(s['subject'] for s in samples))} subjects.")
    
    # Prepare results structure
    results = {
        "total": 0,
        "correct": 0,
        "by_subject": {}
    }
    
    # Evaluate on each sample
    print("Starting evaluation...")
    for sample in tqdm(samples, desc="Evaluating"):
        subject = sample["subject"]
        
        # Initialize subject in results if not already present
        if subject not in results["by_subject"]:
            results["by_subject"][subject] = {"total": 0, "correct": 0}
        
        # Format prompt for this question
        prompt = format_prompt(sample["question"], sample["choices"], subject)
        
        # Get model response
        output = model(
            prompt,
            max_tokens=10,
            temperature=0.1,
            top_p=0.95,
            stop=["\n"],
            echo=False
        )
        
        # Extract answer
        response_text = output["choices"][0]["text"]
        model_answer = extract_answer(response_text)
        
        # Check if correct
        is_correct = model_answer == sample["answer"]
        
        # Update counters
        results["total"] += 1
        results["correct"] += int(is_correct)
        results["by_subject"][subject]["total"] += 1
        results["by_subject"][subject]["correct"] += int(is_correct)
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    else:
        results["accuracy"] = 0
    
    # Calculate subject accuracies
    for subject in results["by_subject"]:
        subj_total = results["by_subject"][subject]["total"]
        if subj_total > 0:
            subj_correct = results["by_subject"][subject]["correct"]
            results["by_subject"][subject]["accuracy"] = subj_correct / subj_total
        else:
            results["by_subject"][subject]["accuracy"] = 0
    
    return results

def download_mmlu_data(output_dir="mmlu_data"):
    """Download the MMLU dataset."""
    import requests
    import zipfile
    import io
    
    print("Downloading MMLU dataset...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check if files already exist
    if list(output_path.glob("*_dev.csv")):
        print("MMLU data already exists.")
        return output_path
    
    # Download the zip file
    url = "https://github.com/hendrycks/test/archive/refs/heads/master.zip"
    r = requests.get(url, stream=True)
    
    # Extract the data directory
    z = zipfile.ZipFile(io.BytesIO(r.content))
    
    # Extract only the necessary files
    for file in tqdm(z.namelist(), desc="Extracting"):
        if file.startswith("test-master/data/") and file.endswith(("_dev.csv", "_test.csv")):
            z.extract(file, output_path)
    
    # Move files to the root directory
    data_dir = output_path / "test-master" / "data"
    for file in data_dir.glob("**/*"):
        if file.is_file():
            dest = output_path / file.name
            os.rename(file, dest)
    
    # Clean up
    import shutil
    shutil.rmtree(output_path / "test-master")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Benchmark a GGUF model using MMLU")
    parser.add_argument("model_path", type=str, help="Path to the GGUF model file")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Number of MMLU samples to evaluate (default: 100)")
    parser.add_argument("--data-dir", type=str, default="mmlu_data",
                        help="Directory to store/load MMLU data (default: mmlu_data)")
    args = parser.parse_args()
    
    # Verify the model file exists
    if not os.path.isfile(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Download MMLU data if needed
    data_dir = download_mmlu_data(args.data_dir)
    
    # Run the evaluation
    start_time = time.time()
    results = evaluate_model(args.model_path, data_dir, args.samples)
    elapsed_time = time.time() - start_time
    
    if not results:
        return
    
    # Print results
    print("\n===== MMLU Benchmark Results =====")
    print(f"Model: {os.path.basename(args.model_path)}")
    print(f"Total questions: {results['total']}")
    print(f"Overall accuracy: {results['accuracy']:.2%}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Sort subjects by accuracy
    sorted_subjects = sorted(
        results["by_subject"].items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )
    
    print("\nResults by subject:")
    for subject, data in sorted_subjects:
        if data["total"] > 0:
            print(f"  {subject}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    
    # Save results to JSON file
    output_file = f"mmlu_results_{os.path.basename(args.model_path)}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
