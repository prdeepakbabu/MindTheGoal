#!/usr/bin/env python3
"""Download MultiWOZ dataset and save as JSON for local use."""

import json
import os
import urllib.request
import zipfile
import tempfile

def download_multiwoz():
    """Download MultiWOZ from GitHub and save locally."""
    
    # MultiWOZ 2.4 from official repository
    url = "https://github.com/smartyfh/MultiWOZ2.4/raw/main/data/MULTIWOZ2.4.zip"
    
    print(f"Downloading MultiWOZ 2.4 from GitHub...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "multiwoz")
    os.makedirs(output_dir, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "multiwoz.zip")
        
        # Download
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete. Extracting...")
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        
        # Find and process the data files
        dialogues = []
        
        for root, dirs, files in os.walk(tmpdir):
            for filename in files:
                if filename.endswith('.json') and filename not in ['schema.json', 'ontology.json']:
                    filepath = os.path.join(root, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        if isinstance(data, dict):
                            for dialogue_id, dialogue_data in data.items():
                                turns = process_dialogue(dialogue_id, dialogue_data)
                                if turns:
                                    dialogues.append({
                                        "dialogue_id": dialogue_id,
                                        "turns": turns,
                                        "domains": extract_domains(dialogue_data)
                                    })
                    except Exception as e:
                        print(f"Warning: Could not process {filename}: {e}")
        
        # Save processed data
        output_file = os.path.join(output_dir, "dialogues.json")
        with open(output_file, "w") as f:
            json.dump(dialogues, f)
        
        print(f"\nSaved {len(dialogues)} dialogues to {output_file}")
        return len(dialogues)


def process_dialogue(dialogue_id, dialogue_data):
    """Process a single dialogue into turns."""
    log = dialogue_data.get("log", [])
    
    if not log:
        return None
    
    turns = []
    
    # Log alternates between user and system
    for i in range(0, len(log) - 1, 2):
        user_turn = log[i]
        system_turn = log[i + 1] if i + 1 < len(log) else None
        
        if system_turn:
            turns.append({
                "user": user_turn.get("text", ""),
                "agent": system_turn.get("text", "")
            })
    
    return turns if turns else None


def extract_domains(dialogue_data):
    """Extract domains from dialogue data."""
    goal = dialogue_data.get("goal", {})
    domains = []
    
    for domain in ["restaurant", "hotel", "attraction", "taxi", "train", "hospital", "police"]:
        if domain in goal and goal[domain]:
            domains.append(domain)
    
    return domains


if __name__ == "__main__":
    download_multiwoz()
