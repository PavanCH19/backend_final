import json
import os

# -------------------------
# DOMAIN QUESTION LOADER
# -------------------------
def load_domain_questions(domain_key):
    """Load and clean questions from a specific domain JSON file."""

    # BASE DIRECTORY FOR DOMAIN JSON FILES
    BASE_DIR = os.path.join(os.path.dirname(__file__), "domains")
    
    # full path to file
    filename = os.path.join(BASE_DIR, f"{domain_key}.json")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"JSON file for domain '{domain_key}' not found at {filename}")
    
    with open(filename, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    cleaned = []
    
    for item in raw:
        if "options" in item and isinstance(item["options"], list):
            options = item["options"]
        else:
            options = [item[k] for k in ["0", "1", "2", "3"] if k in item]

        if "tags" in item and isinstance(item["tags"], list):
            tags = item["tags"]
        else:
            tags = [item[k] for k in item if k.isdigit() and int(k) >= 4]

        required_skills = item.get("required_skills", []) if isinstance(item.get("required_skills"), list) else []
        hints = item.get("hints", []) if isinstance(item.get("hints"), list) else []

        cleaned_question = {
            "_id": item.get("_id"),
            "question_type": item.get("question_type", "unknown"),
            "domain": item.get("domain", ""),
            "topic": item.get("topic", ""),
            "difficulty": item.get("difficulty", 3),
            "difficulty_label": item.get("difficulty_label", ""),
            "estimated_time_sec": item.get("estimated_time_sec"),
            "text": item.get("text", ""),
            "options": options,
            "answer": item.get("answer"),
            "explanation": item.get("explanation"),
            "tags": tags,
            "required_skills": required_skills,
            "hints": hints
        }
        
        cleaned.append(cleaned_question)
    
    return cleaned
