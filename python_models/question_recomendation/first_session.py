import random
import json
from utils import load_domain_questions

# -------------------------
# FIRST SESSION (Easy Questions)
# -------------------------
def first_session_recommendations(user_json, k=5):
    """
    STATELESS: Recommend EASY questions (difficulty 1 or 2).
    Returns data for MongoDB storage.
    
    Args:
        user_json: {
            "_id": "user_001",
            "skills": ["java", "nlp", "tensorflow"],
            "target_domain": "aiml"
        }
        k: Number of questions to recommend
    
    Returns:
        JSON object to store in MongoDB
    """
    print("\n=== EASY QUESTION RECOMMENDER START ===")
    
    domain_key = user_json["target_domain"]
    
    # 1) Load questions
    questions = load_domain_questions(domain_key)
    print(f"[INFO] Loaded {len(questions)} questions from domain '{domain_key}'")
    
    # 2) Filter EASY questions only
    easy_questions = [q for q in questions if q.get("difficulty") in [1, 2]]
    print(f"[INFO] Found {len(easy_questions)} easy questions (difficulty 1 or 2)")
    
    if not easy_questions:
        raise ValueError("No easy questions available in this domain.")
    
    # 3) Prepare user skills
    user_skills = set(s.lower() for s in user_json.get("skills", []))
    print(f"[INFO] User Skills: {list(user_skills)}")
    
    # 4) Score by relevance (overlap)
    scored = []
    for q in easy_questions:
        tags = set(t.lower() for t in q.get("tags", []))
        req = set(s.lower() for s in q.get("required_skills", []))
        overlap = len((tags | req) & user_skills)
        
        # Only skill relevance matters here (no difficulty penalty)
        score = overlap
        scored.append((q, score))
    
    # 5) Random factor between 0 and 1
    ranked = sorted(
        scored,
        key=lambda x: (x[1] * 0.7 + random.random() * 0.3),
        reverse=True
    )

    
    # 7) Pick top-k
    top_k = ranked[:k]
    print(f"[INFO] Selected Top {k} Easy Questions")
    
    # 8) Build result - Ready for MongoDB
    result = {
        "user_id": user_json.get("_id"),
        "domain": domain_key,
        "session_type": "first_session",
        "session_number": 1,
        "message": "Easy questions for warm-up and baseline assessment",
        "questions_recommended": [],
        "question_ids": []
    }
    
    for q, score in top_k:
        result["questions_recommended"].append({
            "question": q,
            "score": score,
            "reasoning": (
                f"Skill match: {score} overlaps"
                if score > 0 else
                "Low relevance but chosen as easy question"
            )
        })
        result["question_ids"].append(q["_id"])
    
    print("=== EASY QUESTION RECOMMENDER READY ===\n")
    return result


def main(input_json):
    # Python receives data as dict
    return first_session_recommendations(input_json, k=5)

print("\n\n[EXAMPLE 2] First Session")
print("=" * 70)

user = {
        "_id": "user_001",
        "skills": ['java', 'nlp', 'tensorflow', 'data-structures'],
        "target_domain": "ai_ml"
    }

first_result = first_session_recommendations(user, k=5)
print("\n📦 Result (Store this in MongoDB):")
print(json.dumps(first_result, indent=2))