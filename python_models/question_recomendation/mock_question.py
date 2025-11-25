from utils import load_domain_questions
import random
import json

# -------------------------
# MOCK SESSION (Initial Cold Start)
# -------------------------
def mock_session_recommendations(user_json, k=5):
    """
    STATELESS: First session with skill-based recommendations.
    Returns full question objects for MongoDB storage.
    
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
    
    print("\n=== MOCK SESSION START ===")
    
    # 1) Load questions
    domain = user_json["target_domain"]
    questions = load_domain_questions(domain)
    print(f"[INFO] Loaded {len(questions)} questions from domain '{domain}'")
    
    user_skills = [s.lower() for s in user_json.get("skills", [])]
    print(f"[INFO] User Skills: {user_skills}")
    
    # 2) Build diverse pool (match on tags OR required_skills)
    diverse_pool = []
    for q in questions:
        q_tags = [t.lower() for t in q.get("tags", [])]
        q_required = [r.lower() for r in q.get("required_skills", [])]
        # match if any user skill appears in tags OR required_skills
        if any(skill in q_tags or skill in q_required for skill in user_skills):
            diverse_pool.append(q)
    
    if not diverse_pool:
        print("[INFO] No direct skill match found → using full question list")
        diverse_pool = questions
    else:
        print(f"[INFO] Diverse pool created with {len(diverse_pool)} questions")
    
    # 3) Score questions
    scored = []
    user_skills_set = set(user_skills)
    for q in diverse_pool:
        q_tags_set = set([t.lower() for t in q.get("tags", [])])
        q_required_set = set([r.lower() for r in q.get("required_skills", [])])
        
        # skill overlap counts matches against both tags and required_skills
        overlap = len((q_tags_set | q_required_set) & user_skills_set)
        
        # difficulty penalty
        difficulty_penalty = 0.9 if q.get("difficulty") in [1, 5] else 1.0
        
        score = overlap * difficulty_penalty
        scored.append((q, score))
    
    # Shuffle to avoid serial ranking when many ties
    random.shuffle(scored)
    
    # 4) Rank by score (descending) and pick top-k
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    top_k = ranked[:k]
    print(f"[INFO] Selected Top {k} Questions")
    
    # 5) Build full output - Ready for MongoDB
    result = {
        "user_id": user_json.get("_id"),
        "domain": domain,
        "session_type": "mock_session",
        "session_number": 1,
        "phase": "cold_start",
        "message": "Initial skill-based assessment",
        "questions_recommended": [],
        "question_ids": []  # For easy tracking
    }
    
    for q, score in top_k:
        result["questions_recommended"].append({
            "question": q,
            "score": round(score, 2),
            "reasoning": f"Matched tags/required_skills: {', '.join((q.get('tags') or []) + (q.get('required_skills') or [])) or 'No direct match'}"
        })
        result["question_ids"].append(q["_id"])
    
    print("=== MOCK SESSION READY ===\n")
    return result




def main(input_json):
    # Python receives data as dict
    return mock_session_recommendations(input_json, k=5)


user = {
        "_id": "user_001",
        "skills": ['java', 'nlp', 'tensorflow', 'data-structures'],
        "target_domain": "ai_ml"
    }
    
mock_result = mock_session_recommendations(user, k=5)
print(json.dumps(mock_result, indent=2))
print()