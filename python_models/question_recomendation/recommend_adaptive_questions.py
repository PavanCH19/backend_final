# import random
# import json
# from utils import load_domain_questions

# # -------------------------
# # ADAPTIVE QUESTION RECOMMENDER
# # -------------------------
# def recommend_adaptive_questions(domain_key, session_summary, already_asked_ids=None, k=5):
#     """
#     STATELESS: Adaptive recommendation based on completed session.
#     All state (asked questions, session history) passed as parameters.
    
#     Args:
#         domain_key: Domain identifier (e.g., 'aiml', 'devops')
#         session_summary: Completed session summary with skill analysis
#         already_asked_ids: List of question IDs already asked (from MongoDB)
#         k: Number of questions to recommend
    
#     Returns:
#         JSON object to store in MongoDB
#     """
#     print("\n=== ADAPTIVE QUESTION RECOMMENDER START ===")
#     print(f"[INFO] Domain: {domain_key}")
    
#     # Initialize already asked set
#     if already_asked_ids is None:
#         already_asked_ids = []
    
#     already_asked_set = set(already_asked_ids)
    
#     # Add current session questions to already asked
#     for answer in session_summary.get("answers", []):
#         already_asked_set.add(answer["_id"])
    
#     print(f"[INFO] Total questions to avoid: {len(already_asked_set)}")
    
#     # Load all domain questions
#     questions = load_domain_questions(domain_key)
#     print(f"[INFO] Loaded {len(questions)} questions from domain '{domain_key}'")
    
#     # Filter out already asked questions
#     available_questions = [q for q in questions if q["_id"] not in already_asked_set]
#     print(f"[INFO] Available questions (not yet asked): {len(available_questions)}")
    
#     if len(available_questions) == 0:
#         print("[WARNING] No more questions available in this domain!")
#         return {
#             "user_id": session_summary.get("user_id"),
#             "domain": domain_key,
#             "session_type": "adaptive_recommendation",
#             "session_number": session_summary.get("session_number", 1) + 1,
#             "message": "No more questions available in this domain",
#             "questions_recommended": [],
#             "question_ids": []
#         }
    
#     # Extract skill analysis
#     skill_analysis = session_summary.get("skill_analysis", {})
#     weaker_skills = skill_analysis.get("weaker_skills", [])
#     stronger_skills = skill_analysis.get("stronger_skills", [])
#     skill_averages = skill_analysis.get("skill_averages", {})
    
#     print(f"[INFO] Weaker Skills Identified: {weaker_skills}")
#     print(f"[INFO] Stronger Skills Identified: {stronger_skills}")
    
#     # -------------------------
#     # SCORING STRATEGY
#     # -------------------------
#     scored_questions = []
    
#     for q in available_questions:
#         q_skills = set(s.lower() for s in q.get("required_skills", []))
#         q_tags = set(t.lower() for t in q.get("tags", []))
#         q_all_skills = q_skills | q_tags
        
#         # Initialize score components
#         remedial_score = 0
#         progressive_score = 0
#         difficulty = q.get("difficulty", 3)
        
#         # REMEDIAL: Target weaker skills with appropriate difficulty
#         weaker_overlap = sum(1 for skill in weaker_skills if skill.lower() in q_all_skills)
#         if weaker_overlap > 0:
#             # For weak skills, prefer medium difficulty (2-3)
#             if difficulty in [2, 3]:
#                 remedial_score = weaker_overlap * 3.0
#             elif difficulty == 1:
#                 remedial_score = weaker_overlap * 2.0
#             else:
#                 remedial_score = weaker_overlap * 1.0
        
#         # PROGRESSIVE: Challenge stronger skills with harder questions
#         stronger_overlap = sum(1 for skill in stronger_skills if skill.lower() in q_all_skills)
#         if stronger_overlap > 0:
#             # For strong skills, prefer harder difficulty (3-5)
#             if difficulty in [4, 5]:
#                 progressive_score = stronger_overlap * 3.0
#             elif difficulty == 3:
#                 progressive_score = stronger_overlap * 2.0
#             else:
#                 progressive_score = stronger_overlap * 1.0
        
#         # EXPLORATORY: Include some questions on untested skills
#         exploratory_score = 0
#         tested_skills = set(skill_averages.keys())
#         untested_overlap = len(q_all_skills - set(s.lower() for s in tested_skills))
#         if untested_overlap > 0:
#             exploratory_score = untested_overlap * 1.5
        
#         # Total score with weights
#         total_score = (
#             remedial_score * 0.5 +      # 50% weight on fixing weaknesses
#             progressive_score * 0.3 +    # 30% weight on advancing strengths
#             exploratory_score * 0.2      # 20% weight on exploring new areas
#         )
        
#         # Build reasoning
#         reasoning_parts = []
#         if remedial_score > 0:
#             matching_weak = [s for s in weaker_skills if s.lower() in q_all_skills]
#             reasoning_parts.append(f"Targets weak skills: {', '.join(matching_weak)}")
#         if progressive_score > 0:
#             matching_strong = [s for s in stronger_skills if s.lower() in q_all_skills]
#             reasoning_parts.append(f"Advances strong skills: {', '.join(matching_strong)}")
#         if exploratory_score > 0:
#             reasoning_parts.append(f"Explores new skills")
        
#         reasoning = " | ".join(reasoning_parts) if reasoning_parts else "General coverage"
        
#         scored_questions.append({
#             "question": q,
#             "score": total_score,
#             "remedial_score": remedial_score,
#             "progressive_score": progressive_score,
#             "exploratory_score": exploratory_score,
#             "reasoning": reasoning
#         })
    
#     # -------------------------
#     # RANKING & SELECTION
#     # -------------------------
#     # Shuffle to add randomness for ties
#     random.shuffle(scored_questions)
    
#     # Sort by total score (descending)
#     ranked_questions = sorted(scored_questions, key=lambda x: x["score"], reverse=True)
    
#     # Select top-k (or all available if less than k)
#     top_k = ranked_questions[:min(k, len(ranked_questions))]
    
#     print(f"[INFO] Selected Top {len(top_k)} Adaptive Questions")
    
#     # -------------------------
#     # BUILD RESULT - Ready for MongoDB
#     # -------------------------
#     result = {
#         "user_id": session_summary.get("user_id"),
#         "domain": domain_key,
#         "session_type": "adaptive_recommendation",
#         "session_number": session_summary.get("session_number", 1) + 1,
#         "message": "Personalized adaptive questions based on your performance",
#         "skill_analysis": skill_analysis,
#         "questions_recommended": [],
#         "question_ids": []
#     }
    
#     for item in top_k:
#         result["questions_recommended"].append({
#             "question": item["question"],
#             "score": round(item["score"], 2),
#             "score_breakdown": {
#                 "remedial": round(item["remedial_score"], 2),
#                 "progressive": round(item["progressive_score"], 2),
#                 "exploratory": round(item["exploratory_score"], 2)
#             },
#             "reasoning": item["reasoning"]
#         })
#         result["question_ids"].append(item["question"]["_id"])
    
#     print("=== ADAPTIVE QUESTION RECOMMENDER READY ===\n")
#     return result



# # ============================================================
# # EXAMPLE 4: ADAPTIVE RECOMMENDATIONS
# # ============================================================
# print("\n\n[EXAMPLE 4] Adaptive Recommendations")
# print("=" * 70)

# # JavaScript would fetch this from MongoDB
# already_asked = ["q_ai_0001", "q_ai_subj_0006"]

# adaptive_result = recommend_adaptive_questions(
#     domain_key="aiml",
#     session_summary=summary,
#     already_asked_ids=already_asked,
#     k=5
# )

# print("\n📦 Result (Store this in MongoDB):")
# print(json.dumps(adaptive_result, indent=2))



import random
import json

# -------------------------
# ADAPTIVE QUESTION RECOMMENDER
# -------------------------
def recommend_adaptive_questions(domain_key, session_summary, questions_db, already_asked_ids=None, k=5):
    """
    Enhanced adaptive recommendation based on completed session analysis.
    Supports multiple question types: multiple-choice, subjective, voice, coding.
    
    Args:
        domain_key: Domain identifier (e.g., 'aiml', 'devops')
        session_summary: Completed session summary with skill analysis
        questions_db: List of all available questions in the domain
        already_asked_ids: List of question IDs already asked (from MongoDB)
        k: Number of questions to recommend
    
    Returns:
        JSON object with personalized question recommendations
    """
    print("\n=== ADAPTIVE QUESTION RECOMMENDER START ===")
    print(f"[INFO] Domain: {domain_key}")
    
    # Initialize already asked set
    if already_asked_ids is None:
        already_asked_ids = []
    
    already_asked_set = set(already_asked_ids)
    
    # Add current session questions to already asked
    for answer in session_summary.get("answers", []):
        already_asked_set.add(answer["_id"])
    
    print(f"[INFO] Total questions to avoid: {len(already_asked_set)}")
    print(f"[INFO] Total questions in database: {len(questions_db)}")
    
    # Filter out already asked questions
    available_questions = [q for q in questions_db if q["_id"] not in already_asked_set]
    print(f"[INFO] Available questions (not yet asked): {len(available_questions)}")
    
    if len(available_questions) == 0:
        print("[WARNING] No more questions available in this domain!")
        return {
            "user_id": session_summary.get("user_id"),
            "domain": domain_key,
            "session_type": "adaptive_recommendation",
            "session_number": session_summary.get("session_number", 1) + 1,
            "message": "No more questions available in this domain",
            "questions_recommended": [],
            "question_ids": [],
            "recommendations": session_summary.get("recommendations", {})
        }
    
    # Extract skill analysis
    skill_analysis = session_summary.get("skill_analysis", {})
    skill_details = skill_analysis.get("skill_details", {})
    weaker_skills = skill_analysis.get("weaker_skills", [])
    moderate_skills = skill_analysis.get("moderate_skills", [])
    stronger_skills = skill_analysis.get("stronger_skills", [])
    
    # Extract recommendations
    recommendations = session_summary.get("recommendations", {})
    focus_skills = recommendations.get("focus_skills", weaker_skills)
    suggested_difficulty = recommendations.get("suggested_difficulty", "Intermediate")
    
    print(f"[INFO] Focus Skills: {focus_skills}")
    print(f"[INFO] Moderate Skills: {moderate_skills}")
    print(f"[INFO] Stronger Skills: {stronger_skills}")
    print(f"[INFO] Suggested Difficulty: {suggested_difficulty}")
    
    # Map difficulty labels to numeric values
    difficulty_map = {
        "Novice": 2,
        "Intermediate": 3,
        "Advanced": 4,
        "Expert": 5
    }
    target_difficulty = difficulty_map.get(suggested_difficulty, 3)
    
    # -------------------------
    # ENHANCED SCORING STRATEGY
    # -------------------------
    scored_questions = []
    
    for q in available_questions:
        # Extract question metadata
        q_id = q["_id"]
        q_type = q.get("question_type", "multiple-choice")
        q_difficulty = q.get("difficulty", 3)
        q_difficulty_label = q.get("difficulty_label", "Intermediate")
        q_topic = q.get("topic", "")
        
        # Extract skills from both required_skills and tags
        q_required_skills = set(s.lower() for s in q.get("required_skills", []))
        q_tags = set(t.lower() for t in q.get("tags", []))
        q_all_skills = q_required_skills | q_tags
        
        # Initialize score components
        remedial_score = 0
        progressive_score = 0
        exploratory_score = 0
        difficulty_match_score = 0
        type_diversity_score = 0
        
        # -------------------------
        # 1. REMEDIAL SCORE: Target weaker skills
        # -------------------------
        weaker_overlap = sum(1 for skill in focus_skills if skill.lower() in q_all_skills)
        if weaker_overlap > 0:
            # For weak skills, prefer appropriate difficulty (not too hard)
            if q_difficulty <= target_difficulty:
                remedial_score = weaker_overlap * 4.0
            elif q_difficulty == target_difficulty + 1:
                remedial_score = weaker_overlap * 2.5
            else:
                remedial_score = weaker_overlap * 1.0
        
        # -------------------------
        # 2. PROGRESSIVE SCORE: Challenge stronger skills
        # -------------------------
        stronger_overlap = sum(1 for skill in stronger_skills if skill.lower() in q_all_skills)
        if stronger_overlap > 0:
            # For strong skills, prefer harder questions
            if q_difficulty >= target_difficulty:
                progressive_score = stronger_overlap * 3.5
            elif q_difficulty == target_difficulty - 1:
                progressive_score = stronger_overlap * 2.0
            else:
                progressive_score = stronger_overlap * 1.0
        
        # -------------------------
        # 3. MODERATE SKILLS: Practice and reinforce
        # -------------------------
        moderate_overlap = sum(1 for skill in moderate_skills if skill.lower() in q_all_skills)
        if moderate_overlap > 0:
            # For moderate skills, prefer questions at current difficulty
            if q_difficulty == target_difficulty:
                remedial_score += moderate_overlap * 3.0
            else:
                remedial_score += moderate_overlap * 1.5
        
        # -------------------------
        # 4. EXPLORATORY SCORE: New untested skills
        # -------------------------
        tested_skills = set(s.lower() for s in skill_details.keys())
        untested_overlap = len(q_all_skills - tested_skills)
        if untested_overlap > 0:
            exploratory_score = untested_overlap * 1.8
        
        # -------------------------
        # 5. DIFFICULTY MATCH SCORE
        # -------------------------
        difficulty_diff = abs(q_difficulty - target_difficulty)
        if difficulty_diff == 0:
            difficulty_match_score = 3.0
        elif difficulty_diff == 1:
            difficulty_match_score = 1.5
        else:
            difficulty_match_score = 0.5
        
        # -------------------------
        # 6. TYPE DIVERSITY SCORE
        # -------------------------
        # Get types attempted in current session
        attempted_types = set(ans["type"] for ans in session_summary.get("answers", []))
        
        # Map question types
        type_map = {
            "multiple-choice": "mcq",
            "subjective": "subjective",
            "voice": "voice",
            "coding": "coding"
        }
        normalized_type = type_map.get(q_type, q_type)
        
        # Prefer diverse question types
        if normalized_type not in attempted_types:
            type_diversity_score = 2.0
        else:
            type_diversity_score = 0.5
        
        # -------------------------
        # CALCULATE TOTAL SCORE
        # -------------------------
        total_score = (
            remedial_score * 0.40 +         # 40% weight on fixing weaknesses
            progressive_score * 0.25 +      # 25% weight on advancing strengths
            exploratory_score * 0.15 +      # 15% weight on exploring new areas
            difficulty_match_score * 0.12 + # 12% weight on difficulty match
            type_diversity_score * 0.08     # 8% weight on type diversity
        )
        
        # -------------------------
        # BUILD REASONING
        # -------------------------
        reasoning_parts = []
        
        if remedial_score > 0:
            matching_weak = [s for s in focus_skills + moderate_skills if s.lower() in q_all_skills]
            if matching_weak:
                reasoning_parts.append(f"Targets skills needing improvement: {', '.join(matching_weak[:2])}")
        
        if progressive_score > 0:
            matching_strong = [s for s in stronger_skills if s.lower() in q_all_skills]
            if matching_strong:
                reasoning_parts.append(f"Challenges strong skills: {', '.join(matching_strong[:2])}")
        
        if exploratory_score > 0:
            reasoning_parts.append("Explores new skills")
        
        if difficulty_match_score >= 1.5:
            reasoning_parts.append(f"Appropriate difficulty ({q_difficulty_label})")
        
        if type_diversity_score >= 1.5:
            reasoning_parts.append(f"Diverse question type ({q_type})")
        
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "General skill coverage"
        
        scored_questions.append({
            "question": q,
            "total_score": total_score,
            "score_breakdown": {
                "remedial": round(remedial_score, 2),
                "progressive": round(progressive_score, 2),
                "exploratory": round(exploratory_score, 2),
                "difficulty_match": round(difficulty_match_score, 2),
                "type_diversity": round(type_diversity_score, 2)
            },
            "reasoning": reasoning,
            "skills_targeted": list(q_all_skills)
        })
    
    # -------------------------
    # RANKING & SELECTION
    # -------------------------
    # Shuffle to add randomness for ties
    random.shuffle(scored_questions)
    
    # Sort by total score (descending)
    ranked_questions = sorted(scored_questions, key=lambda x: x["total_score"], reverse=True)
    
    # Select top-k ensuring type diversity
    selected = []
    type_counts = {"mcq": 0, "subjective": 0, "voice": 0, "coding": 0}
    max_per_type = max(2, k // 2)  # Allow at most half to be same type
    
    for item in ranked_questions:
        if len(selected) >= k:
            break
        
        q_type = item["question"].get("question_type", "multiple-choice")
        normalized_type = {"multiple-choice": "mcq", "subjective": "subjective", 
                          "voice": "voice", "coding": "coding"}.get(q_type, q_type)
        
        # Enforce type diversity
        if type_counts[normalized_type] < max_per_type:
            selected.append(item)
            type_counts[normalized_type] += 1
    
    # If we don't have enough, add more without type constraint
    if len(selected) < k:
        for item in ranked_questions:
            if len(selected) >= k:
                break
            if item not in selected:
                selected.append(item)
    
    print(f"[INFO] Selected {len(selected)} adaptive questions")
    print(f"[INFO] Type distribution: {type_counts}")
    
    # -------------------------
    # BUILD RESULT - MongoDB Ready
    # -------------------------
    result = {
        "user_id": session_summary.get("user_id"),
        "domain": domain_key,
        "session_type": "adaptive_recommendation",
        "session_number": session_summary.get("session_number", 1) + 1,
        "timestamp": None,  # Add when storing
        "recommendation_basis": {
            "focus_skills": focus_skills,
            "moderate_skills": moderate_skills,
            "stronger_skills": stronger_skills,
            "suggested_difficulty": suggested_difficulty,
            "previous_average": session_summary.get("session_stats", {}).get("overall_average", 0)
        },
        "questions_recommended": [],
        "question_ids": [],
        "metadata": {
            "total_available": len(available_questions),
            "total_selected": len(selected),
            "selection_strategy": "adaptive_skill_based",
            "type_distribution": type_counts
        }
    }
    
    for idx, item in enumerate(selected, 1):
        q = item["question"]
        result["questions_recommended"].append({
            "rank": idx,
            "question_id": q["_id"],
            "question_type": q.get("question_type", "multiple-choice"),
            "topic": q.get("topic", ""),
            "difficulty": q.get("difficulty", 3),
            "difficulty_label": q.get("difficulty_label", "Intermediate"),
            "estimated_time_sec": q.get("estimated_time_sec", 120),
            "text": q.get("text", ""),
            "skills_required": q.get("required_skills", []),
            "tags": q.get("tags", []),
            "recommendation_score": round(item["total_score"], 2),
            "score_breakdown": item["score_breakdown"],
            "reasoning": item["reasoning"]
        })
        result["question_ids"].append(q["_id"])
    
    print("=== ADAPTIVE QUESTION RECOMMENDER COMPLETE ===\n")
    return result


# ============================================================
# EXAMPLE USAGE WITH YOUR DATA
# ============================================================

# Sample session summary (from previous artifact)
session_summary = {
    "user_id": "user_12345",
    "domain": "aiml",
    "session_number": 1,
    "session_stats": {
        "overall_average": 51.8
    },
    "answers": [
        {"_id": "q_ai_0031", "type": "subjective", "score": 75.2},
        {"_id": "q_ai_0033", "type": "voice", "score": 32.0},
        {"_id": "q_ai_0044_mcq", "type": "mcq", "score": 100},
        {"_id": "q_ai_0042_mcq", "type": "mcq", "score": 0}
    ],
    "skill_analysis": {
        "weaker_skills": ["Fundamentals", "Model Evaluation", "Overfitting"],
        "moderate_skills": [],
        "stronger_skills": ["Neural Networks", "Deep Learning", "Transfer Learning"],
        "skill_details": {
            "Fundamentals": {"average_score": 16.0},
            "Model Evaluation": {"average_score": 16.0},
            "Overfitting": {"average_score": 16.0},
            "Deep Learning": {"average_score": 87.6},
            "Transfer Learning": {"average_score": 87.6},
            "Neural Networks": {"average_score": 75.2}
        }
    },
    "recommendations": {
        "focus_skills": ["Fundamentals", "Model Evaluation", "Overfitting"],
        "suggested_difficulty": "Intermediate"
    }
}

# Questions database with your format
questions_db = [
    {
        "_id": "q_ai_0028",
        "question_type": "multiple-choice",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Computer Vision",
        "difficulty": 3,
        "difficulty_label": "Intermediate",
        "estimated_time_sec": 105,
        "text": "After a convolution layer in a CNN, what is the primary role of a 'Max Pooling' layer?",
        "options": [
            "To increase the spatial dimensions of the feature map.",
            "To reduce the spatial dimensions, providing translational invariance and reducing computation.",
            "To apply an activation function to the outputs.",
            "To add noise for regularization."
        ],
        "answer": "To reduce the spatial dimensions, providing translational invariance and reducing computation.",
        "tags": ["Computer Vision", "Deep Learning"],
        "required_skills": ["Computer Vision", "Deep Learning"]
    },
    {
        "_id": "q_ai_0029",
        "question_type": "multiple-choice",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Deep Learning",
        "difficulty": 4,
        "difficulty_label": "Advanced",
        "estimated_time_sec": 150,
        "text": "What is 'Dropout' and how does it prevent overfitting in neural networks?",
        "options": [
            "A technique to automatically stop training when loss increases.",
            "It randomly reduces the learning rate during training.",
            "It randomly sets a fraction of the neuron outputs to zero during training, preventing co-adaptation of features.",
            "A method to normalize the input data features."
        ],
        "answer": "It randomly sets a fraction of the neuron outputs to zero during training, preventing co-adaptation of features.",
        "tags": ["Deep Learning", "Optimization"],
        "required_skills": ["Deep Learning"]
    },
    {
        "_id": "q_ai_0032",
        "question_type": "subjective",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Model Evaluation",
        "difficulty": 4,
        "difficulty_label": "Advanced",
        "estimated_time_sec": 360,
        "text": "Compare and contrast Precision, Recall, and F1-Score as evaluation metrics.",
        "tags": ["Evaluation Metrics", "Model Performance"],
        "required_skills": ["Python", "Machine Learning"]
    },
    {
        "_id": "q_ai_0034",
        "question_type": "voice",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "NLP",
        "difficulty": 3,
        "difficulty_label": "Intermediate",
        "estimated_time_sec": 180,
        "text": "Describe the architecture of a Transformer model and explain why it revolutionized NLP.",
        "tags": ["NLP", "Deep Learning", "Transformers"],
        "required_skills": ["NLP", "Deep Learning"]
    },
    {
        "_id": "q_ai_0035",
        "question_type": "coding",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Python Libraries",
        "difficulty": 2,
        "difficulty_label": "Novice",
        "estimated_time_sec": 300,
        "text": "Write a Python function that takes a NumPy array and normalizes it using Min-Max scaling.",
        "tags": ["Python", "NumPy", "Data Preprocessing"],
        "required_skills": ["Python"]
    }
]

# Already asked questions
already_asked_ids = ["q_ai_0031", "q_ai_0033", "q_ai_0044_mcq", "q_ai_0042_mcq"]

# Generate adaptive recommendations
recommendations = recommend_adaptive_questions(
    domain_key="aiml",
    session_summary=session_summary,
    questions_db=questions_db,
    already_asked_ids=already_asked_ids,
    k=5
)

print("\n📦 MongoDB Document (Adaptive Recommendations):")
print(json.dumps(recommendations, indent=2))