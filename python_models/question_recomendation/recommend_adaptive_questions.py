# # import random
# # import json
# # from utils import load_domain_questions

# # # -------------------------
# # # ADAPTIVE QUESTION RECOMMENDER
# # # -------------------------
# # def recommend_adaptive_questions(domain_key, session_summary, already_asked_ids=None, k=5):
# #     """
# #     STATELESS: Adaptive recommendation based on completed session.
# #     All state (asked questions, session history) passed as parameters.
    
# #     Args:
# #         domain_key: Domain identifier (e.g., 'aiml', 'devops')
# #         session_summary: Completed session summary with skill analysis
# #         already_asked_ids: List of question IDs already asked (from MongoDB)
# #         k: Number of questions to recommend
    
# #     Returns:
# #         JSON object to store in MongoDB
# #     """
# #     print("\n=== ADAPTIVE QUESTION RECOMMENDER START ===")
# #     print(f"[INFO] Domain: {domain_key}")
    
# #     # Initialize already asked set
# #     if already_asked_ids is None:
# #         already_asked_ids = []
    
# #     already_asked_set = set(already_asked_ids)
    
# #     # Add current session questions to already asked
# #     for answer in session_summary.get("answers", []):
# #         already_asked_set.add(answer["_id"])
    
# #     print(f"[INFO] Total questions to avoid: {len(already_asked_set)}")
    
# #     # Load all domain questions
# #     questions = load_domain_questions(domain_key)
# #     print(f"[INFO] Loaded {len(questions)} questions from domain '{domain_key}'")
    
# #     # Filter out already asked questions
# #     available_questions = [q for q in questions if q["_id"] not in already_asked_set]
# #     print(f"[INFO] Available questions (not yet asked): {len(available_questions)}")
    
# #     if len(available_questions) == 0:
# #         print("[WARNING] No more questions available in this domain!")
# #         return {
# #             "user_id": session_summary.get("user_id"),
# #             "domain": domain_key,
# #             "session_type": "adaptive_recommendation",
# #             "session_number": session_summary.get("session_number", 1) + 1,
# #             "message": "No more questions available in this domain",
# #             "questions_recommended": [],
# #             "question_ids": []
# #         }
    
# #     # Extract skill analysis
# #     skill_analysis = session_summary.get("skill_analysis", {})
# #     weaker_skills = skill_analysis.get("weaker_skills", [])
# #     stronger_skills = skill_analysis.get("stronger_skills", [])
# #     skill_averages = skill_analysis.get("skill_averages", {})
    
# #     print(f"[INFO] Weaker Skills Identified: {weaker_skills}")
# #     print(f"[INFO] Stronger Skills Identified: {stronger_skills}")
    
# #     # -------------------------
# #     # SCORING STRATEGY
# #     # -------------------------
# #     scored_questions = []
    
# #     for q in available_questions:
# #         q_skills = set(s.lower() for s in q.get("required_skills", []))
# #         q_tags = set(t.lower() for t in q.get("tags", []))
# #         q_all_skills = q_skills | q_tags
        
# #         # Initialize score components
# #         remedial_score = 0
# #         progressive_score = 0
# #         difficulty = q.get("difficulty", 3)
        
# #         # REMEDIAL: Target weaker skills with appropriate difficulty
# #         weaker_overlap = sum(1 for skill in weaker_skills if skill.lower() in q_all_skills)
# #         if weaker_overlap > 0:
# #             # For weak skills, prefer medium difficulty (2-3)
# #             if difficulty in [2, 3]:
# #                 remedial_score = weaker_overlap * 3.0
# #             elif difficulty == 1:
# #                 remedial_score = weaker_overlap * 2.0
# #             else:
# #                 remedial_score = weaker_overlap * 1.0
        
# #         # PROGRESSIVE: Challenge stronger skills with harder questions
# #         stronger_overlap = sum(1 for skill in stronger_skills if skill.lower() in q_all_skills)
# #         if stronger_overlap > 0:
# #             # For strong skills, prefer harder difficulty (3-5)
# #             if difficulty in [4, 5]:
# #                 progressive_score = stronger_overlap * 3.0
# #             elif difficulty == 3:
# #                 progressive_score = stronger_overlap * 2.0
# #             else:
# #                 progressive_score = stronger_overlap * 1.0
        
# #         # EXPLORATORY: Include some questions on untested skills
# #         exploratory_score = 0
# #         tested_skills = set(skill_averages.keys())
# #         untested_overlap = len(q_all_skills - set(s.lower() for s in tested_skills))
# #         if untested_overlap > 0:
# #             exploratory_score = untested_overlap * 1.5
        
# #         # Total score with weights
# #         total_score = (
# #             remedial_score * 0.5 +      # 50% weight on fixing weaknesses
# #             progressive_score * 0.3 +    # 30% weight on advancing strengths
# #             exploratory_score * 0.2      # 20% weight on exploring new areas
# #         )
        
# #         # Build reasoning
# #         reasoning_parts = []
# #         if remedial_score > 0:
# #             matching_weak = [s for s in weaker_skills if s.lower() in q_all_skills]
# #             reasoning_parts.append(f"Targets weak skills: {', '.join(matching_weak)}")
# #         if progressive_score > 0:
# #             matching_strong = [s for s in stronger_skills if s.lower() in q_all_skills]
# #             reasoning_parts.append(f"Advances strong skills: {', '.join(matching_strong)}")
# #         if exploratory_score > 0:
# #             reasoning_parts.append(f"Explores new skills")
        
# #         reasoning = " | ".join(reasoning_parts) if reasoning_parts else "General coverage"
        
# #         scored_questions.append({
# #             "question": q,
# #             "score": total_score,
# #             "remedial_score": remedial_score,
# #             "progressive_score": progressive_score,
# #             "exploratory_score": exploratory_score,
# #             "reasoning": reasoning
# #         })
    
# #     # -------------------------
# #     # RANKING & SELECTION
# #     # -------------------------
# #     # Shuffle to add randomness for ties
# #     random.shuffle(scored_questions)
    
# #     # Sort by total score (descending)
# #     ranked_questions = sorted(scored_questions, key=lambda x: x["score"], reverse=True)
    
# #     # Select top-k (or all available if less than k)
# #     top_k = ranked_questions[:min(k, len(ranked_questions))]
    
# #     print(f"[INFO] Selected Top {len(top_k)} Adaptive Questions")
    
# #     # -------------------------
# #     # BUILD RESULT - Ready for MongoDB
# #     # -------------------------
# #     result = {
# #         "user_id": session_summary.get("user_id"),
# #         "domain": domain_key,
# #         "session_type": "adaptive_recommendation",
# #         "session_number": session_summary.get("session_number", 1) + 1,
# #         "message": "Personalized adaptive questions based on your performance",
# #         "skill_analysis": skill_analysis,
# #         "questions_recommended": [],
# #         "question_ids": []
# #     }
    
# #     for item in top_k:
# #         result["questions_recommended"].append({
# #             "question": item["question"],
# #             "score": round(item["score"], 2),
# #             "score_breakdown": {
# #                 "remedial": round(item["remedial_score"], 2),
# #                 "progressive": round(item["progressive_score"], 2),
# #                 "exploratory": round(item["exploratory_score"], 2)
# #             },
# #             "reasoning": item["reasoning"]
# #         })
# #         result["question_ids"].append(item["question"]["_id"])
    
# #     print("=== ADAPTIVE QUESTION RECOMMENDER READY ===\n")
# #     return result



# # # ============================================================
# # # EXAMPLE 4: ADAPTIVE RECOMMENDATIONS
# # # ============================================================
# # print("\n\n[EXAMPLE 4] Adaptive Recommendations")
# # print("=" * 70)

# # # JavaScript would fetch this from MongoDB
# # already_asked = ["q_ai_0001", "q_ai_subj_0006"]

# # adaptive_result = recommend_adaptive_questions(
# #     domain_key="aiml",
# #     session_summary=summary,
# #     already_asked_ids=already_asked,
# #     k=5
# # )

# # print("\n📦 Result (Store this in MongoDB):")
# # print(json.dumps(adaptive_result, indent=2))



# import random
# import json

# # -------------------------
# # ADAPTIVE QUESTION RECOMMENDER
# # -------------------------
# def recommend_adaptive_questions(domain_key, session_summary, questions_db, already_asked_ids=None, k=5):
#     """
#     Enhanced adaptive recommendation based on completed session analysis.
#     Supports multiple question types: multiple-choice, subjective, voice, coding.
    
#     Args:
#         domain_key: Domain identifier (e.g., 'aiml', 'devops')
#         session_summary: Completed session summary with skill analysis
#         questions_db: List of all available questions in the domain
#         already_asked_ids: List of question IDs already asked (from MongoDB)
#         k: Number of questions to recommend
    
#     Returns:
#         JSON object with personalized question recommendations
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
#     print(f"[INFO] Total questions in database: {len(questions_db)}")
    
#     # Filter out already asked questions
#     available_questions = [q for q in questions_db if q["_id"] not in already_asked_set]
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
#             "question_ids": [],
#             "recommendations": session_summary.get("recommendations", {})
#         }
    
#     # Extract skill analysis
#     skill_analysis = session_summary.get("skill_analysis", {})
#     skill_details = skill_analysis.get("skill_details", {})
#     weaker_skills = skill_analysis.get("weaker_skills", [])
#     moderate_skills = skill_analysis.get("moderate_skills", [])
#     stronger_skills = skill_analysis.get("stronger_skills", [])
    
#     # Extract recommendations
#     recommendations = session_summary.get("recommendations", {})
#     focus_skills = recommendations.get("focus_skills", weaker_skills)
#     suggested_difficulty = recommendations.get("suggested_difficulty", "Intermediate")
    
#     print(f"[INFO] Focus Skills: {focus_skills}")
#     print(f"[INFO] Moderate Skills: {moderate_skills}")
#     print(f"[INFO] Stronger Skills: {stronger_skills}")
#     print(f"[INFO] Suggested Difficulty: {suggested_difficulty}")
    
#     # Map difficulty labels to numeric values
#     difficulty_map = {
#         "Novice": 2,
#         "Intermediate": 3,
#         "Advanced": 4,
#         "Expert": 5
#     }
#     target_difficulty = difficulty_map.get(suggested_difficulty, 3)
    
#     # -------------------------
#     # ENHANCED SCORING STRATEGY
#     # -------------------------
#     scored_questions = []
    
#     for q in available_questions:
#         # Extract question metadata
#         q_id = q["_id"]
#         q_type = q.get("question_type", "multiple-choice")
#         q_difficulty = q.get("difficulty", 3)
#         q_difficulty_label = q.get("difficulty_label", "Intermediate")
#         q_topic = q.get("topic", "")
        
#         # Extract skills from both required_skills and tags
#         q_required_skills = set(s.lower() for s in q.get("required_skills", []))
#         q_tags = set(t.lower() for t in q.get("tags", []))
#         q_all_skills = q_required_skills | q_tags
        
#         # Initialize score components
#         remedial_score = 0
#         progressive_score = 0
#         exploratory_score = 0
#         difficulty_match_score = 0
#         type_diversity_score = 0
        
#         # -------------------------
#         # 1. REMEDIAL SCORE: Target weaker skills
#         # -------------------------
#         weaker_overlap = sum(1 for skill in focus_skills if skill.lower() in q_all_skills)
#         if weaker_overlap > 0:
#             # For weak skills, prefer appropriate difficulty (not too hard)
#             if q_difficulty <= target_difficulty:
#                 remedial_score = weaker_overlap * 4.0
#             elif q_difficulty == target_difficulty + 1:
#                 remedial_score = weaker_overlap * 2.5
#             else:
#                 remedial_score = weaker_overlap * 1.0
        
#         # -------------------------
#         # 2. PROGRESSIVE SCORE: Challenge stronger skills
#         # -------------------------
#         stronger_overlap = sum(1 for skill in stronger_skills if skill.lower() in q_all_skills)
#         if stronger_overlap > 0:
#             # For strong skills, prefer harder questions
#             if q_difficulty >= target_difficulty:
#                 progressive_score = stronger_overlap * 3.5
#             elif q_difficulty == target_difficulty - 1:
#                 progressive_score = stronger_overlap * 2.0
#             else:
#                 progressive_score = stronger_overlap * 1.0
        
#         # -------------------------
#         # 3. MODERATE SKILLS: Practice and reinforce
#         # -------------------------
#         moderate_overlap = sum(1 for skill in moderate_skills if skill.lower() in q_all_skills)
#         if moderate_overlap > 0:
#             # For moderate skills, prefer questions at current difficulty
#             if q_difficulty == target_difficulty:
#                 remedial_score += moderate_overlap * 3.0
#             else:
#                 remedial_score += moderate_overlap * 1.5
        
#         # -------------------------
#         # 4. EXPLORATORY SCORE: New untested skills
#         # -------------------------
#         tested_skills = set(s.lower() for s in skill_details.keys())
#         untested_overlap = len(q_all_skills - tested_skills)
#         if untested_overlap > 0:
#             exploratory_score = untested_overlap * 1.8
        
#         # -------------------------
#         # 5. DIFFICULTY MATCH SCORE
#         # -------------------------
#         difficulty_diff = abs(q_difficulty - target_difficulty)
#         if difficulty_diff == 0:
#             difficulty_match_score = 3.0
#         elif difficulty_diff == 1:
#             difficulty_match_score = 1.5
#         else:
#             difficulty_match_score = 0.5
        
#         # -------------------------
#         # 6. TYPE DIVERSITY SCORE
#         # -------------------------
#         # Get types attempted in current session
#         attempted_types = set(ans["type"] for ans in session_summary.get("answers", []))
        
#         # Map question types
#         type_map = {
#             "multiple-choice": "mcq",
#             "subjective": "subjective",
#             "voice": "voice",
#             "coding": "coding"
#         }
#         normalized_type = type_map.get(q_type, q_type)
        
#         # Prefer diverse question types
#         if normalized_type not in attempted_types:
#             type_diversity_score = 2.0
#         else:
#             type_diversity_score = 0.5
        
#         # -------------------------
#         # CALCULATE TOTAL SCORE
#         # -------------------------
#         total_score = (
#             remedial_score * 0.40 +         # 40% weight on fixing weaknesses
#             progressive_score * 0.25 +      # 25% weight on advancing strengths
#             exploratory_score * 0.15 +      # 15% weight on exploring new areas
#             difficulty_match_score * 0.12 + # 12% weight on difficulty match
#             type_diversity_score * 0.08     # 8% weight on type diversity
#         )
        
#         # -------------------------
#         # BUILD REASONING
#         # -------------------------
#         reasoning_parts = []
        
#         if remedial_score > 0:
#             matching_weak = [s for s in focus_skills + moderate_skills if s.lower() in q_all_skills]
#             if matching_weak:
#                 reasoning_parts.append(f"Targets skills needing improvement: {', '.join(matching_weak[:2])}")
        
#         if progressive_score > 0:
#             matching_strong = [s for s in stronger_skills if s.lower() in q_all_skills]
#             if matching_strong:
#                 reasoning_parts.append(f"Challenges strong skills: {', '.join(matching_strong[:2])}")
        
#         if exploratory_score > 0:
#             reasoning_parts.append("Explores new skills")
        
#         if difficulty_match_score >= 1.5:
#             reasoning_parts.append(f"Appropriate difficulty ({q_difficulty_label})")
        
#         if type_diversity_score >= 1.5:
#             reasoning_parts.append(f"Diverse question type ({q_type})")
        
#         reasoning = " | ".join(reasoning_parts) if reasoning_parts else "General skill coverage"
        
#         scored_questions.append({
#             "question": q,
#             "total_score": total_score,
#             "score_breakdown": {
#                 "remedial": round(remedial_score, 2),
#                 "progressive": round(progressive_score, 2),
#                 "exploratory": round(exploratory_score, 2),
#                 "difficulty_match": round(difficulty_match_score, 2),
#                 "type_diversity": round(type_diversity_score, 2)
#             },
#             "reasoning": reasoning,
#             "skills_targeted": list(q_all_skills)
#         })
    
#     # -------------------------
#     # RANKING & SELECTION
#     # -------------------------
#     # === Smart Randomized Ranking (recommended) ===
#     ranked_questions = sorted(
#         scored_questions,
#         key=lambda x: (
#             x["total_score"] * 0.85   # 85% real relevance
#             + random.random() * 0.15  # 15% exploration randomness
#         ),
#         reverse=True
#     )

    
#     # Select top-k ensuring type diversity
#     selected = []
#     type_counts = {"mcq": 0, "subjective": 0, "voice": 0, "coding": 0}
#     max_per_type = max(2, k // 2)  # Allow at most half to be same type
    
#     for item in ranked_questions:
#         if len(selected) >= k:
#             break
        
#         q_type = item["question"].get("question_type", "multiple-choice")
#         normalized_type = {"multiple-choice": "mcq", "subjective": "subjective", 
#                           "voice": "voice", "coding": "coding"}.get(q_type, q_type)
        
#         # Enforce type diversity
#         if type_counts[normalized_type] < max_per_type:
#             selected.append(item)
#             type_counts[normalized_type] += 1
    
#     # If we don't have enough, add more without type constraint
#     if len(selected) < k:
#         for item in ranked_questions:
#             if len(selected) >= k:
#                 break
#             if item not in selected:
#                 selected.append(item)
    
#     print(f"[INFO] Selected {len(selected)} adaptive questions")
#     print(f"[INFO] Type distribution: {type_counts}")
    
#     # -------------------------
#     # BUILD RESULT - MongoDB Ready
#     # -------------------------
#     result = {
#         "user_id": session_summary.get("user_id"),
#         "domain": domain_key,
#         "session_type": "adaptive_recommendation",
#         "session_number": session_summary.get("session_number", 1) + 1,
#         "timestamp": None,  # Add when storing
#         "recommendation_basis": {
#             "focus_skills": focus_skills,
#             "moderate_skills": moderate_skills,
#             "stronger_skills": stronger_skills,
#             "suggested_difficulty": suggested_difficulty,
#             "previous_average": session_summary.get("session_stats", {}).get("overall_average", 0)
#         },
#         "questions_recommended": [],
#         "question_ids": [],
#         "metadata": {
#             "total_available": len(available_questions),
#             "total_selected": len(selected),
#             "selection_strategy": "adaptive_skill_based",
#             "type_distribution": type_counts
#         }
#     }
    
#     for idx, item in enumerate(selected, 1):
#         q = item["question"]
#         result["questions_recommended"].append({
#             "rank": idx,
#             "question_id": q["_id"],
#             "question_type": q.get("question_type", "multiple-choice"),
#             "topic": q.get("topic", ""),
#             "difficulty": q.get("difficulty", 3),
#             "difficulty_label": q.get("difficulty_label", "Intermediate"),
#             "estimated_time_sec": q.get("estimated_time_sec", 120),
#             "text": q.get("text", ""),
#             "skills_required": q.get("required_skills", []),
#             "tags": q.get("tags", []),
#             "recommendation_score": round(item["total_score"], 2),
#             "score_breakdown": item["score_breakdown"],
#             "reasoning": item["reasoning"]
#         })
#         result["question_ids"].append(q["_id"])
    
#     print("=== ADAPTIVE QUESTION RECOMMENDER COMPLETE ===\n")
#     return result


# # ============================================================
# # EXAMPLE USAGE WITH YOUR DATA
# # ============================================================

# # Sample session summary (from previous artifact)
# session_summary = {
#     "user_id": "user_12345",
#     "domain": "aiml",
#     "session_number": 1,
#     "session_stats": {
#         "overall_average": 51.8
#     },
#     "answers": [
#         {"_id": "q_ai_0031", "type": "subjective", "score": 75.2},
#         {"_id": "q_ai_0033", "type": "voice", "score": 32.0},
#         {"_id": "q_ai_0044_mcq", "type": "mcq", "score": 100},
#         {"_id": "q_ai_0042_mcq", "type": "mcq", "score": 0}
#     ],
#     "skill_analysis": {
#         "weaker_skills": ["Fundamentals", "Model Evaluation", "Overfitting"],
#         "moderate_skills": [],
#         "stronger_skills": ["Neural Networks", "Deep Learning", "Transfer Learning"],
#         "skill_details": {
#             "Fundamentals": {"average_score": 16.0},
#             "Model Evaluation": {"average_score": 16.0},
#             "Overfitting": {"average_score": 16.0},
#             "Deep Learning": {"average_score": 87.6},
#             "Transfer Learning": {"average_score": 87.6},
#             "Neural Networks": {"average_score": 75.2}
#         }
#     },
#     "recommendations": {
#         "focus_skills": ["Fundamentals", "Model Evaluation", "Overfitting"],
#         "suggested_difficulty": "Intermediate"
#     }
# }

# # Questions database with your format
# questions_db = [
#     {
#         "_id": "q_ai_0028",
#         "question_type": "multiple-choice",
#         "domain": "Artificial Intelligence & Machine Learning",
#         "topic": "Computer Vision",
#         "difficulty": 3,
#         "difficulty_label": "Intermediate",
#         "estimated_time_sec": 105,
#         "text": "After a convolution layer in a CNN, what is the primary role of a 'Max Pooling' layer?",
#         "options": [
#             "To increase the spatial dimensions of the feature map.",
#             "To reduce the spatial dimensions, providing translational invariance and reducing computation.",
#             "To apply an activation function to the outputs.",
#             "To add noise for regularization."
#         ],
#         "answer": "To reduce the spatial dimensions, providing translational invariance and reducing computation.",
#         "tags": ["Computer Vision", "Deep Learning"],
#         "required_skills": ["Computer Vision", "Deep Learning"]
#     },
#     {
#         "_id": "q_ai_0029",
#         "question_type": "multiple-choice",
#         "domain": "Artificial Intelligence & Machine Learning",
#         "topic": "Deep Learning",
#         "difficulty": 4,
#         "difficulty_label": "Advanced",
#         "estimated_time_sec": 150,
#         "text": "What is 'Dropout' and how does it prevent overfitting in neural networks?",
#         "options": [
#             "A technique to automatically stop training when loss increases.",
#             "It randomly reduces the learning rate during training.",
#             "It randomly sets a fraction of the neuron outputs to zero during training, preventing co-adaptation of features.",
#             "A method to normalize the input data features."
#         ],
#         "answer": "It randomly sets a fraction of the neuron outputs to zero during training, preventing co-adaptation of features.",
#         "tags": ["Deep Learning", "Optimization"],
#         "required_skills": ["Deep Learning"]
#     },
#     {
#         "_id": "q_ai_0032",
#         "question_type": "subjective",
#         "domain": "Artificial Intelligence & Machine Learning",
#         "topic": "Model Evaluation",
#         "difficulty": 4,
#         "difficulty_label": "Advanced",
#         "estimated_time_sec": 360,
#         "text": "Compare and contrast Precision, Recall, and F1-Score as evaluation metrics.",
#         "tags": ["Evaluation Metrics", "Model Performance"],
#         "required_skills": ["Python", "Machine Learning"]
#     },
#     {
#         "_id": "q_ai_0034",
#         "question_type": "voice",
#         "domain": "Artificial Intelligence & Machine Learning",
#         "topic": "NLP",
#         "difficulty": 3,
#         "difficulty_label": "Intermediate",
#         "estimated_time_sec": 180,
#         "text": "Describe the architecture of a Transformer model and explain why it revolutionized NLP.",
#         "tags": ["NLP", "Deep Learning", "Transformers"],
#         "required_skills": ["NLP", "Deep Learning"]
#     },
#     {
#         "_id": "q_ai_0035",
#         "question_type": "coding",
#         "domain": "Artificial Intelligence & Machine Learning",
#         "topic": "Python Libraries",
#         "difficulty": 2,
#         "difficulty_label": "Novice",
#         "estimated_time_sec": 300,
#         "text": "Write a Python function that takes a NumPy array and normalizes it using Min-Max scaling.",
#         "tags": ["Python", "NumPy", "Data Preprocessing"],
#         "required_skills": ["Python"]
#     }
# ]

# # Already asked questions
# already_asked_ids = ["q_ai_0031", "q_ai_0033", "q_ai_0044_mcq", "q_ai_0042_mcq"]

# # Generate adaptive recommendations
# recommendations = recommend_adaptive_questions(
#     domain_key="aiml",
#     session_summary=session_summary,
#     questions_db=questions_db,
#     already_asked_ids=already_asked_ids,
#     k=3
# )

# print("\n📦 MongoDB Document (Adaptive Recommendations):")
# print(json.dumps(recommendations, indent=2))






import random
import json
from collections import defaultdict
from typing import List, Dict, Set, Optional

# -------------------------
# ENHANCED ADAPTIVE QUESTION RECOMMENDER
# -------------------------
class AdaptiveRecommender:
    """
    Improved adaptive learning system with better skill tracking,
    performance analysis, and personalized question selection.
    """
    
    def __init__(self):
        self.difficulty_map = {
            "Novice": 1,
            "Beginner": 2,
            "Intermediate": 3,
            "Advanced": 4,
            "Expert": 5
        }
        
        # Performance thresholds
        self.mastery_threshold = 70.0  # Score >= 70 indicates mastery
        self.struggle_threshold = 40.0  # Score < 40 indicates struggle
        
    def analyze_skill_performance(self, session_data: Dict) -> Dict:
        """
        Deep analysis of skill performance patterns.
        Returns detailed insights about each skill.
        """
        skill_analysis = session_data.get("skill_analysis", {})
        skill_insights = {}
        
        for skill_name, skill_data in skill_analysis.items():
            # Robustness: Skip non-dict items (e.g. stronger_skills list or skill_averages map from Mongoose)
            if not isinstance(skill_data, dict):
                continue

            if skill_name in ["top_skills", "weak_skills", "stronger_skills", "weaker_skills"]:
                continue
                
            avg_score = skill_data.get("average_score", 0)
            questions_tested = skill_data.get("questions_tested", 0)
            questions = skill_data.get("questions", [])
            
            # Analyze question types attempted
            type_performance = defaultdict(list)
            for q in questions:
                q_type = q.get("type", "mcq")
                score = q.get("score", 0)
                type_performance[q_type].append(score)
            
            # Calculate consistency (standard deviation)
            scores = [q.get("score", 0) for q in questions]
            consistency = self._calculate_consistency(scores)
            
            # Determine skill status
            if avg_score >= self.mastery_threshold:
                status = "mastered"
                priority = "low"
            elif avg_score >= self.struggle_threshold:
                status = "developing"
                priority = "medium"
            else:
                status = "struggling"
                priority = "high"
            
            # Check if skill needs more diverse question types
            types_attempted = set(type_performance.keys())
            needs_diversity = len(types_attempted) < 2 and questions_tested >= 2
            
            skill_insights[skill_name] = {
                "average_score": avg_score,
                "questions_tested": questions_tested,
                "status": status,
                "priority": priority,
                "consistency": consistency,
                "type_performance": dict(type_performance),
                "types_attempted": list(types_attempted),
                "needs_diversity": needs_diversity,
                "min_score": skill_data.get("min_score", 0),
                "max_score": skill_data.get("max_score", 0)
            }
        
        return skill_insights
    
    def _calculate_consistency(self, scores: List[float]) -> str:
        """Calculate performance consistency."""
        if len(scores) < 2:
            return "insufficient_data"
        
        avg = sum(scores) / len(scores)
        variance = sum((x - avg) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Coefficient of variation
        if avg > 0:
            cv = (std_dev / avg) * 100
            if cv < 20:
                return "very_consistent"
            elif cv < 40:
                return "consistent"
            elif cv < 60:
                return "inconsistent"
            else:
                return "very_inconsistent"
        return "insufficient_data"
    
    def determine_optimal_difficulty(self, session_data: Dict, skill_insights: Dict) -> Dict[str, int]:
        """
        Determine optimal difficulty level for each skill category.
        Returns difficulty mapping for weak, moderate, and strong skills.
        """
        overall_avg = session_data.get("session_stats", {}).get("overall_average", 50)
        
        # Base difficulty on overall performance
        if overall_avg < 30:
            base_difficulty = 1  # Start with Novice
        elif overall_avg < 50:
            base_difficulty = 2  # Beginner
        elif overall_avg < 70:
            base_difficulty = 3  # Intermediate
        elif overall_avg < 85:
            base_difficulty = 4  # Advanced
        else:
            base_difficulty = 5  # Expert
        
        # Adjust for specific skill categories
        difficulty_strategy = {
            "struggling_skills": max(1, base_difficulty - 1),  # Easier questions
            "developing_skills": base_difficulty,               # Current level
            "mastered_skills": min(5, base_difficulty + 1)      # Challenge more
        }
        
        return difficulty_strategy
    
    def calculate_question_score(
        self,
        question: Dict,
        skill_insights: Dict,
        difficulty_strategy: Dict,
        session_data: Dict,
        attempted_types: Set[str]
    ) -> Dict:
        """
        Calculate comprehensive score for each question based on multiple factors.
        """
        q_id = question["_id"]
        q_type = question.get("question_type", "multiple-choice")
        q_difficulty = question.get("difficulty", 3)
        q_skills = set(s.lower() for s in question.get("required_skills", []))
        q_tags = set(t.lower() for t in question.get("tags", []))
        q_all_skills = q_skills | q_tags
        
        # Initialize score components
        scores = {
            "remedial": 0.0,
            "reinforcement": 0.0,
            "challenge": 0.0,
            "exploration": 0.0,
            "diversity": 0.0,
            "difficulty_match": 0.0,
            "urgency": 0.0
        }
        
        reasoning_parts = []
        matched_skills = []
        
        # Analyze each skill in the question
        for skill_key, insight in skill_insights.items():
            skill_lower = skill_key.lower()
            if skill_lower not in q_all_skills:
                continue
            
            matched_skills.append(skill_key)
            status = insight["status"]
            priority = insight["priority"]
            avg_score = insight["average_score"]
            consistency = insight["consistency"]
            
            # 1. REMEDIAL SCORE - For struggling skills
            if status == "struggling":
                # Higher score for appropriate difficulty
                if q_difficulty == difficulty_strategy["struggling_skills"]:
                    scores["remedial"] += 5.0
                    reasoning_parts.append(f"Remedial practice for {skill_key}")
                elif q_difficulty == difficulty_strategy["struggling_skills"] + 1:
                    scores["remedial"] += 2.5
                else:
                    scores["remedial"] += 0.5
                
                # Boost if skill has high priority
                if priority == "high":
                    scores["urgency"] += 3.0
            
            # 2. REINFORCEMENT SCORE - For developing skills
            elif status == "developing":
                if q_difficulty == difficulty_strategy["developing_skills"]:
                    scores["reinforcement"] += 4.0
                    reasoning_parts.append(f"Reinforces {skill_key}")
                elif abs(q_difficulty - difficulty_strategy["developing_skills"]) == 1:
                    scores["reinforcement"] += 2.0
                else:
                    scores["reinforcement"] += 0.5
            
            # 3. CHALLENGE SCORE - For mastered skills
            elif status == "mastered":
                if q_difficulty >= difficulty_strategy["mastered_skills"]:
                    scores["challenge"] += 3.5
                    reasoning_parts.append(f"Challenges {skill_key}")
                else:
                    scores["challenge"] += 1.0
            
            # Boost for inconsistent performance (needs more practice)
            if consistency in ["inconsistent", "very_inconsistent"]:
                scores["reinforcement"] += 1.5
                reasoning_parts.append(f"Stabilizes {skill_key}")
            
            # Check for type diversity needs
            if insight.get("needs_diversity", False):
                type_map = {
                    "multiple-choice": "mcq",
                    "subjective": "subjective",
                    "voice": "voice",
                    "coding": "coding"
                }
                normalized_type = type_map.get(q_type, q_type)
                
                if normalized_type not in insight.get("types_attempted", []):
                    scores["diversity"] += 2.5
                    reasoning_parts.append(f"New format for {skill_key}")
        
        # 4. EXPLORATION SCORE - Untested skills
        tested_skills = set(s.lower() for s in skill_insights.keys())
        untested_skills = q_all_skills - tested_skills
        if untested_skills:
            scores["exploration"] = len(untested_skills) * 2.0
            reasoning_parts.append(f"Explores new skills")
        
        # 5. TYPE DIVERSITY SCORE
        type_map = {
            "multiple-choice": "mcq",
            "subjective": "subjective",
            "voice": "voice",
            "coding": "coding"
        }
        normalized_type = type_map.get(q_type, q_type)
        
        if normalized_type not in attempted_types:
            scores["diversity"] += 2.0
        
        # Special boost for underrepresented types
        session_stats = session_data.get("session_stats", {})
        mcq_count = session_stats.get("mcq_attempted", 0)
        coding_count = session_stats.get("coding_attempted", 0)
        subjective_count = session_stats.get("subjective_attempted", 0)
        voice_count = session_stats.get("voice_attempted", 0)
        
        if normalized_type == "coding" and coding_count == 0:
            scores["diversity"] += 3.0
            reasoning_parts.append("Essential coding practice")
        elif normalized_type == "subjective" and subjective_count == 0:
            scores["diversity"] += 2.5
            reasoning_parts.append("Develops written explanation")
        elif normalized_type == "voice" and voice_count == 0:
            scores["diversity"] += 2.0
            reasoning_parts.append("Builds verbal communication")
        
        # 6. DIFFICULTY MATCH SCORE
        overall_avg = session_data.get("session_stats", {}).get("overall_average", 50)
        if overall_avg < 40:
            target_diff = difficulty_strategy["struggling_skills"]
        elif overall_avg < 70:
            target_diff = difficulty_strategy["developing_skills"]
        else:
            target_diff = difficulty_strategy["mastered_skills"]
        
        diff_distance = abs(q_difficulty - target_diff)
        if diff_distance == 0:
            scores["difficulty_match"] = 3.0
        elif diff_distance == 1:
            scores["difficulty_match"] = 1.5
        else:
            scores["difficulty_match"] = 0.3
        
        # Calculate weighted total score
        total_score = (
            scores["remedial"] * 0.30 +        # 30% - Fix critical gaps
            scores["reinforcement"] * 0.25 +   # 25% - Build confidence
            scores["challenge"] * 0.15 +       # 15% - Push boundaries
            scores["exploration"] * 0.10 +     # 10% - Discover new areas
            scores["diversity"] * 0.12 +       # 12% - Vary question types
            scores["difficulty_match"] * 0.05 + # 5% - Appropriate level
            scores["urgency"] * 0.03           # 3% - Priority boost
        )
        
        # Build reasoning
        reasoning = " | ".join(reasoning_parts) if reasoning_parts else "General skill development"
        
        return {
            "total_score": total_score,
            "scores": {k: round(v, 2) for k, v in scores.items()},
            "reasoning": reasoning,
            "matched_skills": matched_skills,
            "skills_targeted": list(q_all_skills)
        }
    
    def recommend_questions(
        self,
        domain_key: str,
        session_data: Dict,
        questions_db: List[Dict],
        already_asked_ids: Optional[List[str]] = None,
        k: int = 5
    ) -> Dict:
        """
        Main recommendation function with enhanced adaptive logic.
        """
        # print("\n" + "="*70)
        # print("ENHANCED ADAPTIVE QUESTION RECOMMENDER")
        # print("="*70)
        # print(f"Domain: {domain_key}")
        # print(f"User: {session_data.get('user_id', 'unknown')}")
        # print(f"Session: #{session_data.get('session_number', 1)}")
        
        # Initialize
        if already_asked_ids is None:
            already_asked_ids = []
        already_asked_set = set(already_asked_ids)
        
        # Filter available questions
        available_questions = [
            q for q in questions_db 
            if q["_id"] not in already_asked_set
        ]
        
        # print(f"\nQuestion Pool:")
        # print(f"  • Total in database: {len(questions_db)}")
        # print(f"  • Already asked: {len(already_asked_set)}")
        # print(f"  • Available: {len(available_questions)}")
        
        if len(available_questions) == 0:
            return self._empty_recommendation(session_data, domain_key)
        
        # Analyze skill performance
        skill_insights = self.analyze_skill_performance(session_data)
        
        # print(f"\nSkill Analysis:")
        # for skill, insight in skill_insights.items():
        #     print(f"  • {skill}: {insight['status']} "
        #           f"(avg: {insight['average_score']:.1f}, "
        #           f"priority: {insight['priority']})")
        
        # Determine difficulty strategy
        difficulty_strategy = self.determine_optimal_difficulty(
            session_data, skill_insights
        )
        
        # print(f"\nDifficulty Strategy:")
        # print(f"  • Struggling skills: Level {difficulty_strategy['struggling_skills']}")
        # print(f"  • Developing skills: Level {difficulty_strategy['developing_skills']}")
        # print(f"  • Mastered skills: Level {difficulty_strategy['mastered_skills']}")
        
        # Get attempted types
        attempted_types = set()
        session_stats = session_data.get("session_stats", {})
        if session_stats.get("mcq_attempted", 0) > 0:
            attempted_types.add("mcq")
        if session_stats.get("coding_attempted", 0) > 0:
            attempted_types.add("coding")
        if session_stats.get("subjective_attempted", 0) > 0:
            attempted_types.add("subjective")
        if session_stats.get("voice_attempted", 0) > 0:
            attempted_types.add("voice")
        
        # Score all questions
        scored_questions = []
        for question in available_questions:
            scoring_result = self.calculate_question_score(
                question,
                skill_insights,
                difficulty_strategy,
                session_data,
                attempted_types
            )
            
            scored_questions.append({
                "question": question,
                **scoring_result
            })
        
        # Smart ranking with exploration randomness
        ranked_questions = sorted(
            scored_questions,
            key=lambda x: x["total_score"] * 0.90 + random.random() * 0.10,
            reverse=True
        )
        
        # Select with diversity constraints
        selected = self._select_diverse_questions(
            ranked_questions, k, attempted_types
        )
        
        # print(f"\nRecommendation Results:")
        # print(f"  • Questions recommended: {len(selected)}")
        
        # Build result
        return self._build_result(
            session_data,
            domain_key,
            selected,
            skill_insights,
            difficulty_strategy,
            len(available_questions)
        )
    
    def _select_diverse_questions(
        self,
        ranked_questions: List[Dict],
        k: int,
        attempted_types: Set[str]
    ) -> List[Dict]:
        """Select questions ensuring type diversity."""
        selected = []
        type_counts = {"mcq": 0, "subjective": 0, "voice": 0, "coding": 0}
        max_per_type = max(2, k // 2)
        
        # First pass: prioritize high-scoring diverse questions
        for item in ranked_questions:
            if len(selected) >= k:
                break
            
            q_type = item["question"].get("question_type", "multiple-choice")
            type_map = {
                "multiple-choice": "mcq",
                "subjective": "subjective",
                "voice": "voice",
                "coding": "coding"
            }
            normalized_type = type_map.get(q_type, q_type)
            
            if type_counts[normalized_type] < max_per_type:
                selected.append(item)
                type_counts[normalized_type] += 1
        
        # Second pass: fill remaining slots
        if len(selected) < k:
            for item in ranked_questions:
                if len(selected) >= k:
                    break
                if item not in selected:
                    selected.append(item)
        
        return selected
    
    def _build_result(
        self,
        session_data: Dict,
        domain_key: str,
        selected: List[Dict],
        skill_insights: Dict,
        difficulty_strategy: Dict,
        total_available: int
    ) -> Dict:
        """Build final recommendation result."""
        
        # Categorize skills by status
        struggling = [s for s, i in skill_insights.items() if i["status"] == "struggling"]
        developing = [s for s, i in skill_insights.items() if i["status"] == "developing"]
        mastered = [s for s, i in skill_insights.items() if i["status"] == "mastered"]
        
        result = {
            "user_id": session_data.get("user_id"),
            "domain": domain_key,
            "session_type": "adaptive_recommendation",
            "session_number": session_data.get("session_number", 1) + 1,
            "timestamp": None,
            "recommendation_basis": {
                "struggling_skills": struggling,
                "developing_skills": developing,
                "mastered_skills": mastered,
                "difficulty_strategy": difficulty_strategy,
                "previous_performance": {
                    "overall_average": session_data.get("session_stats", {}).get("overall_average", 0),
                    "total_questions": session_data.get("session_stats", {}).get("total_questions", 0)
                }
            },
            "questions_recommended": [],
            "question_ids": [],
            "metadata": {
                "total_available": total_available,
                "total_selected": len(selected),
                "selection_strategy": "enhanced_adaptive_v2",
                "recommendation_quality": "personalized"
            }
        }
        
        
        for idx, item in enumerate(selected, 1):
            q = item["question"]
            result["questions_recommended"].append({
                "rank": idx,
                "_id": q["_id"],  # CRITICAL: Controller expects _id
                "question_id": q["_id"],
                "question_type": q.get("question_type", "multiple-choice"),
                "topic": q.get("topic", ""),
                "difficulty": q.get("difficulty", 3),
                "difficulty_label": q.get("difficulty_label", "Intermediate"),
                "estimated_time_sec": q.get("estimated_time_sec", 120),
                "text": q.get("text", ""),
                "options": q.get("options", []), # Include options if available
                "skills_required": q.get("required_skills", []),
                "tags": q.get("tags", []),
                "recommendation_score": round(item["total_score"], 2),
                "score_breakdown": item["scores"],
                "reasoning": item["reasoning"],
                "matched_skills": item["matched_skills"]
            })
            result["question_ids"].append(q["_id"])
        
        return result
    
    def _empty_recommendation(self, session_data: Dict, domain_key: str) -> Dict:
        """Return empty recommendation when no questions available."""
        # Ensure we return the expected structure even if empty
        return {
            "user_id": session_data.get("user_id"),
            "domain": domain_key,
            "session_type": "adaptive_recommendation",
            "session_number": session_data.get("session_number", 1) + 1,
            "message": "No more questions available in this domain",
            "questions_recommended": [],
            "question_ids": [],
            "metadata": {"status": "exhausted"}
        }


# ============================================================
# USAGE EXAMPLE WITH YOUR DATA
# ============================================================


# -------------------------
# BRIDGE ENTRY POINT
# -------------------------
def generate_adaptive_recommendations(data):
    """
    Entry point for Node.js bridge.
    """
    import sys
    print(f"[Python] generate_adaptive_recommendations called for domain: {data.get('domain')}", file=sys.stderr)
    
    try:
        user_id = data.get("user_id")
        domain = data.get("domain")
        session_data = data.get("last_session")
        questions_dir = data.get("questions_dir")
        
        # Handle already_asked_ids: prefer direct list, fallback to extraction from session
        already_asked_ids = data.get("already_asked_ids", [])
        if not already_asked_ids and session_data and "skill_analysis" in session_data:
            extracted_ids = set()
            for skill_data in session_data["skill_analysis"].values():
                if isinstance(skill_data, dict) and "questions" in skill_data:
                    for q in skill_data["questions"]:
                        if "question_id" in q:
                            extracted_ids.add(q["question_id"])
            if extracted_ids:
                already_asked_ids = list(extracted_ids)
                
        k = data.get("k", 5)
        
        if not session_data:
            print("[Python] Error: Missing last_session data", file=sys.stderr)
            return {"error": "Missing last_session data"}
            
        # Load questions db from file
        import os
        questions_file = os.path.join(questions_dir, f"{domain}.json")
        
        if not os.path.exists(questions_file):
             # Try resolving safely
             questions_file = os.path.join(questions_dir, f"{domain.split('_')[0]}_{domain.split('_')[1]}.json") if '_' in domain else os.path.join(questions_dir, f"{domain}.json")
             if not os.path.exists(questions_file):
                print(f"[Python] Error: Domain file not found at {questions_file}", file=sys.stderr)
                return {"error": f"Domain file not found: {questions_file}"}

        print(f"[Python] Loading questions from: {questions_file}", file=sys.stderr)

        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_db = json.load(f)
            
        print(f"[Python] Loaded {len(questions_db)} questions", file=sys.stderr)
            
        recommender = AdaptiveRecommender()
        recommendations = recommender.recommend_questions(
            domain_key=domain,
            session_data=session_data,
            questions_db=questions_db,
            already_asked_ids=already_asked_ids,
            k=k
        )
        
        print("[Python] Recommendations generated successfully", file=sys.stderr)
        return recommendations

    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"[Python] Exception in generate_adaptive_recommendations: {str(e)}\n{trace}", file=sys.stderr)
        return {"error": str(e), "traceback": trace}

if __name__ == "__main__":
    import sys
    
    # Read from stdin to support direct execution with JSON payload
    try:
        # Check if there's input from pipe
        if not sys.stdin.isatty():
            input_str = sys.stdin.read()
            if input_str.strip():
                data = json.loads(input_str)
                print("[Python] PARSED DATA:\n", json.dumps(data, indent=2), file=sys.stderr)

                # Call the bridge entry point function
                result = generate_adaptive_recommendations(data)
                
                # Print result to stdout
                print(json.dumps(result, indent=2))
            else:
                print("Error: Empty input received.", file=sys.stderr)
        else:
            print("Usage: echo 'JSON_PAYLOAD' | python recommend_adaptive_questions.py", file=sys.stderr)
            print("To test manually, pipe JSON data matching the structure sent by interviewController.js", file=sys.stderr)

    except Exception as e:
        print(f"Error executing script: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
