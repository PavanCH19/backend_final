# import json

# # -------------------------
# # BUILD USER SUMMARY FROM ANSWERS
# # -------------------------
# def build_user_summary_from_answers(user_id, domain, answers, session_number=1):
#     """
#     STATELESS: Build completed session summary with skill analysis.
#     Returns data for MongoDB storage.
    
#     Args:
#         user_id: User identifier
#         domain: Domain name (e.g., 'aiml', 'devops')
#         answers: List of answer objects with scores/evaluations
#         session_number: Session number for tracking
    
#     Returns:
#         JSON object to store in MongoDB
#     """
    
#     print(f"\n=== BUILD SESSION SUMMARY START ===")
#     print(f"[INFO] User: {user_id}, Domain: {domain}, Session: {session_number}")
    
#     mcq_count = 0
#     subj_count = 0
#     voice_count = 0
    
#     formatted_answers = []
#     skill_scores = {}
    
#     # -------------------------------
#     # PARSE ANSWERS
#     # -------------------------------
#     for item in answers:
#         qid = item["_id"]
#         skills = item.get("skills_required", [])
        
#         # Detect type
#         if "answer" in item and "evaluation" not in item and "transcription" not in item:
#             q_type = "mcq"
#             score = 100
#             mcq_count += 1
            
#             formatted_answers.append({
#                 "_id": qid,
#                 "type": "mcq",
#                 "answer": item["answer"],
#                 "score": score
#             })
        
#         elif "evaluation" in item and "transcription" not in item:
#             q_type = "subjective"
#             score = item["evaluation"]["overallScore"]
#             subj_count += 1
            
#             formatted_answers.append({
#                 "_id": qid,
#                 "type": "subjective",
#                 "answer": item["answer"],
#                 "evaluation": item["evaluation"],
#                 "score": score
#             })
        
#         elif "evaluation" in item and "transcription" in item:
#             q_type = "voice"
#             score = item["evaluation"]["overallScore"]
#             voice_count += 1
            
#             formatted_answers.append({
#                 "_id": qid,
#                 "type": "voice",
#                 "transcription": item["transcription"],
#                 "analytics": item["analytics"],
#                 "evaluation": item["evaluation"],
#                 "score": score
#             })
#         else:
#             score = 0
        
#         # Add to skill scores
#         for skill in skills:
#             if skill not in skill_scores:
#                 skill_scores[skill] = []
#             skill_scores[skill].append(score)
    
#     # -------------------------------
#     # SKILL ANALYSIS
#     # -------------------------------
#     # Compute average score for each skill
#     skill_avg = {
#         skill: sum(scores) / len(scores)
#         for skill, scores in skill_scores.items()
#     }
    
#     # Sort skills by score
#     sorted_skills = sorted(skill_avg.items(), key=lambda x: x[1])
    
#     total = len(sorted_skills)
#     cutoff = max(1, total // 3)
    
#     weaker_skills = [s for s, v in sorted_skills[:cutoff]]
#     stronger_skills = [s for s, v in sorted_skills[-cutoff:]]
    
#     print(f"[INFO] Skill Analysis Complete")
#     print(f"[INFO] Weaker Skills: {weaker_skills}")
#     print(f"[INFO] Stronger Skills: {stronger_skills}")
    
#     # -------------------------------
#     # FINAL OUTPUT JSON - Ready for MongoDB
#     # -------------------------------
#     result = {
#         "user_id": user_id,
#         "domain": domain,
#         "session_type": "completed_session",
#         "session_number": session_number,
#         "total_questions": len(answers),
#         "mcq_attempted": mcq_count,
#         "subjective_attempted": subj_count,
#         "voice_attempted": voice_count,
#         "answers": formatted_answers,
#         "skill_analysis": {
#             "stronger_skills": stronger_skills,
#             "weaker_skills": weaker_skills,
#             "skill_averages": skill_avg
#         }
#     }
    
#     print("=== SESSION SUMMARY READY ===\n")
#     return result

# answers = [
#   {
#     "_id": "q_ai_0001",
#     "topic": "Fundamentals",
#     "skills_required": ["Fundamentals", "Supervised Learning", "Python"],
#     "answer": "Data with labeled output/target variables"
#   },
#   {
#     "_id": "q_ai_0002",
#     "topic": "Deep Learning",
#     "skills_required": ["Deep Learning", "Neural Networks"],
#     "answer": "A single neuron with a threshold activation function"
#   },
#   {
#     "_id": "q_ai_0003",
#     "topic": "Python Libraries",
#     "skills_required": ["Python", "Libraries"],
#     "answer": "NumPy"
#   },
#   {
#     "_id": "q_ai_0004",
#     "topic": "Frameworks",
#     "skills_required": ["TensorFlow", "Keras"],
#     "answer": "TensorFlow is the underlying framework; Keras is a high-level API for rapid prototyping on top of it."
#   },
#   {
#     "_id": "q_ai_0005",
#     "topic": "Deep Learning",
#     "skills_required": ["Deep Learning", "Neural Networks"],
#     "answer": "Sigmoid"
#   },

#   {
#     "_id": "q_ai_subj_0006",
#     "topic": "Supervised Learning",
#     "skills_required": ["Supervised Learning", "Fundamentals", "Python"],
#     "answer": "Supervised learning is a process where a model learns from labeled data to predict outcomes based on learned patterns.",
#     "evaluation": {
#       "grammarScore": 100,
#       "relevanceScore": 96,
#       "completenessScore": 88,
#       "overallScore": 93,
#       "feedback": "Clear and accurate explanation. Adding an example would improve completeness."
#     }
#   },
#   {
#     "_id": "q_ai_subj_0007",
#     "topic": "Deep Learning",
#     "skills_required": ["Deep Learning", "Neural Networks"],
#     "answer": "A perceptron is the simplest neural network model that represents a single neuron capable of making binary decisions.",
#     "evaluation": {
#       "grammarScore": 100,
#       "relevanceScore": 94,
#       "completenessScore": 87,
#       "overallScore": 92,
#       "feedback": "Good definition. Adding its role in neural networks would strengthen the explanation."
#     }
#   },
#   {
#     "_id": "q_ai_voice_0008",
#     "topic": "Python Libraries",
#     "skills_required": ["Python", "Libraries"],
#     "transcription": "NumPy is the main Python library used for numerical operations and handling arrays in machine learning.",
#     "analytics": {
#       "speech_ratio_percent": 87.2,
#       "words_per_minute": 143.5,
#       "clarity_percent": 97.4,
#       "confidence_score_percent": 73.2
#     },
#     "evaluation": {
#       "grammarScore": 100,
#       "relevanceScore": 95,
#       "completenessScore": 90,
#       "overallScore": 94,
#       "feedback": "Concise and accurate statement."
#     }
#   },
#   {
#     "_id": "q_ai_voice_0009",
#     "topic": "Frameworks",
#     "skills_required": ["TensorFlow", "Keras"],
#     "transcription": "Keras is a high level API used for building models easily, while TensorFlow works underneath as the core backend framework.",
#     "analytics": {
#       "speech_ratio_percent": 85.5,
#       "words_per_minute": 138.2,
#       "clarity_percent": 96.8,
#       "confidence_score_percent": 71.5
#     },
#     "evaluation": {
#       "grammarScore": 100,
#       "relevanceScore": 96,
#       "completenessScore": 89,
#       "overallScore": 94,
#       "feedback": "Good clarity. Well expressed comparison."
#     }
#   },
#   {
#     "_id": "q_ai_voice_0010",
#     "topic": "Deep Learning",
#     "skills_required": ["Deep Learning", "Neural Networks"],
#     "transcription": "For binary classification problems, the sigmoid activation function is used because it produces probabilities between zero and one.",
#     "analytics": {
#       "speech_ratio_percent": 88.9,
#       "words_per_minute": 149.0,
#       "clarity_percent": 98.1,
#       "confidence_score_percent": 74.8
#     },
#     "evaluation": {
#       "grammarScore": 100,
#       "relevanceScore": 97,
#       "completenessScore": 93,
#       "overallScore": 96,
#       "feedback": "Great explanation with a clear reason."
#     }
#   }
# ]

# summary = build_user_summary_from_answers(
#         user_id="user_001",
#         domain="aiml",
#         answers=answers,
#         session_number=1
#     )
    
# print("\n📦 Result (Store this in MongoDB):")
# print(json.dumps(summary, indent=2))





import json

def build_user_summary_from_answers(user_id, domain, evaluation_json, questions_db, session_number=1):
    """
    Enhanced version that processes the complete evaluation JSON structure.
    
    Args:
        user_id: User identifier
        domain: Domain name (e.g., 'aiml', 'devops')
        evaluation_json: Complete evaluation results JSON (with summary, detailed_results, mcq_questions, voice_analysis)
        questions_db: List of question objects containing skills_required field
        session_number: Session number for tracking
    
    Returns:
        JSON object to store in MongoDB with comprehensive skill analysis
    """
    
    print(f"\n=== BUILD SESSION SUMMARY START ===")
    print(f"[INFO] User: {user_id}, Domain: {domain}, Session: {session_number}")
    
    # Create question lookup dictionary
    questions_map = {q["_id"]: q for q in questions_db}
    
    mcq_count = 0
    subj_count = 0
    voice_count = 0
    
    formatted_answers = []
    skill_scores = {}
    skill_question_map = {}  # Track which questions test each skill
    
    # -------------------------------
    # PROCESS MCQ QUESTIONS
    # -------------------------------
    mcq_questions = evaluation_json.get("mcq_questions", [])
    
    for mcq in mcq_questions:
        qid = mcq["question_id"]
        question_info = questions_map.get(qid, {})
        skills = question_info.get("skills_required", [])
        
        # If no skills in DB, infer from topic
        if not skills:
            topic = question_info.get("topic", "General")
            skills = [topic]
        
        # Calculate score (100 if correct, 0 if wrong)
        is_correct = mcq["user_answer"] == mcq["answer"]
        score = 100 if is_correct else 0
        mcq_count += 1
        
        formatted_answers.append({
            "_id": qid,
            "type": "mcq",
            "question": mcq["question"],
            "user_answer": mcq["user_answer"],
            "correct_answer": mcq["answer"],
            "is_correct": is_correct,
            "score": score,
            "skills": skills
        })
        
        # Map scores to skills
        for skill in skills:
            if skill not in skill_scores:
                skill_scores[skill] = []
                skill_question_map[skill] = []
            skill_scores[skill].append(score)
            skill_question_map[skill].append({
                "question_id": qid,
                "score": score,
                "type": "mcq",
                "is_correct": is_correct
            })
    
    # -------------------------------
    # PROCESS SUBJECTIVE QUESTIONS
    # -------------------------------
    detailed_results = evaluation_json.get("detailed_results", [])
    
    for result in detailed_results:
        qid = result["question_id"]
        question_info = questions_map.get(qid, {})
        skills = question_info.get("skills_required", [])
        
        # If no skills in DB, infer from topic
        if not skills:
            topic = question_info.get("topic", "General")
            skills = [topic]
        
        score = result["score"]
        subj_count += 1
        
        formatted_answers.append({
            "_id": qid,
            "type": "subjective",
            "question": result["question_text"],
            "score": score,
            "grade": result["grade"],
            "performance_summary": result.get("performance_summary", {}),
            "detailed_feedback": result.get("detailed_feedback", {}),
            "criteria_performance": result.get("criteria_performance", []),
            "skills": skills
        })
        
        # Map scores to skills
        for skill in skills:
            if skill not in skill_scores:
                skill_scores[skill] = []
                skill_question_map[skill] = []
            skill_scores[skill].append(score)
            skill_question_map[skill].append({
                "question_id": qid,
                "score": score,
                "type": "subjective",
                "grade": result["grade"]
            })
    
    # -------------------------------
    # PROCESS VOICE ANALYSIS (if present)
    # -------------------------------
    voice_analysis = evaluation_json.get("voice_analysis", {})
    if voice_analysis:
        voice_count = 1  # Assuming voice analysis is for the session overall
    
    # -------------------------------
    # COMPREHENSIVE SKILL ANALYSIS
    # -------------------------------
    skill_details = {}
    for skill, scores in skill_scores.items():
        avg_score = sum(scores) / len(scores)
        skill_details[skill] = {
            "average_score": round(avg_score, 2),
            "question_count": len(scores),
            "questions": skill_question_map[skill],
            "min_score": round(min(scores), 2),
            "max_score": round(max(scores), 2),
            "performance_level": (
                "Strong" if avg_score >= 75 else
                "Moderate" if avg_score >= 50 else
                "Needs Improvement"
            )
        }
    
    # Sort skills by average score
    sorted_skills = sorted(skill_details.items(), key=lambda x: x[1]["average_score"])
    
    # Categorize skills based on score ranges
    weaker_skills = []
    moderate_skills = []
    stronger_skills = []
    
    for skill, details in sorted_skills:
        avg = details["average_score"]
        if avg < 50:
            weaker_skills.append(skill)
        elif avg < 75:
            moderate_skills.append(skill)
        else:
            stronger_skills.append(skill)
    
    print(f"\n[INFO] Skill Analysis Complete")
    print(f"[INFO] Weaker Skills (< 50%): {weaker_skills}")
    print(f"[INFO] Moderate Skills (50-75%): {moderate_skills}")
    print(f"[INFO] Stronger Skills (≥ 75%): {stronger_skills}")
    
    # -------------------------------
    # OVERALL SESSION STATISTICS
    # -------------------------------
    summary = evaluation_json.get("summary", {})
    all_scores = [item["score"] for item in formatted_answers]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # -------------------------------
    # DETAILED FEEDBACK ANALYSIS
    # -------------------------------
    areas_needing_work = []
    strengths_found = []
    
    for result in detailed_results:
        feedback = result.get("detailed_feedback", {})
        
        # Extract areas for improvement
        improvements = feedback.get("areas_for_improvement", [])
        for improvement in improvements:
            area = improvement.get("area", "")
            if area and area not in areas_needing_work:
                areas_needing_work.append(area)
        
        # Extract strengths
        strengths = feedback.get("strengths", [])
        for strength in strengths:
            if strength and strength not in strengths_found:
                strengths_found.append(strength)
    
    # -------------------------------
    # RECOMMENDATIONS FOR NEXT SESSION
    # -------------------------------
    recommendations = {
        "focus_skills": weaker_skills[:3] if weaker_skills else moderate_skills[:2],
        "practice_skills": moderate_skills[:2] if moderate_skills else [],
        "maintain_skills": stronger_skills[:3] if stronger_skills else [],
        "suggested_difficulty": (
            "Novice" if overall_avg < 50 else
            "Intermediate" if overall_avg < 75 else
            "Advanced"
        ),
        "recommended_topics": list(set([
            questions_map[ans["_id"]].get("topic", "General") 
            for ans in formatted_answers 
            if ans["score"] < 60 and ans["_id"] in questions_map
        ])),
        "key_improvement_areas": areas_needing_work[:3],
        "learning_priorities": []
    }
    
    # Generate learning priorities based on weakest skills
    for skill in weaker_skills[:3]:
        details = skill_details.get(skill, {})
        recommendations["learning_priorities"].append({
            "skill": skill,
            "current_score": details.get("average_score", 0),
            "target_score": 75,
            "questions_attempted": details.get("question_count", 0),
            "recommended_practice": "Complete 3-5 more questions on this skill"
        })
    
    # -------------------------------
    # VOICE ANALYSIS INSIGHTS (if available)
    # -------------------------------
    voice_insights = None
    if voice_analysis:
        results = voice_analysis.get("results", {})
        voice_insights = {
            "confidence_score": results.get("confidence", {}).get("confidence_score_percent", 0),
            "clarity_score": results.get("clarity", {}).get("clarity_score_percent", 0),
            "speech_ratio": results.get("silence_metrics", {}).get("speech_ratio_percent", 0),
            "words_per_minute": results.get("speech_rate", {}).get("words_per_minute_actual", 0),
            "recommendations": []
        }
        
        # Add voice-specific recommendations
        if voice_insights["confidence_score"] < 60:
            voice_insights["recommendations"].append("Work on speaking with more confidence")
        if voice_insights["clarity_score"] < 90:
            voice_insights["recommendations"].append("Focus on clearer pronunciation")
        if voice_insights["speech_ratio"] < 70:
            voice_insights["recommendations"].append("Reduce pauses and maintain speech flow")
    
    # -------------------------------
    # FINAL OUTPUT JSON - Ready for MongoDB
    # -------------------------------
    result = {
        "user_id": user_id,
        "domain": domain,
        "session_type": "completed_session",
        "session_number": session_number,
        "timestamp": None,  # Add timestamp when storing
        
        "session_stats": {
            "total_questions": summary.get("total_questions", len(formatted_answers)),
            "questions_answered": summary.get("answered", len(formatted_answers)),
            "mcq_attempted": mcq_count,
            "subjective_attempted": subj_count,
            "voice_attempted": voice_count,
            "overall_average": round(overall_avg, 2),
            "grade_distribution": summary.get("grade_distribution", {})
        },
        
        # "answers": formatted_answers,
        
        "skill_analysis": {
            "stronger_skills": stronger_skills,
            "moderate_skills": moderate_skills,
            "weaker_skills": weaker_skills,
            "skill_details": skill_details,
            "total_skills_assessed": len(skill_details)
        },
        
        "performance_insights": {
            "strengths": strengths_found,
            "areas_for_improvement": areas_needing_work,
            "overall_level": recommendations["suggested_difficulty"]
        },
        
        "recommendations": recommendations,
        
        "voice_analysis": voice_insights
    }
    
    print("\n=== SESSION SUMMARY READY ===")
    print(f"Overall Average Score: {overall_avg:.2f}%")
    print(f"Focus Areas: {recommendations['focus_skills']}")
    print(f"Suggested Difficulty: {recommendations['suggested_difficulty']}\n")
    
    return result


# -------------------------
# EXAMPLE USAGE WITH YOUR ACTUAL DATA
# -------------------------

# Your complete evaluation JSON
evaluation_json = {
    "summary": {
        "total_questions": 4,
        "answered": 2,
        "average_score": 53.6,
        "grade_distribution": {
            "C": 1,
            "F": 1
        }
    },
    "mcq_questions": [
        {
            "question_id": "q_ai_0044_mcq",
            "question": "Which statement best describes transfer learning?",
            "answer": "Using a pre-trained model on a new related task",
            "user_answer": "Using a pre-trained model on a new related task"
        },
        {
            "question_id": "q_ai_0042_mcq",
            "question": "How do you detect overfitting in a machine learning model?",
            "answer": "Training accuracy is high but validation accuracy is low",
            "user_answer": "Training accuracy is low and validation accuracy is high"
        }
    ],
    "detailed_results": [
        {
            "question_id": "q_ai_0031",
            "question_text": "Explain the concept of transfer learning in deep learning...",
            "score": 75.2,
            "grade": "C",
            "performance_summary": {
                "key_concepts": "4/4 covered (100.0%)",
                "examples": "1/2 provided (50.0%)",
                "word_count": "82/150 words",
                "criteria_score": "67.3%"
            },
            "detailed_feedback": {
                "overall_assessment": "Good answer showing solid understanding. Focus on the areas below to improve.",
                "strengths": ["✓ Strong coverage of key concepts (4/4)"],
                "areas_for_improvement": [
                    {"area": "Missing Examples/Scenarios"},
                    {"area": "Answer Length"},
                    {"area": "Evaluation Criteria"}
                ]
            },
            "criteria_performance": [
                {"criterion": "Conceptual Understanding", "score": 100.0, "weight": 0.4},
                {"criterion": "Real-world Application", "score": 50.0, "weight": 0.3},
                {"criterion": "Technical Depth", "score": 41.0, "weight": 0.3}
            ]
        },
        {
            "question_id": "q_ai_0033",
            "question_text": "Explain in your own words what overfitting is...",
            "score": 32.0,
            "grade": "F",
            "performance_summary": {
                "key_concepts": "2/4 covered (50.0%)",
                "examples": "0/0 provided (0.0%)",
                "word_count": "29/100 words",
                "criteria_score": "40.0%"
            },
            "detailed_feedback": {
                "overall_assessment": "Your answer needs significant improvement.",
                "strengths": [],
                "areas_for_improvement": [
                    {"area": "Missing Key Concepts"},
                    {"area": "Partially Covered Concepts"},
                    {"area": "Answer Length"},
                    {"area": "Evaluation Criteria"}
                ]
            },
            "criteria_performance": [
                {"criterion": "Clarity", "score": 100, "weight": 0.3},
                {"criterion": "Completeness", "score": 25.0, "weight": 0.4},
                {"criterion": "Examples", "score": 0.0, "weight": 0.3}
            ]
        }
    ],
    "voice_analysis": {
        "results": {
            "confidence": {"confidence_score_percent": 56.08},
            "clarity": {"clarity_score_percent": 98.67},
            "silence_metrics": {"speech_ratio_percent": 82.99},
            "speech_rate": {"words_per_minute_actual": 40.13}
        }
    }
}

# Your questions database with skills
questions_db = [
    {
        "_id": "q_ai_0031",
        "question_type": "subjective",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Deep Learning",
        "difficulty": 3,
        "difficulty_label": "Intermediate",
        "skills_required": ["Deep Learning", "Transfer Learning", "Neural Networks"]
    },
    {
        "_id": "q_ai_0033",
        "question_type": "subjective",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Fundamentals",
        "difficulty": 2,
        "difficulty_label": "Novice",
        "skills_required": ["Fundamentals", "Model Evaluation", "Overfitting"]
    },
    {
        "_id": "q_ai_0044_mcq",
        "question_type": "mcq",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Deep Learning",
        "difficulty": 3,
        "skills_required": ["Deep Learning", "Transfer Learning"]
    },
    {
        "_id": "q_ai_0042_mcq",
        "question_type": "mcq",
        "domain": "Artificial Intelligence & Machine Learning",
        "topic": "Fundamentals",
        "difficulty": 2,
        "skills_required": ["Fundamentals", "Model Evaluation", "Overfitting"]
    }
]

# Generate comprehensive summary
summary = build_user_summary_from_answers(
    user_id="user_12345",
    domain="aiml",
    evaluation_json=evaluation_json,
    questions_db=questions_db,
    session_number=1
)

print("\n📦 Complete MongoDB Document:")
print(json.dumps(summary, indent=2))