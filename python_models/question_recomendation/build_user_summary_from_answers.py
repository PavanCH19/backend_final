import json
import os

def build_user_summary_from_answers(user_id, domain, evaluation_json, questions_dir, session_number=1):
    """
    Enhanced version that processes the complete evaluation JSON structure.
    
    Args:
        user_id: User identifier
        domain: Domain name (filename without extension, e.g., 'ai_ml')
        evaluation_json: Complete evaluation results JSON from the Node.js controller
        questions_dir: Directory path containing domain JSON files
        session_number: Session number for tracking
        
    Returns:
        Dictionary containing comprehensive session analysis
    """
    print(f"\n=== BUILD SESSION SUMMARY START ===")
    print(f"[INFO] User: {user_id}, Domain: {domain}, Session: {session_number}")
    
    # -------------------------------
    # LOAD QUESTIONS FOR DOMAIN
    # -------------------------------
    questions_db = []
    domain_file = os.path.join(questions_dir, f"{domain}.json")
    
    if os.path.exists(domain_file):
        try:
            with open(domain_file, 'r', encoding='utf-8') as f:
                questions_db = json.load(f)
            print(f"[INFO] Loaded {len(questions_db)} questions from {domain}.json")
        except Exception as e:
            print(f"[ERROR] Failed to load domain questions: {e}")
            return None
    else:
        print(f"[ERROR] Domain file not found: {domain_file}")
        return None
    
    # Create question lookup dictionary
    questions_map = {q["_id"]: q for q in questions_db}
    
    mcq_count = 0
    subj_count = 0
    voice_count = 0
    coding_count = 0
    
    formatted_answers = []
    skill_scores = {}
    skill_question_map = {}  # Track which questions test each skill
    
    # -------------------------------
    # PROCESS RESULTS from evaluation_json["results"]
    # -------------------------------
    results_list = evaluation_json.get("results", [])
    
    for result_item in results_list:
        qid = result_item.get("qid")
        q_type = result_item.get("question_type")
        eval_data = result_item.get("evaluation", {})
        
        question_info = questions_map.get(qid, {})
        skills = question_info.get("skills_required", []) or question_info.get("tags", [])
        
        # If no skills in DB, infer from topic
        if not skills:
            topic = question_info.get("topic", "General")
            skills = [topic]
        
        score = 0
        grade = "N/A"
        is_correct = False
        
        # --- MCQ ---
        if q_type in ["mcq", "multiple-choice"]:
            score = eval_data.get("score", 0)
            is_correct = eval_data.get("correct", False)
            mcq_count += 1
            
            formatted_answers.append({
                "_id": qid,
                "type": "mcq",
                "question": question_info.get("text", ""),
                "user_answer": eval_data.get("user_answer"),
                "correct_answer": eval_data.get("expected_answer"),
                "is_correct": is_correct,
                "score": score,
                "skills": skills
            })
            
            # Map for MCQ
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
        
        # --- SUBJECTIVE / CODING ---
        elif q_type in ["subjective", "coding"]:
            # Check structure of eval_data (might be nested under 'result' due to evaluate_text output)
            text_result = eval_data.get("result", {}) if "result" in eval_data else eval_data
            score = text_result.get("score", 0)
            grade = text_result.get("grade", "F")
            
            if q_type == "subjective":
                subj_count += 1
            else:
                coding_count += 1
            
            formatted_answers.append({
                "_id": qid,
                "type": q_type,
                "question": question_info.get("text", "") or text_result.get("question_text", ""),
                "score": score,
                "grade": grade,
                "performance_summary": text_result.get("performance_summary", {}),
                "detailed_feedback": text_result.get("detailed_feedback", {}),
                "criteria_performance": text_result.get("criteria_performance", []),
                "skills": skills
            })
            
            # Map for Subjective/Coding
            for skill in skills:
                if skill not in skill_scores:
                    skill_scores[skill] = []
                    skill_question_map[skill] = []
                skill_scores[skill].append(score)
                skill_question_map[skill].append({
                    "question_id": qid,
                    "score": score,
                    "type": q_type,
                    "grade": grade
                })
        
        # --- VOICE ---
        elif q_type == "voice":
            voice_count += 1
            voice_res = eval_data.get("voice_analysis", {}).get("result", {})
            text_res = eval_data.get("text_evaluation", {}) or {}
            
            # Composite score: e.g. 40% voice confidence + 60% content score if available
            text_score = text_res.get("score", 0) if text_res else 0
            voice_conf = voice_res.get("confidence", {}).get("confidence_score_percent", 0)
            
            # Heuristic: mostly content based if transcribed successfully
            score = text_score if text_res else voice_conf
            
            formatted_answers.append({
                "_id": qid,
                "type": "voice",
                "question": question_info.get("text", ""),
                "voice_analysis": voice_res,
                "text_evaluation": text_res,
                "score": score,
                "skills": skills
            })
            
            # Map for Voice
            for skill in skills:
                if skill not in skill_scores:
                    skill_scores[skill] = []
                    skill_question_map[skill] = []
                skill_scores[skill].append(score)
                skill_question_map[skill].append({
                    "question_id": qid,
                    "score": score,
                    "type": "voice"
                })
    
    # -------------------------------
    # COMPREHENSIVE SKILL ANALYSIS
    # -------------------------------
    skill_details = {}
    for skill, scores in skill_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Determine proficiency level
        if avg_score >= 80:
            proficiency = "Expert"
        elif avg_score >= 60:
            proficiency = "Proficient"
        elif avg_score >= 40:
            proficiency = "Developing"
        else:
            proficiency = "Needs Improvement"
        
        skill_details[skill] = {
            "average_score": round(avg_score, 2),
            "proficiency_level": proficiency,
            "questions_tested": len(scores),
            "questions": skill_question_map[skill],
            "min_score": min(scores),
            "max_score": max(scores)
        }
    
    # Sort skills by performance
    sorted_skills = sorted(skill_details.items(), key=lambda x: x[1]["average_score"], reverse=True)
    
    # Identify strengths and weaknesses
    top_skills = [skill for skill, details in sorted_skills[:3]]
    weak_skills = [skill for skill, details in sorted_skills[-3:]]
    
    # -------------------------------
    # OVERALL SESSION STATISTICS
    # -------------------------------
    metadata = evaluation_json.get("metadata", {})
    all_scores = [item["score"] for item in formatted_answers]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # Calculate grade distribution
    grade_distribution = {}
    for ans in formatted_answers:
        if ans.get("grade") and ans["grade"] != "N/A":
            grade = ans["grade"]
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    
    # -------------------------------
    # DETAILED FEEDBACK ANALYSIS (Gather from sub/coding/voice)
    # -------------------------------
    areas_needing_work = []
    strengths_found = []
    
    for ans in formatted_answers:
        feedback = ans.get("detailed_feedback", {})
        
        # Extract areas for improvement
        improvements = feedback.get("areas_for_improvement", [])
        for item in improvements:
            area = item.get("area", "")
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
        "focus_skills": weak_skills,
        "strong_skills": top_skills,
        "suggested_difficulty": "Intermediate",  # Can be dynamic based on performance
        "areas_to_improve": areas_needing_work[:5],  # Top 5 areas
        "next_steps": []
    }
    
    # Generate actionable next steps
    if overall_avg < 40:
        recommendations["suggested_difficulty"] = "Novice"
        recommendations["next_steps"].append("Focus on fundamental concepts")
        recommendations["next_steps"].append("Review basic principles before attempting complex problems")
    elif overall_avg < 70:
        recommendations["suggested_difficulty"] = "Intermediate"
        recommendations["next_steps"].append("Continue practicing intermediate-level problems")
        recommendations["next_steps"].append("Work on identified weak areas")
    else:
        recommendations["suggested_difficulty"] = "Advanced"
        recommendations["next_steps"].append("Challenge yourself with advanced topics")
        recommendations["next_steps"].append("Explore real-world applications")
    
    # Add skill-specific recommendations
    for weak_skill in weak_skills[:2]:  # Top 2 weak skills
        recommendations["next_steps"].append(f"Strengthen your {weak_skill} skills with targeted practice")
    
    # -------------------------------
    # VOICE ANALYSIS INSIGHTS (Aggregated)
    # -------------------------------
    voice_insights = None
    
    # Collect all voice analysis results from answers
    voice_results_list = [ans.get("voice_analysis") for ans in formatted_answers 
                          if ans.get("type") == "voice" and ans.get("voice_analysis")]
    
    if voice_results_list:
        # Just taking the first one for summary purposes or average them (taking first for simplicity)
        results = voice_results_list[0]
        voice_insights = {
            "confidence_score": results.get("confidence", {}).get("confidence_score_percent", 0),
            "clarity_score": results.get("clarity", {}).get("clarity_score_percent", 0),
            "speech_ratio": results.get("silence_metrics", {}).get("speech_ratio_percent", 0),
            "words_per_minute": results.get("speech_rate", {}).get("words_per_minute_actual", 0),
            "recommendations": []
        }
        
        # Voice-specific recommendations
        if voice_insights["confidence_score"] < 50:
            voice_insights["recommendations"].append("Practice speaking more confidently")
        if voice_insights["clarity_score"] < 70:
            voice_insights["recommendations"].append("Work on articulation and clarity")
        if voice_insights["speech_ratio"] < 60:
            voice_insights["recommendations"].append("Reduce pauses and maintain speech flow")
    
    # -------------------------------
    # BUILD FINAL RESULT
    # -------------------------------
    result = {
        "user_id": user_id,
        "domain": domain,
        "session_number": session_number,
        "timestamp": None,  # Add timestamp when storing
        
        "session_stats": {
            "total_questions": metadata.get("totalQuestions", len(formatted_answers)),
            "questions_answered": metadata.get("answeredCount", len(formatted_answers)),
            "mcq_attempted": mcq_count,
            "subjective_attempted": subj_count,
            "voice_attempted": voice_count,
            "coding_attempted": coding_count,
            "overall_average": round(overall_avg, 2),
            "grade_distribution": grade_distribution
        },
        
        "skill_analysis": skill_details,
        "top_skills": top_skills,
        "weak_skills": weak_skills,
        
        "recommendations": recommendations,
        "voice_insights": voice_insights,
        
        "detailed_feedback": {
            "strengths": strengths_found,
            "areas_needing_work": areas_needing_work
        }
    }
    
    # Print Summary
    print("\n=== SESSION SUMMARY ===")
    print(f"Overall Score: {result['session_stats']['overall_average']}/100")
    print(f"Questions Answered: {result['session_stats']['questions_answered']}/{result['session_stats']['total_questions']}")
    print(f"Top Skills: {result['top_skills']}")
    print(f"Weak Skills: {result['weak_skills']}")
    print(f"Focus Areas: {recommendations['focus_skills']}")
    print(f"Suggested Difficulty: {recommendations['suggested_difficulty']}\n")
    
    return result


# ============================================================
# NODE.JS BRIDGE ENTRY POINT
# ============================================================
def process_user_summary(data):
    """
    Entry point for Node.js bridge.
    data: {
        "user_id": "...",
        "domain": "...",
        "evaluation_json": {...},
        "questions_dir": "..."
    }
    """
    try:
        user_id = data.get("user_id")
        domain = data.get("domain")
        evaluation_json = data.get("evaluation_json")
        questions_dir = data.get("questions_dir")
        
        if not all([user_id, domain, evaluation_json, questions_dir]):
            return {"error": "Missing required fields"}
        
        result = build_user_summary_from_answers(user_id, domain, evaluation_json, questions_dir)
        return result
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


if __name__ == "__main__":
    # Test data (matches your new input structure)
    evaluation_json = {
        "type": "success",
        "domain": "ai_ml",
        "message": "Test submitted successfully",
        "results": [
            {
                "qid": "q_ai_0035",
                "question_type": "coding",
                "evaluation": {
                    "success": True,
                    "result": {
                        "question_id": "q_ai_0035",
                        "question_text": "Write a Python function that takes a NumPy array and normalizes it using Min-Max scaling to the range [0, 1].",
                        "score": 0,
                        "grade": "F",
                        "performance_summary": {
                            "key_concepts": "0/3 covered (0.0%)",
                            "examples": "0/0 provided (0.0%)",
                            "word_count": "6/0 words",
                            "criteria_score": "0.0%"
                        },
                        "detailed_feedback": {
                            "overall_assessment": "Your answer needs significant improvement. Review the missing content below.",
                            "strengths": ["✓ Adequate answer length (6 words)"],
                            "areas_for_improvement": [
                                {
                                    "area": "Missing Key Concepts",
                                    "count": 3,
                                    "details": [
                                        {"point": "Min-Max formula: (x - min) / (max - min)", "importance": "Required"},
                                        {"point": "Handle edge case of constant array", "importance": "Required"},
                                        {"point": "Use NumPy operations for efficiency", "importance": "Required"}
                                    ]
                                }
                            ]
                        },
                        "criteria_performance": [
                            {"criterion": "Correctness", "score": 0, "weight": 0.5},
                            {"criterion": "Code Quality", "score": 0, "weight": 0.25},
                            {"criterion": "Edge Cases", "score": 0, "weight": 0.25}
                        ]
                    }
                }
            },
            {
                "qid": "q_ai_0001",
                "question_type": "multiple-choice",
                "evaluation": {
                    "correct": True,
                    "score": 10,
                    "expected_answer": "Data with labeled output/target variables",
                    "user_answer": "Data with labeled output/target variables"
                }
            },
            {
                "qid": "q_ai_0023",
                "question_type": "multiple-choice",
                "evaluation": {
                    "correct": False,
                    "score": 0,
                    "expected_answer": "To tune hyperparameters and prevent overfitting during training.",
                    "user_answer": "To calculate the final, unbiased performance metric."
                }
            },
            {
                "qid": "q_ai_0033",
                "question_type": "voice",
                "evaluation": {
                    "voice_analysis": {
                        "success": True,
                        "result": {
                            "confidence": {"confidence_score_percent": 41.31},
                            "clarity": {"clarity_score_percent": 98.05},
                            "silence_metrics": {"speech_ratio_percent": 45.8},
                            "speech_rate": {"words_per_minute_actual": 170.31}
                        }
                    },
                    "text_evaluation": None
                }
            },
            {
                "qid": "q_ai_0024",
                "question_type": "multiple-choice",
                "evaluation": {
                    "correct": False,
                    "score": 0,
                    "expected_answer": "Series",
                    "user_answer": "Panel"
                }
            }
        ],
        "metadata": {
            "totalQuestions": 5,
            "answeredCount": 5,
            "hintsUsed": {},
            "completedAt": "2025-12-06T07:26:02.681Z",
            "timeRemaining": "44:17"
        }
    }
    
    # Use real domains directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    questions_dir = os.path.join(current_dir, "domains")
    
    # Test the function
    if os.path.exists(questions_dir):
        summary = build_user_summary_from_answers(
            user_id="user_12345",
            domain="ai_ml",
            evaluation_json=evaluation_json,
            questions_dir=questions_dir,
            session_number=1
        )
        
        print("\n📦 Complete Summary:")
        print(json.dumps(summary, indent=2))
    else:
        print(f"Error: Domains directory not found at {questions_dir}")