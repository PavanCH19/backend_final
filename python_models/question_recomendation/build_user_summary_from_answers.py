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
        JSON object to store in MongoDB with comprehensive skill analysis
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
        skills = question_info.get("skills_required", []) or question_info.get("tags", []) # Fallback to tags if skills_required missing
        
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
                "question": question_info.get("text", ""),
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
            # For now, let's use text evaluation score if available, else derive from voice confidence
            
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
    # COMPREHENSIVE SKILL ANALYSIS (Reused Logic)
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
    # -------------------------------
    # OVERALL SESSION STATISTICS
    # -------------------------------
    metadata = evaluation_json.get("metadata", {})
    all_scores = [item["score"] for item in formatted_answers]
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # -------------------------------
    # DETAILED FEEDBACK ANALYSIS (Gather from sub/coding/voice)
    # -------------------------------
    areas_needing_work = []
    strengths_found = []
    
    for ans in formatted_answers:
        feedback = ans.get("detailed_feedback", {})
        
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
    # -------------------------------
    # VOICE ANALYSIS INSIGHTS (Aggregated)
    # -------------------------------
    voice_insights = None
    
    # Collect all voice analysis results from answers
    voice_results_list = [ans.get("voice_analysis") for ans in formatted_answers if ans.get("type") == "voice" and ans.get("voice_analysis")]
    
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
            "total_questions": metadata.get("totalQuestions", len(formatted_answers)),
            "questions_answered": metadata.get("answeredCount", len(formatted_answers)),
            "mcq_attempted": mcq_count,
            "subjective_attempted": subj_count,
            "voice_attempted": voice_count,
            "coding_attempted": coding_count,
            "overall_average": round(overall_avg, 2),
            "grade_distribution": {} # Can be calculated if needed
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
        return {"error": str(e), "traceback": str(e)}

if __name__ == "__main__":
    # Test data (Optional: for local debugging)
    data = {
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
                        "strengths": [
                            "✓ Adequate answer length (6 words)"
                        ],
                        "areas_for_improvement": [
                            {
                                "area": "Missing Key Concepts",
                                "count": 3,
                                "details": [
                                    {
                                        "point": "Min-Max formula: (x - min) / (max - min)",
                                        "importance": "Required",
                                        "suggestion": "Include this key concept in your answer"
                                    },
                                    {
                                        "point": "Handle edge case of constant array",
                                        "importance": "Required",
                                        "suggestion": "Include this key concept in your answer"
                                    },
                                    {
                                        "point": "Use NumPy operations for efficiency",
                                        "importance": "Required",
                                        "suggestion": "Include this key concept in your answer"
                                    }
                                ]
                            },
                            {
                                "area": "Evaluation Criteria",
                                "count": 3,
                                "details": [
                                    {
                                        "criterion": "Correctness",
                                        "feedback": "Overall performance on this criterion",
                                        "current_score": 0
                                    },
                                    {
                                        "criterion": "Code Quality",
                                        "feedback": "Overall performance on this criterion",
                                        "current_score": 0
                                    },
                                    {
                                        "criterion": "Edge Cases",
                                        "feedback": "Overall performance on this criterion",
                                        "current_score": 0
                                    }
                                ]
                            }
                        ],
                        "what_was_covered": [],
                        "what_to_add": {
                            "missing_concepts": [
                                {
                                    "point": "Min-Max formula: (x - min) / (max - min)",
                                    "importance": "Required",
                                    "suggestion": "Include this key concept in your answer"
                                },
                                {
                                    "point": "Handle edge case of constant array",
                                    "importance": "Required",
                                    "suggestion": "Include this key concept in your answer"
                                },
                                {
                                    "point": "Use NumPy operations for efficiency",
                                    "importance": "Required",
                                    "suggestion": "Include this key concept in your answer"
                                }
                            ],
                            "partially_covered": [],
                            "missing_examples": []
                        }
                    },
                    "criteria_performance": [
                        {
                            "criterion": "Correctness",
                            "score": 0,
                            "weight": 0.5,
                            "weighted_score": 0,
                            "feedback": "Overall performance on this criterion"
                        },
                        {
                            "criterion": "Code Quality",
                            "score": 0,
                            "weight": 0.25,
                            "weighted_score": 0,
                            "feedback": "Overall performance on this criterion"
                        },
                        {
                            "criterion": "Edge Cases",
                            "score": 0,
                            "weight": 0.25,
                            "weighted_score": 0,
                            "feedback": "Overall performance on this criterion"
                        }
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
                        "silence_metrics": {
                            "silence_ratio_percent": 54.2,
                            "speech_ratio_percent": 45.79999999999999,
                            "silence_time_seconds": 5.420000000000001,
                            "normal_speech_time_seconds": 3.54,
                            "low_volume_speech_time_seconds": 1.0399999999999991,
                            "total_speech_time_seconds": 4.579999999999999,
                            "total_duration_seconds": 10,
                            "silence_segment_count": 7,
                            "normal_speech_segment_count": 26,
                            "low_volume_segment_count": 13
                        },
                        "speech_rate": {
                            "estimated_syllables": 50,
                            "estimated_words": 29.411764705882355,
                            "words_per_minute_estimated": 385.30696121243267,
                            "words_per_second_estimated": 6.421782686873878,
                            "speaking_time_seconds": 4.579999999999999,
                            "actual_word_count": 13,
                            "words_per_minute_actual": 170.30567685589523,
                            "words_per_second_actual": 2.838427947598254
                        },
                        "pitch_variation": {
                            "mean_pitch_hz": 173.28402217287874,
                            "median_pitch_hz": 164.81377845643496,
                            "pitch_std_hz": 27.990579446146867,
                            "pitch_range_hz": 152.31156558491202,
                            "min_pitch_hz": 134.6458897635121,
                            "max_pitch_hz": 286.9574553484241,
                            "pitch_variation_percent": 16.15300654680196,
                            "voiced_frame_count": 168
                        },
                        "clarity": {
                            "spectral_flatness": 0.019460702314972878,
                            "zero_crossing_rate": 0.10050169728434505,
                            "spectral_centroid_hz": 1394.7114826943389,
                            "clarity_score_percent": 98.0539321899414
                        },
                        "pacing": {
                            "avg_speech_segment_duration_seconds": 0.11743589743589741,
                            "speech_duration_std_seconds": 0.10833750560923255,
                            "pacing_consistency_percent": 7.747538891701524,
                            "total_speech_segments": 39,
                            "min_segment_duration_seconds": 0.019999999999999574,
                            "max_segment_duration_seconds": 0.47999999999999954
                        },
                        "expressiveness": {
                            "energy_variation_rms": 0.006538851652294397,
                            "mean_energy_rms": 0.006436733528971672,
                            "dynamic_range_db": 58.58256912231445,
                            "energy_variation_percent": 101.58648681640625,
                            "expressiveness_score_percent": 58.07650327340098
                        },
                        "confidence": {
                            "confidence_score_percent": 41.306053161621094,
                            "silence_contribution": 45.79999999999999,
                            "pitch_stability_contribution": 83.84699345319804,
                            "energy_contribution": 26.955514907836914,
                            "pacing_contribution": 20
                        },
                        "transcription": {
                            "transcript": "hello welcome to my YouTube channel this is a Whisper audio thank you",
                            "word_count": 13,
                            "confidence": "high",
                            "method": "Google Speech Recognition"
                        }
                    }
                },
                "text_evaluation": null
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
    process_user_summary(data)