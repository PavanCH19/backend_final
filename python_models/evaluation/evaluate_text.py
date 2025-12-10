
"""
Production-ready completeness scoring with detailed feedback
Requirements: pip install sentence-transformers nltk scikit-learn
"""
from typing import List, Dict, Tuple
import re
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# -------------------------
# Setup (downloads + model)
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)

lemmatizer = WordNetLemmatizer()
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# -------------------------
# Helpers: preprocessing
# -------------------------
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

def normalize_text(text: str) -> str:
    """Lowercase, remove extra whitespace and normalize quotes/apostrophes."""
    t = text.lower()
    t = t.replace("'", "'").replace(""", '"').replace(""", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t

def lemmatize_text(text: str) -> str:
    """Simple lemmatization (preserve punctuation removal for embeddings)."""
    text = normalize_text(text)
    tokens = word_tokenize(text)
    lem = [lemmatizer.lemmatize(t) for t in tokens]
    lem = [re.sub(r"[^\w\s]", "", token) for token in lem]
    lem = [token for token in lem if token]
    return " ".join(lem)

# -------------------------
# Embedding helpers
# -------------------------
def embed_texts(texts: List[str]):
    if not texts:
        return None
    return embedder.encode(texts, convert_to_tensor=True, show_progress_bar=False)

def compute_similarity(text1: str, text2: str) -> float:
    """Compute semantic similarity between two texts."""
    if not text1 or not text2:
        return 0.0
    emb1 = embed_texts([lemmatize_text(text1)])
    emb2 = embed_texts([lemmatize_text(text2)])
    sim = util.cos_sim(emb1, emb2).cpu().numpy()
    return float(sim[0][0])

# -------------------------
# Key Point Evaluation
# -------------------------
def evaluate_key_points(key_points: List[str], user_answer: str, 
                       threshold: float = 0.60) -> Dict:
    """Evaluate each key point against user answer with detailed tracking."""
    if not key_points:
        return {
            "coverage": 0.0,
            "covered_count": 0,
            "total_count": 0,
            "covered_points": [],
            "missing_points": [],
            "partially_covered": []
        }
    
    user_sentences = sent_tokenize(user_answer)
    covered_points = []
    missing_points = []
    partially_covered = []
    
    for point in key_points:
        best_sim = 0.0
        best_match = ""
        
        for sent in user_sentences:
            sim = compute_similarity(point, sent)
            if sim > best_sim:
                best_sim = sim
                best_match = sent.strip()
        
        if best_sim >= threshold:
            covered_points.append({
                "point": point,
                "match": best_match,
                "similarity": round(best_sim, 2)
            })
        elif best_sim >= 0.40:  # Partially covered
            partially_covered.append({
                "point": point,
                "partial_match": best_match,
                "similarity": round(best_sim, 2),
                "suggestion": "Expand on this concept for better coverage"
            })
        else:
            missing_points.append({
                "point": point,
                "importance": "Required",
                "suggestion": "Include this key concept in your answer"
            })
    
    coverage_pct = (len(covered_points) / len(key_points)) * 100 if key_points else 0.0
    
    return {
        "coverage": round(coverage_pct, 1),
        "covered_count": len(covered_points),
        "total_count": len(key_points),
        "covered_points": covered_points,
        "missing_points": missing_points,
        "partially_covered": partially_covered
    }

# -------------------------
# Example Scenarios Evaluation
# -------------------------
def evaluate_examples(examples: List[str], user_answer: str, 
                     threshold: float = 0.55) -> Dict:
    """Evaluate if user provided required examples/scenarios."""
    if not examples:
        return {
            "coverage": 0.0,
            "provided_count": 0,
            "total_count": 0,
            "provided_examples": [],
            "missing_examples": []
        }
    
    provided_examples = []
    missing_examples = []
    
    for example in examples:
        sim = compute_similarity(example, user_answer)
        
        if sim >= threshold:
            provided_examples.append({
                "example": example,
                "status": "Provided",
                "similarity": round(sim, 2)
            })
        else:
            missing_examples.append({
                "example": example,
                "importance": "Required",
                "suggestion": "Add this real-world scenario/example"
            })
    
    coverage_pct = (len(provided_examples) / len(examples)) * 100 if examples else 0.0
    
    return {
        "coverage": round(coverage_pct, 1),
        "provided_count": len(provided_examples),
        "total_count": len(examples),
        "provided_examples": provided_examples,
        "missing_examples": missing_examples
    }

# -------------------------
# Evaluation Criteria Assessment
# -------------------------
def evaluate_criteria(criteria: List[Dict], user_answer: str, 
                     key_points_result: Dict, examples_result: Dict) -> Dict:
    """Assess user answer against evaluation criteria with detailed feedback."""
    if not criteria:
        return {
            "overall_score": 0.0,
            "criteria_scores": []
        }
    
    total_weighted_score = 0.0
    criteria_scores = []
    
    for criterion in criteria:
        criterion_name = criterion.get("criterion", "Unknown")
        weight = criterion.get("weight", 0.0)
        description = criterion.get("description", "")
        
        score = 0.0
        feedback = ""
        name_lower = criterion_name.lower()
        
        if any(word in name_lower for word in ["understanding", "conceptual", "concept"]):
            score = key_points_result.get("coverage", 0.0)
            if score >= 80:
                feedback = "Strong conceptual understanding demonstrated"
            elif score >= 60:
                feedback = "Good understanding, but some concepts need expansion"
            else:
                feedback = "More depth needed on key concepts"
        
        elif any(word in name_lower for word in ["application", "example", "scenario", "practical"]):
            score = examples_result.get("coverage", 0.0)
            if score >= 80:
                feedback = "Excellent real-world examples provided"
            elif score >= 50:
                feedback = "Add more diverse practical examples"
            else:
                feedback = "Include specific real-world scenarios"
        
        elif any(word in name_lower for word in ["depth", "technical", "detail"]):
            word_count = len(user_answer.split())
            score = min(100, (word_count / 200) * 100)
            if score >= 80:
                feedback = "Sufficient technical depth"
            else:
                feedback = "Provide more detailed technical explanation"
        
        elif any(word in name_lower for word in ["clarity", "communication", "clear"]):
            sentences = sent_tokenize(user_answer)
            if sentences:
                avg_sent_len = sum(len(s.split()) for s in sentences) / len(sentences)
                score = 100 if 10 <= avg_sent_len <= 30 else 70
                feedback = "Clear and well-structured" if score >= 80 else "Improve sentence structure and clarity"
            else:
                score = 0
                feedback = "Answer needs better structure"
        
        elif any(word in name_lower for word in ["mathematical", "formula", "calculation"]):
            has_formulas = bool(re.search(r'[+\-*/=()]|TP|FP|FN|TN', user_answer))
            score = 100 if has_formulas else 50
            feedback = "Mathematical concepts explained" if score >= 80 else "Include relevant formulas and calculations"
        
        elif any(word in name_lower for word in ["comparative", "comparison", "contrast"]):
            comparison_words = ['however', 'while', 'whereas', 'unlike', 'different', 'compare', 'contrast']
            has_comparison = any(word in user_answer.lower() for word in comparison_words)
            score = 100 if has_comparison else 60
            feedback = "Good comparative analysis" if score >= 80 else "Add more comparison between concepts"
        
        else:
            score = (key_points_result.get("coverage", 0.0) + 
                    examples_result.get("coverage", 0.0)) / 2
            feedback = "Overall performance on this criterion"
        
        weighted_contribution = (score / 100) * weight
        total_weighted_score += weighted_contribution
        
        criteria_scores.append({
            "criterion": criterion_name,
            "score": round(score, 1),
            "weight": weight,
            "weighted_score": round(weighted_contribution * 100, 1),
            "feedback": feedback
        })
    
    overall_score = total_weighted_score * 100
    
    return {
        "overall_score": round(overall_score, 1),
        "criteria_scores": criteria_scores
    }

# -------------------------
# Answer Quality Metrics
# -------------------------
def calculate_answer_metrics(user_answer: str, min_word_count: int = 0) -> Dict:
    """Calculate various metrics about the answer quality."""
    words = user_answer.split()
    sentences = sent_tokenize(user_answer)
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    meets_word_count = word_count >= min_word_count if min_word_count > 0 else True
    
    metrics = {
        "word_count": word_count,
        "required_words": min_word_count,
        "meets_requirement": meets_word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 1)
    }
    
    if not meets_word_count and min_word_count > 0:
        metrics["length_feedback"] = f"Expand your answer by {min_word_count - word_count} words"
    else:
        metrics["length_feedback"] = "Word count requirement met"
    
    return metrics

# -------------------------
# Generate Comprehensive Feedback
# -------------------------
def generate_comprehensive_feedback(key_points_result: Dict, examples_result: Dict, 
                                   criteria_result: Dict, answer_metrics: Dict, 
                                   final_score: float) -> Dict:
    """Generate detailed, production-ready feedback."""
    
    # Overall assessment
    if final_score >= 90:
        overall = "Excellent work! Your answer demonstrates comprehensive understanding."
    elif final_score >= 80:
        overall = "Very good answer with strong understanding. Minor improvements will make it excellent."
    elif final_score >= 70:
        overall = "Good answer showing solid understanding. Focus on the areas below to improve."
    elif final_score >= 60:
        overall = "Satisfactory answer, but needs more depth and coverage of key concepts."
    else:
        overall = "Your answer needs significant improvement. Review the missing content below."
    
    # Strengths
    strengths = []
    if key_points_result["coverage"] >= 80:
        strengths.append(f"✓ Strong coverage of key concepts ({key_points_result['covered_count']}/{key_points_result['total_count']})")
    if examples_result["coverage"] >= 80:
        strengths.append(f"✓ Excellent use of examples ({examples_result['provided_count']}/{examples_result['total_count']})")
    if answer_metrics["meets_requirement"]:
        strengths.append(f"✓ Adequate answer length ({answer_metrics['word_count']} words)")
    
    # Areas for improvement
    improvements = []
    if key_points_result["missing_points"]:
        improvements.append({
            "area": "Missing Key Concepts",
            "count": len(key_points_result["missing_points"]),
            "details": key_points_result["missing_points"]
        })
    
    if key_points_result["partially_covered"]:
        improvements.append({
            "area": "Partially Covered Concepts",
            "count": len(key_points_result["partially_covered"]),
            "details": key_points_result["partially_covered"]
        })
    
    if examples_result["missing_examples"]:
        improvements.append({
            "area": "Missing Examples/Scenarios",
            "count": len(examples_result["missing_examples"]),
            "details": examples_result["missing_examples"]
        })
    
    if not answer_metrics["meets_requirement"]:
        improvements.append({
            "area": "Answer Length",
            "count": 1,
            "details": [{
                "issue": answer_metrics["length_feedback"],
                "suggestion": f"Current: {answer_metrics['word_count']} words, Required: {answer_metrics['required_words']} words"
            }]
        })
    
    # Criteria-specific feedback
    weak_criteria = [c for c in criteria_result["criteria_scores"] if c["score"] < 70]
    if weak_criteria:
        improvements.append({
            "area": "Evaluation Criteria",
            "count": len(weak_criteria),
            "details": [{
                "criterion": c["criterion"],
                "feedback": c["feedback"],
                "current_score": c["score"]
            } for c in weak_criteria]
        })
    
    return {
        "overall_assessment": overall,
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "what_was_covered": key_points_result["covered_points"],
        "what_to_add": {
            "missing_concepts": key_points_result["missing_points"],
            "partially_covered": key_points_result["partially_covered"],
            "missing_examples": examples_result["missing_examples"]
        }
    }

# -------------------------
# Main JSON Evaluation Function
# -------------------------
def evaluate_question_from_json(question_data: Dict, user_answer: str, 
                                params: Dict = None) -> Dict:
    """
    Production-ready evaluation with comprehensive feedback.
    """
    if params is None:
        params = {}
    
    expected = question_data.get("expected_answer", {})
    criteria = question_data.get("evaluation_criteria", [])
    
    # Extract components
    key_points = expected.get("key_points", []) or expected.get("key_concepts", [])
    examples = expected.get("example_scenarios", []) or expected.get("scenario_examples", [])
    min_word_count = expected.get("minimum_word_count", 0)
    
    # Evaluate components
    key_points_result = evaluate_key_points(key_points, user_answer, 
                                           threshold=params.get("key_point_threshold", 0.60))
    
    examples_result = evaluate_examples(examples, user_answer, 
                                       threshold=params.get("example_threshold", 0.55))
    
    criteria_result = evaluate_criteria(criteria, user_answer, 
                                       key_points_result, examples_result)
    
    answer_metrics = calculate_answer_metrics(user_answer, min_word_count)
    
    # Calculate final score (weighted average)
    kp_weight = 0.4
    ex_weight = 0.3
    criteria_weight = 0.3
    
    final_score = (
        (key_points_result["coverage"] * kp_weight) +
        (examples_result["coverage"] * ex_weight) +
        (criteria_result["overall_score"] * criteria_weight)
    )
    
    # Determine grade
    if final_score >= 90:
        grade = "A"
    elif final_score >= 80:
        grade = "B"
    elif final_score >= 70:
        grade = "C"
    elif final_score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    feedback = generate_comprehensive_feedback(key_points_result, examples_result, 
                                              criteria_result, answer_metrics, final_score)
    
    return {
        "question_id": question_data.get("_id", "unknown"),
        "question_text": question_data.get("text", ""),
        "score": round(final_score, 1),
        "grade": grade,
        "performance_summary": {
            "key_concepts": f"{key_points_result['covered_count']}/{key_points_result['total_count']} covered ({key_points_result['coverage']}%)",
            "examples": f"{examples_result['provided_count']}/{examples_result['total_count']} provided ({examples_result['coverage']}%)",
            "word_count": f"{answer_metrics['word_count']}/{answer_metrics['required_words']} words",
            "criteria_score": f"{criteria_result['overall_score']}%"
        },
        "detailed_feedback": feedback,
        "criteria_performance": criteria_result["criteria_scores"]
    }

# -------------------------
# Batch Evaluation
# -------------------------
def evaluate_batch_from_json(questions_json: List[Dict], user_answers: Dict[str, str], 
                             params: Dict = None) -> Dict:
    """Evaluate multiple questions with production-ready output."""
    results = []
    
    for question in questions_json:
        question_id = question.get("_id", "")
        user_answer = user_answers.get(question_id, "")
        
        if not user_answer:
            results.append({
                "question_id": question_id,
                "question_text": question.get("text", ""),
                "score": 0.0,
                "grade": "F",
                "error": "No answer provided",
                "detailed_feedback": {
                    "overall_assessment": "Answer not submitted",
                    "strengths": [],
                    "areas_for_improvement": []
                }
            })
            continue
        
        result = evaluate_question_from_json(question, user_answer, params)
        results.append(result)
    
    # Calculate statistics
    valid_results = [r for r in results if "error" not in r]
    
    if valid_results:
        scores = [r["score"] for r in valid_results]
        avg_score = sum(scores) / len(scores)
        
        grade_distribution = {}
        for r in valid_results:
            grade = r["grade"]
            grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
    else:
        avg_score = 0.0
        grade_distribution = {}
    
    return {
        "summary": {
            "total_questions": len(questions_json),
            "answered": len(valid_results),
            "average_score": round(avg_score, 1),
            "grade_distribution": grade_distribution
        },
        "detailed_results": results
    }

# -------------------------
# Test Function
# -------------------------
def run_comprehensive_test():
    """Test with multiple questions and answers."""
    
    questions = [
        {
            "_id": "q_ai_0031",
            "question_type": "subjective",
            "domain": "Artificial Intelligence & Machine Learning",
            "topic": "Deep Learning",
            "difficulty": 3,
            "difficulty_label": "Intermediate",
            "text": "Explain the concept of transfer learning in deep learning. Describe how it works and provide two real-world scenarios where it would be beneficial.",
            "expected_answer": {
                "key_points": [
                    "Transfer learning involves using a pre-trained model on a new, related task",
                    "The model has learned features from a large dataset that can be reused",
                    "Common approach: freeze early layers, fine-tune later layers",
                    "Benefits include reduced training time and improved performance with limited data"
                ],
                "example_scenarios": [
                    "Using ImageNet pre-trained models for medical image classification",
                    "Using BERT for domain-specific text classification tasks"
                ],
                "minimum_word_count": 150
            },
            "evaluation_criteria": [
                {
                    "criterion": "Conceptual Understanding",
                    "weight": 0.4,
                    "description": "Demonstrates clear understanding of transfer learning principles"
                },
                {
                    "criterion": "Real-world Application",
                    "weight": 0.3,
                    "description": "Provides relevant and practical examples"
                },
                {
                    "criterion": "Technical Depth",
                    "weight": 0.3,
                    "description": "Explains implementation details and benefits"
                }
            ]
        },
        {
            "_id": "q_ai_0033",
            "question_type": "subjective",
            "domain": "Artificial Intelligence & Machine Learning",
            "topic": "Fundamentals",
            "difficulty": 2,
            "difficulty_label": "Novice",
            "text": "Explain in your own words what overfitting is and how you would detect it in a machine learning model.",
            "expected_answer": {
                "key_points": [
                    "Model performs well on training data but poorly on unseen data",
                    "Model memorizes rather than generalizes patterns",
                    "Detection: gap between training and validation performance",
                    "Solutions: regularization, more data, simpler model"
                ],
                "minimum_word_count": 100
            },
            "evaluation_criteria": [
                {
                    "criterion": "Clarity",
                    "weight": 0.3,
                    "description": "Clear and understandable explanation"
                },
                {
                    "criterion": "Completeness",
                    "weight": 0.4,
                    "description": "Covers main concepts of overfitting"
                },
                {
                    "criterion": "Examples",
                    "weight": 0.3,
                    "description": "Provides relevant examples or analogies"
                }
            ]
        }
    ]
    
    user_answers = {
        "q_ai_0031": """Transfer learning is a technique where we use a model that was 
        trained on one task and apply it to a different but related task. The pre-trained 
        model has already learned useful features from a large dataset like ImageNet. 
        We typically freeze the early layers and fine-tune the later layers for our specific task. 
        This approach saves training time and works well when we have limited data. 
        For example, we can use ImageNet pre-trained models for medical imaging tasks like 
        tumor detection.""",
        
        "q_ai_0033": """Overfitting happens when a model performs really well on training data 
        but poorly on new unseen data. The model basically memorizes the training examples instead 
        of learning general patterns."""
    }
    
    print("\n" + "="*80)
    print("PRODUCTION-READY EVALUATION RESULTS")
    print("="*80)
    
    results = evaluate_batch_from_json(questions, user_answers)
    
    # Print as formatted JSON
    print(json.dumps(results, indent=2))
    
    # Also save to file
    with open("evaluation_results_production.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("Results saved to: evaluation_results_production.json")
    print("="*80)


# ============================================================
# NODE.JS BRIDGE ENTRY POINT
# ============================================================
def evaluate_text(data):
    """
    Entry point for Node.js bridge.
    Expected data format:
    {
        "question": { ... question dict from JSON ... },
        "user_answer": "string answer"
    }
    """
    try:
        question = data.get("question")
        user_answer = data.get("user_answer")

        if not question or not user_answer:
            return {"error": "Missing question or user_answer"}

        # Use the existing single-question evaluation logic
        result = evaluate_question_from_json(question, user_answer)
        return result

    except Exception as e:
        return {"error": str(e), "traceback": str(e)}

if __name__ == "__main__":
    run_comprehensive_test()