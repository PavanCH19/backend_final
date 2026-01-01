import json
import math
from statistics import mean
import random
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

# ======================================================
# MODEL CONFIG (CORRECTED - Using proper similarity model)
# ======================================================
SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Only for reranking/feedback

GRADE_SCALE = [
    (90, "A+"),
    (80, "A"),
    (70, "B+"),
    (60, "B"),
    (40, "C"),
    (35, "F")
]

print("Loading models...")
# Primary model for semantic similarity (cosine similarity)
embedder = SentenceTransformer(SIMILARITY_MODEL)
# CrossEncoder only for reranking/feedback (optional)
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
print("Models loaded.")

# ======================================================
# UTILITIES
# ======================================================
def grade(score):
    for t, g in GRADE_SCALE:
        if score >= t:
            return g
    return "F"

def word_count(text):
    return len(text.split())

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def semantic_similarity(a, b):
    """
    Proper semantic similarity using cosine similarity of embeddings
    Returns similarity ∈ [0, 1] (cosine similarity of normalized embeddings)
    """
    # Encode both texts with normalized embeddings
    emb_a = embedder.encode(a, normalize_embeddings=True, convert_to_numpy=True)
    emb_b = embedder.encode(b, normalize_embeddings=True, convert_to_numpy=True)
    
    # Compute cosine similarity (for normalized embeddings, range is [-1, 1])
    # For semantic text, values are typically in [0, 1] range
    similarity = float(util.cos_sim(emb_a, emb_b)[0][0])
    
    # Clamp to [0, 1] - negative similarities are rare for semantic text
    # If negative, it means texts are semantically opposite (very rare)
    return max(0.0, min(1.0, similarity))

# ======================================================
# SEMANTIC FEEDBACK (FAST, NON-RULE)
# ======================================================
def semantic_feedback(answer, expected_points, top_k=2):
    """
    Use proper semantic similarity for feedback ranking
    """
    if not expected_points:
        return {"strengths": [], "improvements": []}

    # Use semantic similarity (cosine) instead of CrossEncoder for ranking
    scores = [semantic_similarity(answer, p) for p in expected_points]

    ranked = sorted(
        zip(expected_points, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "strengths": [x[0] for x in ranked[:top_k]],
        "improvements": [x[0] for x in ranked[-top_k:]]
    }

# ======================================================
# CORE EVALUATION
# ======================================================
def evaluate_answer(question, answer):
    expected = question["expected_answer"]
    criteria = question["evaluation_criteria"]

    expected_points = (
        expected.get("key_points", []) +
        expected.get("example_scenarios", [])
    )

    ideal_text = " ".join(expected_points)

    # -------- overall semantic alignment --------
    base_similarity = semantic_similarity(answer, ideal_text)

    # -------- length influence (moderately lenient) --------
    wc = word_count(answer)
    min_words = expected.get("minimum_word_count", 0)
    if min_words:
        # Moderately lenient: penalize short answers but not too harsh
        ratio = wc / min_words
        if ratio >= 1.0:
            # Small bonus for exceeding minimum (capped at 5%)
            length_factor = min(1.05, 1.0 + (ratio - 1.0) * 0.10)
        elif ratio >= 0.7:
            # Moderate penalty if 70-100% of minimum (0.92 to 1.0)
            length_factor = 0.92 + (ratio - 0.7) * 0.27  # 0.92 to 1.0
        else:
            # Penalty for shorter answers, but not too harsh
            length_factor = 0.75 + (ratio / 0.7) * 0.17  # 0.75 to 0.92
    else:
        length_factor = 1.0

    # -------- criteria evaluation --------
    criteria_results = []
    weighted_sum = 0.0

    for c in criteria:
        criterion_context = (
            question["text"] + " " +
            c["criterion"] + " " +
            c.get("description", "")
        )

        sim = semantic_similarity(answer, criterion_context)
        # With proper similarity model, scores are naturally in good range
        # Scale to 0-100 range (similarity is already normalized to [0, 1])
        score = sim * 100
        # Cap at 100 to avoid exceeding
        score = min(100.0, score)
        weighted = score * c["weight"]

        weighted_sum += weighted

        criteria_results.append({
            "criterion": c["criterion"],
            "score": round(score, 1),
            "weight": c["weight"],
            "weighted_score": round(weighted, 1)
        })

    # -------- coverage-based adjustment (ensures actual content coverage matters) --------
    # Check how many key points and examples are actually covered
    key_points = expected.get("key_points", [])
    example_scenarios = expected.get("example_scenarios", [])
    
    # Domain-aware thresholds: lower for easier/definition questions
    # Difficulty <= 2 (Novice/Beginner): use 0.35 threshold
    # Difficulty > 2 (Intermediate/Advanced): use 0.45 threshold
    difficulty = question.get("difficulty", 3)
    KEYPOINT_THRESHOLD = 0.35 if difficulty <= 2 else 0.45
    EXAMPLE_THRESHOLD = 0.35 if difficulty <= 2 else 0.45
    
    # Calculate coverage for key points
    key_points_covered = 0
    key_point_similarities = []
    for point in key_points:
        sim = semantic_similarity(answer, point)
        key_point_similarities.append(sim)
        if sim >= KEYPOINT_THRESHOLD:
            key_points_covered += 1
    
    # Calculate coverage for examples
    examples_covered = 0
    example_similarities = []
    for example in example_scenarios:
        sim = semantic_similarity(answer, example)
        example_similarities.append(sim)
        if sim >= EXAMPLE_THRESHOLD:
            examples_covered += 1
    
    # Calculate coverage ratios
    key_coverage_ratio = key_points_covered / len(key_points) if key_points else 1.0
    example_coverage_ratio = examples_covered / len(example_scenarios) if example_scenarios else 1.0
    
    # Combined coverage factor (weighted: key points 70%, examples 30%)
    # Single clamp: coverage refines score, doesn't crush it
    if key_points or example_scenarios:
        raw_coverage = (key_coverage_ratio * 0.7 + example_coverage_ratio * 0.3)
        # Coverage factor: 0.0 coverage = 0.7 factor, 1.0 coverage = 1.0 factor
        # This caps/refines the score rather than double-penalizing
        coverage_factor = 0.7 + (raw_coverage * 0.3)  # Range: 0.7 to 1.0
    else:
        coverage_factor = 1.0
    
    # Calculate base score with length factor
    base_score = weighted_sum * length_factor
    
    # Apply coverage adjustment (single clamp, not double penalty)
    base_score = base_score * coverage_factor
    
    # Add boost for high-quality answers (only if coverage is good)
    # This helps excellent answers reach 85+ or 90+, but only with good coverage
    if coverage_factor >= 0.75 and weighted_sum >= 60:
        # Boost: starts at 60, up to 10% boost at 100
        boost = (weighted_sum - 60) / 40 * 0.10  # 0% at 60, 10% at 100
        base_score = base_score * (1.0 + boost)
    elif coverage_factor >= 0.6 and weighted_sum >= 50:
        # Small boost for moderate answers with decent coverage
        boost = (weighted_sum - 50) / 10 * 0.03  # 0% at 50, 3% at 60
        base_score = base_score * (1.0 + boost)
    
    # Human-alignment override: credit implicit technical competence
    # If criteria score is high and overall similarity is good, ensure minimum score
    # This makes excellent answers score like a human examiner would
    if weighted_sum >= 70 and base_similarity >= 0.65:
        # High-confidence override: excellent answers get minimum B+/A
        final_score = max(base_score, 80.0)
    else:
        final_score = base_score
    
    final_score = round(final_score, 1)
    final_score = max(0.0, min(100.0, final_score))

    feedback = semantic_feedback(answer, expected_points)

    return {
        "question_id": question["_id"],
        "question_text": question["text"],
        "score": final_score,
        "grade": grade(final_score),
        "word_count": wc,
        "criteria_performance": criteria_results,
        "key_concept_coverage": key_coverage_ratio,
        "example_coverage": example_coverage_ratio,
        "feedback": feedback
    }

# ============================================================
# SINGLE QUESTION EVALUATION (NODE SAFE)
# ============================================================
def evaluate_question_from_json(question, user_answer):
    """
    Evaluates ONE question and ONE answer.
    This is the function Node.js should use internally.
    """
    if not isinstance(question, dict):
        return {"error": "Invalid question format"}

    if not isinstance(user_answer, str) or not user_answer.strip():
        return {"error": "Invalid user_answer"}

    return evaluate_answer(question, user_answer)


# ======================================================
# BATCH PIPELINE (FIXED MATCHING)
# ======================================================
def evaluate_batch(questions, user_answers):

    results = []
    scores = []

    q_map = {q["_id"]: q for q in questions}

    for qid, answer in user_answers.items():
        if qid not in q_map:
            continue

        result = evaluate_answer(q_map[qid], answer)
        result["user_id"] = qid

        results.append(result)
        scores.append(result["score"])

    grade_distribution = {}
    for r in results:
        grade_distribution[r["grade"]] = grade_distribution.get(r["grade"], 0) + 1

    return {
        "summary": {
            "average_score": round(mean(scores), 1),
            "grade_distribution": grade_distribution
        },
        "detailed_results": results
    }

# ======================================================
# NODE-SAFE BATCH ENTRY POINT (REQUIRED)
# ======================================================
# def evaluate_batch_from_json(data):
#     """
#     Node.js-safe batch evaluation entry point.

#     Expected input:
#     {
#         "questions": [...],
#         "user_answers": { "qid": "answer", ... }
#     }
#     """
#     try:
#         if not isinstance(data, dict):
#             return {"success": False, "error": "Input must be a JSON object"}

#         questions = data.get("questions")
#         user_answers = data.get("user_answers")

#         if not questions or not isinstance(questions, list):
#             return {"success": False, "error": "Invalid or missing 'questions'"}

#         if not user_answers or not isinstance(user_answers, dict):
#             return {"success": False, "error": "Invalid or missing 'user_answers'"}

#         result = evaluate_batch(questions, user_answers)

#         return {
#             "success": True,
#             "result": result
#         }

#     except Exception as e:
#         return {
#             "success": False,
#             "error": str(e)
#         }
def evaluate_batch_from_json(*args):
    """
    Node.js-safe AND bridge-safe batch evaluation entry point.

    Supports BOTH:
    1) evaluate_batch_from_json(data_dict)
    2) evaluate_batch_from_json(questions, user_answers)
    """

    try:
        # --------------------------------------------------
        # CASE 1: Bridge passed (questions, user_answers)
        # --------------------------------------------------
        if len(args) == 2:
            questions, user_answers = args

        # --------------------------------------------------
        # CASE 2: Bridge passed single JSON object
        # --------------------------------------------------
        elif len(args) == 1 and isinstance(args[0], dict):
            data = args[0]
            questions = data.get("questions")
            user_answers = data.get("user_answers")

        else:
            return {
                "success": False,
                "error": "Invalid arguments passed to evaluate_batch_from_json"
            }

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------
        if not isinstance(questions, list):
            return {"success": False, "error": "Invalid or missing 'questions'"}

        if not isinstance(user_answers, dict):
            return {"success": False, "error": "Invalid or missing 'user_answers'"}

        # --------------------------------------------------
        # Actual evaluation
        # --------------------------------------------------
        result = evaluate_batch(questions, user_answers)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


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


def test_multiple_answers(question_id, questions, test_answers):
    q_map = {q["_id"]: q for q in questions}
    q = q_map[question_id]

    results = []

    for label, answer in test_answers.items():
        r = evaluate_answer(q, answer)
        r["test_case"] = label
        results.append(r)

    return results

def analyze_key_coverage(answer, expected_points, question_difficulty=3, threshold=None):
    """
    Analyze which expected points are covered in the answer using proper semantic similarity
    Uses domain-aware thresholds if threshold not provided
    """
    # Use domain-aware threshold if not provided
    if threshold is None:
        threshold = 0.35 if question_difficulty <= 2 else 0.45
    
    covered = []
    missing = []
    
    for point in expected_points:
        sim = semantic_similarity(answer, point)
        if sim >= threshold:
            covered.append((point, round(sim, 2)))
        else:
            missing.append((point, round(sim, 2)))
    
    return covered, missing

def format_production_output(raw_results, questions, user_answers_dict):
    detailed = []
    q_map = {q["_id"]: q for q in questions}

    for r in raw_results["detailed_results"]:
        q = q_map[r["question_id"]]
        expected = q["expected_answer"]
        
        # Get the answer text - use user_id if available, otherwise question_id
        answer_key = r.get("user_id", r["question_id"])
        answer_text = user_answers_dict.get(answer_key, "")

        expected_points = (
            expected.get("key_points", []) +
            expected.get("example_scenarios", [])
        )

        # Get question difficulty for domain-aware threshold
        question_difficulty = q.get("difficulty", 3)
        
        covered, missing = analyze_key_coverage(
            answer_text,
            expected_points,
            question_difficulty=question_difficulty
        )

        detailed.append({
            "question_id": r["question_id"],
            "question_text": r["question_text"],
            "score": r["score"],
            "grade": r["grade"],
            "performance_summary": {
                "key_concepts": f"{len(covered)}/{len(expected_points)} covered ({round(len(covered)/len(expected_points)*100,1)}%)",
                "examples": f"{len([p for p in covered if p[0] in expected.get('example_scenarios', [])])}/{len(expected.get('example_scenarios', []))} provided",
                "word_count": f"{r['word_count']}/{expected.get('minimum_word_count', 0)} words",
                "criteria_score": f"{round(sum(c['weighted_score'] for c in r['criteria_performance']),1)}%"
            },
            "detailed_feedback": {
                "overall_assessment": (
                    "Excellent answer." if r["score"] >= 85
                    else "Good answer showing solid understanding."
                    if r["score"] >= 60
                    else "Your answer needs significant improvement."
                ),
                "strengths": [
                    f"✓ Strong coverage of key concepts ({len(covered)}/{len(expected_points)})"
                ] if covered else [],
                "areas_for_improvement": [
                    {
                        "area": "Missing Key Concepts",
                        "count": len(missing),
                        "details": [
                            {
                                "point": p,
                                "similarity": s,
                                "suggestion": "Include this concept explicitly"
                            } for p, s in missing
                        ]
                    }
                ] if missing else [],
                "what_was_covered": [
                    {
                        "point": p,
                        "similarity": s
                    } for p, s in covered
                ],
                "what_to_add": {
                    "missing_concepts": [
                        {
                            "point": p,
                            "suggestion": "Add this concept"
                        } for p, _ in missing
                    ]
                }
            },
            "criteria_performance": r["criteria_performance"]
        })

    # Calculate proper statistics
    unique_questions = len(set(r["question_id"] for r in raw_results["detailed_results"]))
    total_submissions = len(raw_results["detailed_results"])
    
    return {
        "summary": {
            "total_questions": unique_questions,
            "total_submissions": total_submissions,
            "answered": total_submissions,
            "average_score": raw_results["summary"]["average_score"],
            "grade_distribution": raw_results["summary"]["grade_distribution"]
        },
        "detailed_results": detailed
    }


# ======================================================
# GRAPH GENERATION & VISUALIZATION
# ======================================================
def generate_synthetic_data(questions, count=50):
    """
    Generates synthetic answers for testing visualization.
    Returns a dictionary of {question_id: [answers...]}
    """
    synthetic_answers = {}
    
    # Templates for different quality levels
    templates = {
        "excellent": [
            "Transfer learning is a technique where a pre-trained model is used on a new related task. It freezes early layers and fine-tunes later ones. This saves time and works well with limited data.",
            "In transfer learning, knowledge from a source domain is transferred to a target domain. By using pre-trained weights from ImageNet, we can train better models with less data by fine-tuning specific layers.",
            "Overfitting occurs when a model learns noise in the training data. It is detected by a gap between training and validation accuracy. Regularization and more data are common solutions."
        ],
        "average": [
            "Transfer learning uses a model trained on one thing for another. It helps when you don't have much data.",
            "It is using a pre-trained model. You can freeze layers and train others. It saves time.",
            "Overfitting is when the model is good on training data but bad on test data. You can fix it with more data."
        ],
        "poor": [
            "Transfer learning is learning about transfer.",
            "It is a deep learning method.",
            "Overfitting is bad for the model."
        ]
    }
    
    all_answers = {}
    
    print(f"Generating {count} synthetic submissions for graph testing...")
    
    for i in range(count):
        # Pick a random question
        q = random.choice(questions)
        qid = q["_id"]
        
        # Pick a random quality
        quality = random.choices(["excellent", "average", "poor"], weights=[0.4, 0.4, 0.2])[0]
        base_answer = random.choice(templates[quality])
        
        # Add some random variation
        variation_id = f"submission_{i}"
        answer_text = f"{base_answer} (Variation {i})" 
        
        # We need to structure this so batch_evaluate can handle it
        # Since batch_evaluate expects {qid: answer}, we can't easily pass multiple answers for the same QID 
        # in the standard dict unless we modify evaluate_batch or just run evaluate_batch multiple times.
        # For this graph demo, we will generate a list of result objects directly or 
        # just create many unique user_ids linked to questions.
        
        if qid not in all_answers:
            all_answers[qid] = []
        
        # Just return a flattened dict of unique_id -> answer for the batch processor if possible,
        # but the current evaluate_batch uses user_answers = {qid: answer}. 
        # So we'll have to return a list of (qid, answer) tuples and custom process them 
        # OR create a dictionary { "user_id_X": answer } and map it.
        # But evaluate_batch iterates keys as question_ids... 
        
        # Let's create a custom batch processor for graphs or just adapt the data.
        # The simplest way is to return a list of {"question": q, "answer": answer, "user_id": ...}
        
    # Actually, let's just make a list of user_answers where keys are unique user IDs 
    # and we pass the question object differently? 
    # No, existing evaluate_batch iterates user_answers.items() where KEY is assumed to be QuestionID. 
    # This is a limitation of the current simple script.
    # To support "input data many", I'll create a better batch fuction `evaluate_bulk_system`
    pass 

def evaluate_bulk_system(questions, num_submissions=50):
    """
    Generates data and runs evaluation for many users to populate graphs.
    """
    results = []
    
    templates = {
        "excellent": [
            "Transfer learning involves using a pre-trained model on a new task. It reuses features learned from large datasets like ImageNet. By freezing early layers and fine-tuning later ones, we save training time and improve performance on small datasets.",
            "Overfitting is when a model memorizes training data and fails to generalize. We detect it when training accuracy is high but validation accuracy is low. Techniques like regularization and dropout help prevent it."
        ],
        "good": [
            "Transfer learning uses a pre-trained network. We freeze some layers and train others. It is good for small data.",
            "Overfitting happens when the model learns the training data too well. The validation score will be bad. Use more data to fix it."
        ],
        "poor": [
            "Transfer learning cuts time.",
            "Overfitting is a error in code.",
            "I don't know."
        ]
    }
    
    print(f"Simulating {num_submissions} student submissions...")
    
    for i in range(num_submissions):
        # Randomly pick a question
        q = random.choice(questions)
        
        # Random quality
        quality = random.choices(["excellent", "good", "poor"], weights=[0.3, 0.5, 0.2])[0]
        
        # Get base text
        if "transfer" in q["text"].lower():
             base = templates[quality][0]
        else:
             base = templates[quality][1]
             
        # Add random noise to make word counts and scores vary
        noise = " " + " ".join(["very"] * random.randint(0, 5)) if random.random() > 0.5 else ""
        final_answer = base + noise
        
        # Evaluate
        res = evaluate_answer(q, final_answer)
        res["user_id"] = f"student_{i+1}"
        results.append(res)
        
    return results

def generate_report_graphs(results, output_file="evaluation_report_graphs.png"):
    """
    Simpler, high-level graphs for general reporting.
    """
    if not results:
        return
        
    print(f"Generating Report Graphs -> {output_file}")
    scores = [r["score"] for r in results]
    grades = [r["grade"] for r in results]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Student Performance Overview', fontsize=16)
    
    # 1. Score Distribution
    axs[0].hist(scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0].set_title('Overall Score Distribution')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Students')
    
    # 2. Grade Counts
    grade_order = ["A+", "A", "B+", "B", "C", "F"]
    grade_counts = {g: grades.count(g) for g in grade_order}
    axs[1].bar(grade_order, grade_counts.values(), color=['green', 'lightgreen', 'blue', 'lightblue', 'orange', 'red'])
    axs[1].set_title('Grade Distribution')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_thesis_graphs(results, output_file="evaluation_thesis_graphs.png"):
    """
    Scientific, detailed graphs for research papers.
    Includes Box Plots, Confusion Matrix, and Correlation Analysis.
    """
    if not results:
        return

    print(f"Generating Thesis Graphs -> {output_file}")
    
    scores = [r["score"] for r in results]
    word_counts = [r["word_count"] for r in results]
    # Check if we have true labels from synthetic generation
    true_labels = [r.get("true_quality_label", "unknown") for r in results]
    has_labels = all(l != "unknown" for l in true_labels)
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    
    # 1. Correlation: Word Count vs Score
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(word_counts, scores, alpha=0.5, c='purple')
    ax1.set_title('Correlation: Word Count vs. Evaluation Score')
    ax1.set_xlabel('Word Count')
    ax1.set_ylabel('Score')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Box Plot (if labels exist)
    if has_labels:
        ax2 = fig.add_subplot(gs[0, 1])
        data = [
            [r["score"] for r in results if r.get("true_quality_label") == "poor"],
            [r["score"] for r in results if r.get("true_quality_label") == "average"],
            [r["score"] for r in results if r.get("true_quality_label") == "excellent"]
        ]
        ax2.boxplot(data, labels=["Poor", "Average", "Excellent"], patch_artist=True)
        ax2.set_title('Model Discrimination Sensitivity')
        ax2.set_ylabel('Assigned Score')
        ax2.grid(True)
    
    # 3. Criteria Performance Breakdown
    ax3 = fig.add_subplot(gs[1, :])
    criteria_avgs = {}
    criteria_counts = {}
    
    for r in results:
        for c in r["criteria_performance"]:
            name = c["criterion"]
            criteria_avgs[name] = criteria_avgs.get(name, 0) + c["score"]
            criteria_counts[name] = criteria_counts.get(name, 0) + 1
            
    avg_scores = {k: v/criteria_counts[k] for k,v in criteria_avgs.items()}
    ax3.bar(list(avg_scores.keys()), list(avg_scores.values()), color='teal', alpha=0.8)
    ax3.set_title('Average Performance by Evaluation Criterion')
    ax3.set_ylim(0, 100)
    for i, v in enumerate(avg_scores.values()):
        ax3.text(i, v + 1, str(round(v, 1)), ha='center')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300) # Higher DPI for thesis
    plt.close()

def generate_input_data_dashboard(results, output_file="input_data_dashboard.png"):
    """
    Generates a single image containing variants of graphs based on the actual input data.
    Suitable for small datasets (N=2) or larger ones.
    """
    if not results:
        print("No results to allow graph generation.")
        return

    print(f"Generating Input Data Dashboard -> {output_file}")
    
    # Extract Data
    ids = [str(r.get("user_id", r["question_id"])) for r in results]
    scores = [r["score"] for r in results]
    grades = [r["grade"] for r in results]
    word_counts = [r["word_count"] for r in results]
    
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Input Data Performance Analysis', fontsize=16)
    gs = fig.add_gridspec(2, 2)
    
    # 1. Bar Chart: Individual Scores
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(ids, scores, color=['#4CAF50' if s >= 60 else '#F44336' for s in scores])
    ax1.set_title('Individual Score by Question/User')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 100)
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    # 2. Pie Chart: Grade Distribution (Variants of Graph)
    ax2 = fig.add_subplot(gs[0, 1])
    grade_counts = {}
    for g in grades:
        grade_counts[g] = grade_counts.get(g, 0) + 1
    
    ax2.pie(grade_counts.values(), labels=grade_counts.keys(), autopct='%1.1f%%', 
            colors=['#FFC107', '#FF5722', '#CDDC39', '#8BC34A'], startangle=90)
    ax2.set_title('Grade Distribution')

    # 3. Stacked Bar / Grouped Bar: Criteria Breakdown
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Average Criteria Score vs Overall Score Comparison
    avg_crit_scores = []
    for r in results:
        crit_scores = [c["score"] for c in r["criteria_performance"]]
        avg = mean(crit_scores) if crit_scores else 0
        avg_crit_scores.append(avg)
        
    x_indices = np.arange(len(ids))
    width = 0.35
    
    ax3.bar(x_indices - width/2, scores, width, label='Final Score', color='skyblue')
    ax3.bar(x_indices + width/2, avg_crit_scores, width, label='Avg Criteria Score', color='orange')
    
    ax3.set_ylabel('Scores')
    ax3.set_title('Final Score vs Criteria Average')
    ax3.set_xticks(x_indices)
    ax3.set_xticklabels(ids, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    # 4. Scatter Plot: Word Count vs Score
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(word_counts, scores, s=100, c='purple', alpha=0.7, edgecolors='black')
    for i, txt in enumerate(ids):
        ax4.annotate(txt, (word_counts[i], scores[i]), xytext=(5, 5), textcoords='offset points')
    
    ax4.set_title('Word Count vs Score Correlation')
    ax4.set_xlabel('Word Count')
    ax4.set_ylabel('Score')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
def evaluate_text(data):
    """
    Entry point for Node.js bridge.

    Expected input:
    {
        "question": { ... question dict ... },
        "user_answer": "string"
    }
    """
    try:
        if not isinstance(data, dict):
            return {"success": False, "error": "Input must be a JSON object"}

        question = data.get("question")
        user_answer = data.get("user_answer")

        if not question:
            return {"success": False, "error": "Missing 'question'"}

        if not user_answer:
            return {"success": False, "error": "Missing 'user_answer'"}

        result = evaluate_question_from_json(question, user_answer)

        return {
            "success": True,
            "result": result
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ======================================================
# INDIVIDUAL EVALUATION WRAPPER
# ======================================================
def evaluate_individual(question, user_answer):
    """
    Wrapper for individual question evaluation as requested.
    """
    print(f"\nEvaluating Individual Question: {question['_id']}")
    result = evaluate_question_from_json(question, user_answer)
    print(json.dumps(result, indent=2))
    return result

# ======================================================
# RUN
# ======================================================
# Duplicate main block removed

def generate_academic_graphs(results, output_file="academic_evaluation_analysis.png"):
    """
    Generates the 6 specific academic visualizations requested for project reports.
    1. Score Distribution (Histogram)
    2. Grade Distribution (Bar Chart)
    3. Word Count vs Score (Scatter Plot)
    4. Average Criteria Performance (Horizontal Bar Chart)
    5. Key Concept Coverage Percentage (Bar Chart)
    6. Score vs Concept Coverage (Scatter Plot)
    """
    if not results:
        return

    print(f"Generating Academic Analysis Graphs -> {output_file}")
    
    # Extract Data
    scores = [r["score"] for r in results]
    grades = [r["grade"] for r in results]
    word_counts = [r["word_count"] for r in results]
    concept_coverage = [r.get("key_concept_coverage", 0) * 100 for r in results] # Convert to %
    ids = [str(i) for i in range(1, len(results) + 1)] # Simple IDs for x-axis
    
    # Setup Figure (3 Rows, 2 Cols)
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2)
    fig.suptitle('Automated Semantic Evaluation System: Comprehensive Analysis', fontsize=20, y=0.95)
    
    # 1. Score Distribution (Histogram)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=10, range=(0, 100), color='#42A5F5', edgecolor='black', alpha=0.8)
    ax1.set_title('1. Score Distribution (0-100)')
    ax1.set_xlabel('Score Range')
    ax1.set_ylabel('Number of Students')
    ax1.grid(axis='y', alpha=0.3)
    # Interpretation text
    mean_score = mean(scores)
    ax1.axvline(mean_score, color='red', linestyle='dashed', linewidth=1)
    ax1.text(mean_score + 2, ax1.get_ylim()[1]*0.9, f'Mean: {mean_score:.1f}', color='red')

    # 2. Grade Distribution (Bar Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    grade_order = ["A+", "A", "B+", "B", "C", "F"]
    grade_counts = {g: grades.count(g) for g in grade_order}
    colors = ['#66BB6A', '#9CCC65', '#D4E157', '#FFEE58', '#FFCA28', '#EF5350']
    bars = ax2.bar(grade_order, grade_counts.values(), color=colors, edgecolor='black')
    ax2.set_title('2. Grade Distribution')
    ax2.set_xlabel('Grade Category')
    ax2.set_ylabel('Student Count')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')

    # 3. Word Count vs Score (Scatter Plot)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(word_counts, scores, color='#AB47BC', alpha=0.6, s=50)
    ax3.set_title('3. Word Count vs. Score')
    ax3.set_xlabel('Answer Length (Words)')
    ax3.set_ylabel('Evaluation Score')
    ax3.grid(True, linestyle='--', alpha=0.3)
    # Add trendline
    if len(word_counts) > 1:
        z = np.polyfit(word_counts, scores, 1)
        p = np.poly1d(z)
        ax3.plot(word_counts, p(word_counts), "r--", alpha=0.5)

    # 4. Average Criteria Performance (Horizontal Bar Chart)
    ax4 = fig.add_subplot(gs[1, 1])
    criteria_avgs = {}
    criteria_counts = {}
    for r in results:
        for c in r["criteria_performance"]:
            name = c["criterion"]
            criteria_avgs[name] = criteria_avgs.get(name, 0) + c["score"]
            criteria_counts[name] = criteria_counts.get(name, 0) + 1
    
    avg_scores = {k: v/criteria_counts[k] for k,v in criteria_avgs.items()}
    y_pos = np.arange(len(avg_scores))
    ax4.barh(y_pos, list(avg_scores.values()), color='#26C6DA', edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(list(avg_scores.keys()))
    ax4.set_title('4. Average Performance by Criteria')
    ax4.set_xlabel('Average Score (0-100)')
    ax4.set_xlim(0, 100)
    
    # 5. Key Concept Coverage Percentage (Bar Chart)
    ax5 = fig.add_subplot(gs[2, 0])
    # Show subset if too many, or just plot all tight
    if len(ids) > 30:
        sample_indices = sorted(random.sample(range(len(ids)), 30))
        plot_ids = [ids[i] for i in sample_indices]
        plot_cov = [concept_coverage[i] for i in sample_indices]
        x_label = "Student (Sample of 30)"
    else:
        plot_ids = ids
        plot_cov = concept_coverage
        x_label = "Student Answer Index"
        
    ax5.bar(plot_ids, plot_cov, color='#FF7043', width=0.6)
    ax5.set_title('5. Key Concept Coverage per Student')
    ax5.set_ylabel('Coverage (%)')
    ax5.set_xlabel(x_label)
    ax5.set_ylim(0, 100)
    ax5.tick_params(axis='x', rotation=90, labelsize=8)

    # 6. Score vs Concept Coverage (Scatter Plot)
    ax6 = fig.add_subplot(gs[2, 1])
    sc = ax6.scatter(concept_coverage, scores, c=scores, cmap='viridis', s=60, edgecolors='black', alpha=0.7)
    ax6.set_title('6. Score vs. Concept Coverage Correlation')
    ax6.set_xlabel('Concept Coverage (%)')
    ax6.set_ylabel('Final Score')
    ax6.grid(True, linestyle='--', alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax6)
    cbar.set_label('Score')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=300)
    plt.close()

if __name__ == "__main__":
    print("-" * 50)
    print("STARTING EVALUATION PIPELINE")
    print("-" * 50)
    
    # 1. Real User Input Processing
    print("[1] Evaluating Input User Answers...")
    batch_results = evaluate_batch(questions, user_answers)
    final_output = format_production_output(batch_results, questions, user_answers)
    
    with open("fast_semantic_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(">> Evaluation results saved to 'fast_semantic_evaluation.json'")
    
    # 2. Input Data Dashboard (Requested: variants of graph based on input data)
    print("\n[2] Generating Input Data Dashboard...")
    generate_input_data_dashboard(batch_results["detailed_results"], "input_data_dashboard.png")
    
    # 3. Graph Generation (Simulated Data for robust plotting)
    print("\n[3] Generating Extended Reports (Synthetic Data)...")
    
    # We need a decent dataset for graphs, so we simulate if there aren't many user answers
    if len(user_answers) < 10:
        print(">> Generating synthetic data for graph demonstration (N=60)...")
        graph_data = evaluate_bulk_system(questions, num_submissions=60)
    else:
        graph_data = evaluate_bulk_system(questions, num_submissions=60)
        
    generate_report_graphs(graph_data, "evaluation_report_graphs.png")
    generate_thesis_graphs(graph_data, "evaluation_thesis_graphs.png")
    generate_academic_graphs(graph_data, "academic_evaluation_analysis.png")
    
    print("\n[Success] All tasks completed.")
    print(f"1. JSON Output: fast_semantic_evaluation.json")
    print(f"2. Input Dashboard: input_data_dashboard.png (Based on input data)")
    print(f"3. Report Graph: evaluation_report_graphs.png")
    print(f"4. Thesis Graph: evaluation_thesis_graphs.png")
    print(f"5. Academic Analysis: academic_evaluation_analysis.png (Requested 6-panel)")
    print("-" * 50)