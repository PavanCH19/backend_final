import json
import math
from statistics import mean
import random
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

# ======================================================
# MODEL CONFIG
# ======================================================
SIMILARITY_MODEL = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GRADE_SCALE = [
    (90, "A+"),
    (80, "A"),
    (70, "B+"),
    (60, "B"),
    (40, "C"),
    (35, "F")
]

# Lazy loading to avoid import delays if possible, but standard practice usually loads at top.
# Assuming this runs in a persistent process (like Flask/Node bridge), loading once is good.
print("Loading models...")
embedder = SentenceTransformer(SIMILARITY_MODEL)
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
    Returns similarity in [0, 1]
    """
    emb_a = embedder.encode(a, normalize_embeddings=True, convert_to_numpy=True)
    emb_b = embedder.encode(b, normalize_embeddings=True, convert_to_numpy=True)
    similarity = float(util.cos_sim(emb_a, emb_b)[0][0])
    return max(0.0, min(1.0, similarity))

# ======================================================
# SEMANTIC FEEDBACK
# ======================================================
def semantic_feedback(answer, expected_points, top_k=2):
    if not expected_points:
        return {"strengths": [], "improvements": []}

    scores = [semantic_similarity(answer, p) for p in expected_points]
    ranked = sorted(zip(expected_points, scores), key=lambda x: x[1], reverse=True)

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

    # Overall semantic alignment
    base_similarity = semantic_similarity(answer, ideal_text)

    # Length influence
    wc = word_count(answer)
    min_words = expected.get("minimum_word_count", 0)
    if min_words:
        ratio = wc / min_words
        if ratio >= 1.0:
            length_factor = min(1.05, 1.0 + (ratio - 1.0) * 0.10)
        elif ratio >= 0.7:
            length_factor = 0.92 + (ratio - 0.7) * 0.27
        else:
            length_factor = 0.75 + (ratio / 0.7) * 0.17
    else:
        length_factor = 1.0

    # Criteria evaluation
    criteria_results = []
    weighted_sum = 0.0

    for c in criteria:
        # IMPROVEMENT: Do not include full question text in criterion_context to avoid matching question-copying
        # Instead, use the criterion description and a hint of the question's core subject if needed.
        # But usually criterion description is specific enough.
        criterion_context = (
            c["criterion"] + ": " +
            c.get("description", "")
        )
        sim = semantic_similarity(answer, criterion_context)
        
        # Check against question text - if answer is too similar to question, it's a weak signal
        q_sim = semantic_similarity(answer, question["text"])
        if q_sim > 0.85:
            # Penalty for just copying the question
            sim *= (1.0 - (q_sim - 0.8)) 

        score = min(100.0, sim * 120.0) # Scaled up slightly because pure semantic search can be conservative
        weighted = score * c["weight"]
        weighted_sum += weighted

        criteria_results.append({
            "criterion": c["criterion"],
            "score": round(score, 1),
            "weight": c["weight"],
            "weighted_score": round(weighted, 1)
        })

    # Coverage-based adjustment
    key_points = expected.get("key_points", [])
    example_scenarios = expected.get("example_scenarios", [])
    
    difficulty = question.get("difficulty", 3)
    # Stricter thresholds for higher difficulty
    KEYPOINT_THRESHOLD = 0.35 if difficulty <= 2 else 0.42
    EXAMPLE_THRESHOLD = 0.35 if difficulty <= 2 else 0.40
    
    key_points_covered = 0
    for point in key_points:
        if semantic_similarity(answer, point) >= KEYPOINT_THRESHOLD:
            key_points_covered += 1
            
    examples_covered = 0
    for example in example_scenarios:
        if semantic_similarity(answer, example) >= EXAMPLE_THRESHOLD:
            examples_covered += 1
    
    key_coverage_ratio = key_points_covered / len(key_points) if key_points else 1.0
    example_coverage_ratio = examples_covered / len(example_scenarios) if example_scenarios else 1.0
    
    # Coverage score (0-100)
    if key_points or example_scenarios:
        coverage_score = (key_coverage_ratio * 70 + example_coverage_ratio * 30)
    else:
        coverage_score = 100.0
    
    # ALIGNMENT FIX: Merge weighted_sum (criteria) and coverage_score
    # Main score should be a weighted combination, not a pure product that drops significantly
    # Criteria (60%) + Coverage (40%)
    integrated_score = (weighted_sum * 0.65) + (coverage_score * 0.35)
    
    # Apply length factor
    integrated_score *= length_factor
    
    # Check for near-identical question text (Plagiarism check)
    q_copy_ratio = semantic_similarity(answer, question["text"])
    if q_copy_ratio > 0.90:
        integrated_score *= 0.3 # Heavy penalty for copying the question
        
    # High quality boost
    if coverage_score >= 80 and weighted_sum >= 80:
        integrated_score = min(100.0, integrated_score * 1.05)
        
    # Human-alignment override: If they hit all points and criteria are good, it's an A
    if weighted_sum >= 85 and coverage_score >= 85:
        final_score = max(integrated_score, 90.0)
    elif weighted_sum >= 70 and coverage_score >= 70:
        final_score = max(integrated_score, 75.0)
    else:
        final_score = integrated_score
        
    final_score = max(0.0, min(100.0, final_score))

    feedback = semantic_feedback(answer, expected_points)

    return {
        "question_id": question["_id"],
        "question_text": question["text"],
        "score": round(final_score, 1),
        "grade": grade(final_score),
        "word_count": wc,
        "criteria_performance": criteria_results,
        "key_concept_coverage": key_coverage_ratio,
        "example_coverage": example_coverage_ratio,
        "feedback": feedback
    }

def analyze_key_coverage(answer, expected_points, question_difficulty=3, threshold=None):
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

# ======================================================
# OUTPUT FORMATTING
# ======================================================
def format_single_result(r, q, answer_text):
    """
    Formats the raw evaluation result into the production schema expected by the application.
    """
    expected = q["expected_answer"]
    expected_points = (
        expected.get("key_points", []) +
        expected.get("example_scenarios", [])
    )
    question_difficulty = q.get("difficulty", 3)
    
    covered, missing = analyze_key_coverage(
        answer_text,
        expected_points,
        question_difficulty=question_difficulty
    )

    percent_covered = round(len(covered)/len(expected_points)*100, 1) if expected_points else 0.0
    
    num_examples_covered = len([p for p in covered if p[0] in expected.get('example_scenarios', [])])
    total_examples = len(expected.get('example_scenarios', []))
    
    criteria_score = round(sum(c['weighted_score'] for c in r['criteria_performance']), 1)

    return {
        "question_id": r["question_id"],
        "question_text": r["question_text"],
        "score": r["score"],
        "grade": r["grade"],
        "performance_summary": {
            "key_concepts": f"{len(covered)}/{len(expected_points)} covered ({percent_covered}%)",
            "examples": f"{num_examples_covered}/{total_examples} provided",
            "word_count": f"{r['word_count']}/{expected.get('minimum_word_count', 0)} words",
            "criteria_score": f"{criteria_score}%"
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
    }

def evaluate_question_from_json(question, user_answer):
    """
    Evaluates ONE question and ONE answer, returning production-ready format.
    """
    if not isinstance(question, dict):
        return {"error": "Invalid question format"}

    if not isinstance(user_answer, str) or not user_answer.strip():
        return {"error": "Invalid user_answer"}

    # Run core logic
    raw_result = evaluate_answer(question, user_answer)
    
    # Format to match expected output structure
    formatted_result = format_single_result(raw_result, question, user_answer)
    
    return formatted_result

def evaluate_batch(questions, user_answers):
    results = []
    q_map = {q["_id"]: q for q in questions}

    for qid, answer in user_answers.items():
        if qid in q_map:
            result = evaluate_answer(q_map[qid], answer)
            # Decorate to match format_production_output usage
            result["user_id"] = qid 
            results.append(result)
            
    return {"detailed_results": results} # customized for single use if needed

# ======================================================
# GRAPH GENERATION & VISUALIZATION
# ======================================================
def generate_synthetic_data(questions, count=50):
    # Logic from 4.py for generating data
    # Simplified here to just return empty if not needed, but provided for completeness
    pass 

def evaluate_bulk_system(questions, num_submissions=50):
    results = []
    templates = {
        "excellent": [
            "Transfer learning involves using a pre-trained model on a new task. It reuses features learned from large datasets like ImageNet. By freezing early layers and fine-tuning later ones, we save training time and improve performance on small datasets.",
            "Overfitting is when a model memorizes training data and fails to generalize. We detect it when training accuracy is high but validation accuracy is low. Techniques like regularization and dropout help prevent it."
        ],
        "good": [
            "Transfer learning uses a pre-trained network. We freeze some layers and train others.",
            "Overfitting happens when the model learns the training data too well."
        ],
        "poor": [
            "Transfer learning cuts time.",
            "I don't know."
        ]
    }
    
    for i in range(num_submissions):
        q = random.choice(questions)
        quality = random.choices(["excellent", "good", "poor"], weights=[0.3, 0.5, 0.2])[0]
        base = templates[quality][0] if "transfer" in q["text"].lower() else templates[quality][1]
        final_answer = base + (" very" * random.randint(0, 3))
        res = evaluate_answer(q, final_answer)
        res["user_id"] = f"student_{i+1}"
        results.append(res)
        
    return results

def generate_report_graphs(results, output_file="evaluation_report_graphs.png"):
    if not results: return
    scores = [r["score"] for r in results]
    grades = [r["grade"] for r in results]
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].hist(scores, bins=10, color='skyblue', edgecolor='black', alpha=0.7)
    axs[0].set_title('Overall Score Distribution')
    
    grade_order = ["A+", "A", "B+", "B", "C", "F"]
    grade_counts = {g: grades.count(g) for g in grade_order}
    axs[1].bar(grade_order, grade_counts.values(), color=['green', 'lightgreen', 'blue', 'lightblue', 'orange', 'red'])
    axs[1].set_title('Grade Distribution')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_thesis_graphs(results, output_file="evaluation_thesis_graphs.png"):
    if not results: return
    scores = [r["score"] for r in results]
    word_counts = [r["word_count"] for r in results]
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(word_counts, scores, alpha=0.5, c='purple')
    ax1.set_title('Word Count vs. Score')
    
    criteria_avgs = {}
    criteria_counts = {}
    for r in results:
        for c in r["criteria_performance"]:
            name = c["criterion"]
            criteria_avgs[name] = criteria_avgs.get(name, 0) + c["score"]
            criteria_counts[name] = criteria_counts.get(name, 0) + 1
            
    avg_scores = {k: v/criteria_counts[k] for k,v in criteria_avgs.items()}
    ax3 = fig.add_subplot(gs[1, :])
    ax3.bar(list(avg_scores.keys()), list(avg_scores.values()), color='teal')
    ax3.set_title('Criteria Performance')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_academic_graphs(results, output_file="academic_evaluation_analysis.png"):
    if not results: return
    scores = [r["score"] for r in results]
    grades = [r["grade"] for r in results]
    word_counts = [r["word_count"] for r in results]
    
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(scores, bins=10, range=(0,100), color='#42A5F5', edgecolor='black')
    ax1.set_title('Score Distribution')
    
    ax2 = fig.add_subplot(gs[0, 1])
    grade_order = ["A+", "A", "B+", "B", "C", "F"]
    grade_counts = {g: grades.count(g) for g in grade_order}
    ax2.bar(grade_order, grade_counts.values())
    ax2.set_title('Grade Distribution')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(word_counts, scores, color='#AB47BC')
    ax3.set_title('Word Count vs Score')
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_input_data_dashboard(results, output_file="input_data_dashboard.png"):
    if not results: return
    ids = [str(r.get("user_id", r["question_id"])) for r in results]
    scores = [r["score"] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.bar(ids, scores)
    plt.title('Individual Scores')
    plt.savefig(output_file)
    plt.close()

# ======================================================
# NODE.JS BRIDGE
# ======================================================
def evaluate_text(data):
    """
    Entry point for Node.js bridge.
    Expected input: { "question": {...}, "user_answer": "string" }
    """
    try:
        if not isinstance(data, dict):
            return {"error": "Input must be a JSON object"}
            
        question = data.get("question")
        user_answer = data.get("user_answer")

        if not question or not user_answer:
            return {"error": "Missing question or user_answer"}

        return evaluate_question_from_json(question, user_answer)

    except Exception as e:
        return {"error": str(e), "traceback": str(e)}

if __name__ == "__main__":
    # Test block
    questions = [{
        "_id": "test_q",
        "text": "What is overfitting?",
        "expected_answer": {
            "key_points": ["Memorizes data", "Poor generalization"],
            "example_scenarios": ["Training acc high, validation low"],
            "minimum_word_count": 10
        },
        "evaluation_criteria": [{"criterion": "Accuracy", "weight": 1.0}]
    }]
    user_answer = "Overfitting is when model memorizes data."
    print(json.dumps(evaluate_question_from_json(questions[0], user_answer), indent=2))