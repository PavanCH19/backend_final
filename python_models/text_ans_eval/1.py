# import language_tool_python
# from sentence_transformers import SentenceTransformer, util
# import nltk
# from sklearn.feature_extraction.text import CountVectorizer

# # Download sentence tokenizer if not already
# nltk.download('punkt')

# # Initialize tools
# tool = language_tool_python.LanguageTool('en-US')
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def grammar_score(text):
#     matches = tool.check(text)
#     error_count = len(matches)
#     word_count = len(text.split())
#     grammar_score = max(0, 100 - (error_count / max(1, word_count)) * 200)
#     return round(grammar_score, 2), error_count

# def semantic_similarity(transcript, actual):
#     emb1 = model.encode(transcript, convert_to_tensor=True)
#     emb2 = model.encode(actual, convert_to_tensor=True)
#     score = util.cos_sim(emb1, emb2).item() * 100
#     return round(score, 2)

# def completeness(transcript, actual):
#     vectorizer = CountVectorizer().fit_transform([transcript, actual])
#     vectors = vectorizer.toarray()
#     common = (vectors[0] & vectors[1]).sum()
#     total = vectors[1].sum()
#     completeness_score = (common / total) * 100 if total > 0 else 0
#     return round(completeness_score, 2)

# def evaluate_answer(transcript, actual):
#     g_score, errors = grammar_score(transcript)
#     r_score = semantic_similarity(transcript, actual)
#     c_score = completeness(transcript, actual)

#     return {
#         "grammarScore": g_score,
#         "grammarFeedback": f"{errors} grammar issues found.",
#         "relevanceScore": r_score,
#         "relevanceFeedback": "Good match." if r_score > 70 else "Low conceptual similarity.",
#         "completenessScore": c_score,
#         "completenessFeedback": "Mostly complete." if c_score > 70 else "Missing some points.",
#         "finalSummary": "Overall good answer." if (r_score + c_score)/2 > 70 else "Needs improvement."
#     }

# # --------------------------------------------------
# # 🔍 Example test cases
# # --------------------------------------------------

# actual = "RNN stands for Recurrent Neural Network. It processes sequential data where previous outputs affect the next step."

# transcripts = [
#     # ✅ Very close answer
#     "RNN stands for Recurrent Neural Network. It works well with sequential data like text and speech, where previous outputs influence future predictions.",
    
#     # ⚠️ Partial understanding
#     "RNN is used for text and time series data. It handles data in a sequence but does not remember past outputs well.",
    
#     # ❌ Conceptually wrong
#     "RNN is a network used for recognizing images and videos. It works like CNN.",
    
#     # 🧠 Contains grammar mistakes
#     "RNN stands for Recurrent neural networks it used to handel sequencial data like text and time series where the previos outpoot effect the next one.",
    
#     # 🗣️ Very short and vague
#     "RNN is a kind of network.",
    
#     # ✅ Decent paraphrase
#     "Recurrent Neural Network processes time-based information such that past data helps predict the next element in a sequence."
# ]

# # Evaluate each transcript
# for i, t in enumerate(transcripts, 1):
#     print(f"\n--- TEST CASE {i} ---")
#     result = evaluate_answer(t, actual)
#     for k, v in result.items():
#         print(f"{k}: {v}")



import language_tool_python
from sentence_transformers import SentenceTransformer, util
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Download sentence tokenizer if not already
nltk.download('punkt')

# Initialize tools
tool = language_tool_python.LanguageTool('en-US')
model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- Grammar Evaluation ----------
def grammar_score(text):
    matches = tool.check(text)
    error_count = len(matches)
    word_count = len(text.split())
    grammar_score = max(0, 100 - (error_count / max(1, word_count)) * 200)
    grammar_issues = [m.message for m in matches]
    return round(grammar_score, 2), error_count, grammar_issues

# ---------- Semantic Similarity ----------
def semantic_similarity(transcript, actual):
    emb1 = model.encode(transcript, convert_to_tensor=True)
    emb2 = model.encode(actual, convert_to_tensor=True)
    score = util.cos_sim(emb1, emb2).item() * 100
    return round(score, 2)

# ---------- Completeness Check ----------
def completeness(transcript, actual):
    vectorizer = CountVectorizer(stop_words='english').fit_transform([transcript, actual])
    vectors = vectorizer.toarray()
    common = (vectors[0] & vectors[1]).sum()
    total = vectors[1].sum()
    completeness_score = (common / total) * 100 if total > 0 else 0
    return round(completeness_score, 2)

# ---------- Missing Keywords Extraction ----------
def find_missing_points(transcript, actual):
    transcript_words = set(word_tokenize(transcript.lower()))
    actual_words = set(word_tokenize(actual.lower()))
    missing = [word for word in actual_words if word.isalpha() and word not in transcript_words]
    return missing[:15]  # limit to top 15 words for clarity

# ---------- Evaluate Function ----------
def evaluate_answer(transcript, actual):
    g_score, errors, grammar_issues = grammar_score(transcript)
    r_score = semantic_similarity(transcript, actual)
    c_score = completeness(transcript, actual)
    missing_points = find_missing_points(transcript, actual)

    return {
        "grammarScore": g_score,
        "grammarIssues": grammar_issues if grammar_issues else ["No major grammar issues found."],
        "relevanceScore": r_score,
        "relevanceFeedback": "Good conceptual match." if r_score > 70 else "Low conceptual similarity. Key ideas may be missing.",
        "completenessScore": c_score,
        "missingPoints": missing_points if missing_points else ["No significant missing points."],
        "completenessFeedback": "Answer covers most required content." if c_score > 70 else "Some important information missing.",
        "finalSummary": (
            "Excellent answer with clear grammar and conceptual understanding."
            if g_score > 90 and r_score > 85 and c_score > 75
            else "Good attempt but can improve on completeness or grammar."
            if (r_score + c_score) / 2 > 65
            else "Needs significant improvement in concept coverage and writing."
        )
    }

# --------------------------------------------------
# 🔍 Example test cases
# --------------------------------------------------
actual = "RNN stands for Recurrent Neural Network. It processes sequential data where previous outputs affect the next step."

transcripts = [
    # ✅ Very close answer
    "RNN stands for Recurrent Neural Network. It works well with sequential data like text and speech, where previous outputs influence future predictions.",
    
    # ⚠️ Partial understanding
    "RNN is used for text and time series data. It handles data in a sequence but does not remember past outputs well.",
    
    # ❌ Conceptually wrong
    "RNN is a network used for recognizing images and videos. It works like CNN.",
    
    # 🧠 Contains grammar mistakes
    "RNN stands for Recurrent neural networks it used to handel sequencial data like text and time series where the previos outpoot effect the next one.",
    
    # 🗣️ Very short and vague
    "RNN is a kind of network.",
    
    # ✅ Decent paraphrase
    "Recurrent Neural Network processes time-based information such that past data helps predict the next element in a sequence."
]

# Evaluate each transcript
for i, t in enumerate(transcripts, 1):
    print(f"\n--- TEST CASE {i} ---")
    result = evaluate_answer(t, actual)
    for k, v in result.items():
        print(f"{k}: {v}")
