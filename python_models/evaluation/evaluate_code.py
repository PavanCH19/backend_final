import sys
import json
import subprocess
import tempfile
import os
import traceback
import re
import numpy as np

# ---------------------------------------------------------
# CONSTANTS & CONFIG
# ---------------------------------------------------------
GRADE_SCALE = [
    (90, "A+"), (80, "A"), (70, "B+"), 
    (60, "B"), (40, "C"), (35, "F")
]

# Lazy-loaded models
_sentence_model = None

def get_sentence_model():
    """Lazy load SentenceTransformer to avoid overhead when only evaluating coding."""
    global _sentence_model
    if _sentence_model is None:
        try:
            from sentence_transformers import SentenceTransformer, util
            _sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # If not installed, we can't do deep subjective eval
            sys.stderr.write("Warning: sentence-transformers not installed. Subjective evaluation will be limited.\n")
            return None
    return _sentence_model

def get_grade(score):
    for threshold, letter in GRADE_SCALE:
        if score >= threshold:
            return letter
    return "F"

# ---------------------------------------------------------
# 1. SUBJECTIVE EVALUATION
# ---------------------------------------------------------
def evaluate_subjective(question, user_answer):
    """
    Evaluates subjective answers using Semantic Similarity.
    Features: Key point matching, Example scenario checks, Length penalties.
    """
    expected = question.get("expected_answer", {})
    key_points = expected.get("key_points", [])
    examples = expected.get("example_scenarios", [])
    
    model = get_sentence_model()
    if not model:
        return {"score": 0, "grade": "F", "error": "Model missing"}

    if not user_answer or not isinstance(user_answer, str):
        return {"score": 0, "grade": "F", "feedback": "No answer provided"}

    from sentence_transformers import util
    
    # Encode user answer
    user_emb = model.encode(user_answer, convert_to_tensor=True)
    
    # 1. Coverage of Key Points
    points_covered = 0
    missing_concepts = []
    
    for pt in key_points:
        pt_emb = model.encode(pt, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(user_emb, pt_emb).item()
        if sim > 0.45:  # Similarity threshold
            points_covered += 1
        else:
            missing_concepts.append(pt)
            
    # 2. Coverage of Examples
    examples_covered = 0
    for ex in examples:
        ex_emb = model.encode(ex, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(user_emb, ex_emb).item()
        if sim > 0.40:
             examples_covered += 1

    # Scoring
    total_points = len(key_points) + len(examples)
    if total_points == 0:
        score = 100 # Fallback if no criteria
    else:
        # Weighted: KeyPoints (60%) + Examples (40%)
        kp_score = (points_covered / len(key_points)) * 60 if key_points else 60
        ex_score = (examples_covered / len(examples)) * 40 if examples else 40
        score = kp_score + ex_score

    # Length Factor Penalty (simple version)
    min_words = expected.get("minimum_word_count", 10)
    word_count = len(user_answer.split())
    if word_count < min_words:
        score *= (word_count / min_words)

    final_score = round(min(100, max(0, score)), 1)
    
    return {
        "question_id": question.get("_id"),
        "question_type": "subjective",
        "score": final_score,
        "grade": get_grade(final_score),
        "strengths": [f"Covered {points_covered} key concepts"],
        "missing_concepts": missing_concepts
    }

# ---------------------------------------------------------
# 2. CODING EVALUATION (PYTHON)
# ---------------------------------------------------------
def clean_python_signature(signature):
    """
    Removes type hints from Python function signatures to ensure compatibility
    with basic exec() environments.
    e.g. "is_positive(n: int) -> bool" -> "is_positive(n)"
    """
    if not signature: return ""
    # 1. Remove return type hint "-> type"
    signature = re.sub(r'\s*->\s*[a-zA-Z0-9_\[\], ]+', '', signature)
    # 2. Remove parameter type hints ": type"
    # Matches ": " followed by words, brackets, or dots
    signature = re.sub(r':\s*[a-zA-Z0-9_\[\], \.]+', '', signature)
    return signature

def evaluate_coding_python(question, user_code):
    """
    Evaluates Python code using restricted exec().
    Supports: NumPy, basic primitives.
    """
    tests = question.get("test_cases", [])
    raw_signature = question.get("coding_instructions", {}).get("function_signature", "")
    
    # CLEAN SIGNATURE for wrapping: "func(a: int) -> bool" -> "func(a)"
    func_signature = clean_python_signature(raw_signature)
    func_name = func_signature.split("(")[0].strip()

    if not func_name:
         return {"score": 0, "grade": "F", "error": "Invalid function signature in question definition"}

    passed = 0
    errors = []
    
    # Safe Execution Environment
    local_env = {}
    safe_globals = {
        "np": np,
        "math": __import__("math"),
        "len": len,
        "range": range,
        "list": list,
        "set": set,
        "dict": dict,
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "any": any,
        "all": all
    }

    # Function Wrapper Logic (Auto-fix missing function def)
    if func_name not in user_code and "def " not in user_code:
        # Indent user code
        indented_code = "\n".join(["    " + line for line in user_code.splitlines()])
        user_code = f"def {func_signature}:\n{indented_code}"

    # 1. Compile/Run User Code
    try:
        exec(user_code, safe_globals, local_env)
    except SyntaxError as e:
        return {
            "score": 0, "grade": "F", 
            "error": f"Syntax Error: {e.msg} at line {e.lineno}",
            "details": f"Check your code structure near line {e.lineno}."
        }
    except Exception as e:
        return {
            "score": 0, "grade": "F", 
            "error": f"Runtime Error: {str(e)}"
        }

    if func_name not in local_env:
        return {
            "score": 0, "grade": "F",
            "error": f"Function '{func_name}' not defined. Please define function: {func_signature}"
        }
        
    func = local_env[func_name]

    test_results = []
    # 2. Run Test Cases
    for i, t in enumerate(tests):
        result_entry = {
            "case": i + 1,
            "input": str(t["input"]),
            "expected": str(t["expected_output"]),
            "status": "failed",
            "got": "error"
        }
        try:
            # Parse Input
            inp_val = eval(str(t["input"]), safe_globals)
            exp_val = eval(str(t["expected_output"]), safe_globals)

            # Call Function
            if isinstance(inp_val, tuple):
                output = func(*inp_val)
            else:
                output = func(inp_val)

            result_entry["got"] = str(output)

            # Compare
            matched = False
            if isinstance(output, (np.ndarray, list)) and isinstance(exp_val, (np.ndarray, list)):
                matched = np.allclose(output, exp_val) if isinstance(output, np.ndarray) else (output == exp_val)
            elif isinstance(output, float) or isinstance(exp_val, float):
                matched = abs(output - exp_val) < 1e-5
            else:
                matched = (output == exp_val)

            if matched:
                passed += 1
                result_entry["status"] = "passed"
            else:
                result_entry["status"] = "failed"
                errors.append(result_entry)

        except Exception as e:
            result_entry["status"] = "error"
            result_entry["error"] = str(e)
            errors.append(result_entry)
        
        test_results.append(result_entry)

    score = round((passed / len(tests)) * 100, 1) if tests else 0
    return {
        "question_id": question.get("_id"),
        "question_type": "coding",
        "score": score,
        "grade": get_grade(score),
        "passed_tests": passed,
        "total_tests": len(tests),
        "test_results": test_results,
        "errors": errors[:3] # Limit error size
    }

# ---------------------------------------------------------
# 3. CODING EVALUATION (JAVASCRIPT)
# ---------------------------------------------------------
def preprocess_js_code(code):
    """
    Cleans up JS code by removing comments and basic TypeSafe/Python-style 
    type annotations to prevent syntax errors in Node.js.
    """
    if not code: return ""
    
    # 1. Strip comments
    # Multi-line
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    # Single line
    code = re.sub(r'//.*', '', code)

    # 2. Strip return type arrow (common user mistake or TS style):  ) -> string {  =>  ) {
    # Matches ") -> type {"
    code = re.sub(r'\)\s*->\s*[a-zA-Z0-9_<>\[\]\.]+\s*\{', ') {', code)

    # 3. Strip basic TS colon types in parameters:  (val: any)
    # Matches ": type" followed by comma or closing paren
    # We target common types to avoid false positives in object literals
    common_types = ['any', 'string', 'number', 'boolean', 'void', 'object', 'int', 'float', 'list', 'dict', 'Array<.*?>']
    # Naive check: remove ": type"
    # This regex looks for : type followed by , or )
    # It assumes the type is a simple word or Array<...>
    code = re.sub(r':\s*([a-zA-Z0-9_<>\[\]]+)\s*(?=[,\)])', '', code)

    return code.strip()

def evaluate_coding_javascript(question, user_code):
    """
    Evaluates JavaScript code using a Node.js child process with VM isolation.
    Requires 'node' in system PATH.
    """
    # Preprocess Code to remove comments & types
    user_code = preprocess_js_code(user_code)

    tests = question.get("test_cases", [])
    func_signature = question.get("coding_instructions", {}).get("function_signature", "")
    func_name = func_signature.split("(")[0].strip()
    
    if not func_name:
         return {"score": 0, "grade": "F", "error": "Invalid Function Signature"}

    # Handle Arrow Functions / Expressions
    # If code looks like "(a,b) =>" or "a =>", assign it to func_name
    # Use 'var' instead of 'const' so it becomes a property of the sandbox context
    if re.match(r'^\s*\(?[\w\s,]*\)?\s*=>', user_code):
        user_code = f"var {func_name} = {user_code};"
    
    # Fallback wrapper for standard functions if missing
    elif func_name not in user_code and "function" not in user_code:
         user_code = f"function {func_signature} {{\n{user_code}\n}}"

    # Prepare Tests
    js_tests = []
    for t in tests:
        inp = str(t["input"])
        exp = str(t["expected_output"])
        
        # Normalize Python booleans/None to JS
        inp = inp.replace("True", "true").replace("False", "false").replace("None", "null")
        exp = exp.replace("True", "true").replace("False", "false").replace("None", "null")
        
        js_tests.append({"input": inp, "expected": exp})

    # Robust JS Wrapper using VM
    # We pass user_code as a specific string to be compiled by VM
    js_wrapper = f"""
    const vm = require('vm');
    
    const userCodeSource = {json.dumps(user_code)};
    const tests = {json.dumps(js_tests)};
    const funcName = "{func_name}";
    
    const results = [];
    let passed = 0;
    
    try {{
        // 1. Create Sandbox
        const sandbox = {{ 
            console: console,  // Allow logging
            // Add other standard lib items if needed
        }};
        vm.createContext(sandbox);
        
        // 2. Execute User Code in Sandbox
        vm.runInContext(userCodeSource, sandbox);
        
        // 3. Check if function is defined
        if (typeof sandbox[funcName] !== 'function') {{
            throw new Error(`Function '${{funcName}}' is not defined. Please define: ${{funcName}}`);
        }}
        
        // 4. Run Tests
        tests.forEach((t, index) => {{
            const res = {{
                case: index + 1,
                input: t.input,
                expected: t.expected,
                status: "failed",
                got: "error"
            }};
            
            try {{
                // Evaluate input args in sandbox context to match user environment
                let argsScript;
                if (t.input.trim().startsWith('[') || t.input.trim().startsWith('{{') || !t.input.includes(',')) {{
                     argsScript = `[${{t.input}}]`; // Ensure array wrapper for single obj
                }} else {{
                     argsScript = `[${{t.input}}]`;
                }}
                
                // We use a wrapper execution to call the function with args
                // "funcName(...args)"
                const runScript = `
                    (function() {{
                        const args = ${{t.input.trim().startsWith('[') && !t.input.includes(',') ? t.input : '[' + t.input + ']'}}; 
                        // Note: Heuristic for args parsing is tricky. 
                        // Better: Just eval the input list
                        return ${{funcName}}(...args);
                    }})()
                `; 
                
                // Simpler Arg Parsing Approach:
                // We construct the function call string directly: "func(arg1, arg2)"
                const callScript = `${{funcName}}(${{t.input}})`;
                
                const output = vm.runInContext(callScript, sandbox);
                
                res.got = String(output);

                // Compare
                const expected = eval(t.expected); // Eval expected in host (trusted) or sandbox?
                // Expected values are simple, safe to eval in host or we can parse JSON if rigid.
                // For robustness, let's use loose equality or JSON stringify
                
                let isCorrect = false;
                if (Array.isArray(expected)) {{
                     isCorrect = JSON.stringify(output) === JSON.stringify(expected);
                }} else {{
                     isCorrect = (output == expected); // Loose equality for "5" vs 5
                }}

                if (isCorrect) {{
                    passed++;
                    res.status = "passed";
                }}
                
            }} catch (err) {{
                res.status = "error";
                res.error = err.message;
            }}
            results.push(res);
        }});
        
        console.log(JSON.stringify({{
            passed: passed,
            total: tests.length,
            results: results
        }}));

    }} catch (err) {{
        // Catch Syntax Errors or Runtime Errors in User Code
        console.log(JSON.stringify({{ 
            error: err.name + ": " + err.message,
            results: []
        }}));
    }}
    """
    
    # Write to temp file
    fd, path = tempfile.mkstemp(suffix=".js")
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(js_wrapper)
            
        process = subprocess.run(
            ["node", path],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if process.stderr:
            # Check if it's just a warning or real error
            # Sometimes node prints debugger warnings
            pass 

        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError:
             return {
                 "score": 0, "grade": "F", 
                 "error": "Evaluator Output Error", 
                 "raw": process.stdout + " | " + process.stderr
             }

        if "error" in result and "passed" not in result:
             # User Code Error (Syntax, etc)
             return {"score": 0, "grade": "F", "error": result["error"]}

        passed = result.get("passed", 0)
        total = result.get("total", 0)
        test_results = result.get("results", [])
        errors = [r for r in test_results if r["status"] != "passed"]

        score = round((passed / total) * 100, 1) if total else 0
        return {
            "question_id": question.get("_id"),
            "question_type": "coding",
            "score": score,
            "grade": get_grade(score),
            "passed_tests": passed,
            "total_tests": total,
            "test_results": test_results,
            "errors": errors[:3]
        }
        
    except subprocess.TimeoutExpired:
        return {"score": 0, "grade": "F", "error": "Time Limit Exceeded"}
    except Exception as e:
        return {"score": 0, "grade": "F", "error": str(e)}
    finally:
        if os.path.exists(path):
            os.remove(path)

# ---------------------------------------------------------
# 4. MCQ EVALUATION
# ---------------------------------------------------------
def evaluate_mcq(question, user_answer):
    """Exact match comparison."""
    expected = question.get("expected_answer", {}).get("solution", "")
    # Or simplified structure
    if not expected: 
        expected = question.get("answer", "")
        
    is_correct = (str(user_answer).strip() == str(expected).strip())
    score = 100 if is_correct else 0
    return {
        "question_id": question.get("_id"),
        "question_type": "mcq",
        "score": score,
        "grade": get_grade(score),
        "correct": is_correct
    }

# ---------------------------------------------------------
# 5. VOICE EVALUATION
# ---------------------------------------------------------
def evaluate_voice(question, user_answer):
    """
    Voice evaluation assumes 'user_answer' is the TRANSCRIPT.
    Delegates to subjective evaluation.
    """
    return evaluate_subjective(question, user_answer)

# ---------------------------------------------------------
# DISPATCHER
# ---------------------------------------------------------
def evaluate_question(question, user_answer):
    """
    Master Dispatcher Function.
    Routes to specific evaluators based on 'question_type'.
    """
    q_type = question.get("question_type", "").lower()
    
    if q_type == "coding":
        # Check language
        lang = question.get("coding_instructions", {}).get("language", "python").lower()
        if "javascript" in lang or "node" in lang:
            return evaluate_coding_javascript(question, user_answer)
        elif "python" in lang:
            return evaluate_coding_python(question, user_answer)
        else:
            return {"score": 0, "grade": "F", "error": f"Unsupported language: {lang}"}

    elif q_type == "subjective":
        return evaluate_subjective(question, user_answer)
        
    elif q_type == "mcq" or q_type == "multiple-choice":
        return evaluate_mcq(question, user_answer)
        
    elif q_type == "voice":
        return evaluate_voice(question, user_answer)
        
    else:
        # Fallback or error
        return {"error": f"Unknown question type: {q_type}"}


# ---------------------------------------------------------
# COMPATIBILITY WRAPPER (For existing Node calls)
# ---------------------------------------------------------
def evaluate_coding(data):
    """Alias for Node controller compatibility (Bridge Adapter)."""
    question = data.get("question")
    user_code = data.get("user_code")
    return evaluate_question(question, user_code)

# ---------------------------------------------------------
# MAIN FUNCTION (EXAMPLE CALL)
# ---------------------------------------------------------
def main():
    print("----------------------------------------------------------------")
    print("RUNNING COMPREHENSIVE COMPATIBILITY & ROBUSTNESS TESTS")
    print("----------------------------------------------------------------\n")

    # -------------------------------------------------------------
    # SCENARIO 1: Valid Node.js Solution
    # -------------------------------------------------------------
    q_js_valid = {
        "_id": "test_case_01",
        "question_type": "coding",
        "coding_instructions": {
            "language": "javascript",
            "function_signature": "isServerRunning(status)"
        },
        "test_cases": [{"input": "true", "expected_output": "true"}]
    }
    code_js_valid = """
    function isServerRunning(status) {
        return status;
    }
    """
    print(f"Test 1: Valid JS Code ... ", end="")
    res = evaluate_question(q_js_valid, code_js_valid)
    print(f"[{res['grade']}] Score: {res['score']}")

    # -------------------------------------------------------------
    # SCENARIO 2: Theory/Text Input (User writes text instead of code)
    # -------------------------------------------------------------
    print(f"Test 2: Theory/Text Input (JS) ... ", end="")
    code_js_theory = "I think the answer is to check the boolean status but I dont know code."
    res = evaluate_question(q_js_valid, code_js_theory)
    if res['score'] == 0 and res.get('error'):
        print(f"[PASS] Correctly identified error: {res['error'][:50]}...")
    else:
        print(f"[FAIL] Unexpected result: {res}")

    # -------------------------------------------------------------
    # SCENARIO 3: Syntax Error (JS)
    # -------------------------------------------------------------
    print(f"Test 3: Syntax Error (JS) ... ", end="")
    code_js_syntax = "function isServerRunning(status) { return status "  # Missing brace/semicolon
    res = evaluate_question(q_js_valid, code_js_syntax)
    if res['score'] == 0 and res.get('error'):
        print(f"[PASS] Caught Syntax Error")
    else:
        print(f"[FAIL] Unexpected result: {res}")

    # -------------------------------------------------------------
    # SCENARIO 4: Valid Python Solution
    # -------------------------------------------------------------
    q_py_valid = {
        "_id": "test_case_02",
        "question_type": "coding",
        "coding_instructions": {
            "language": "python",
            "function_signature": "add_nums(a, b)"
        },
        "test_cases": [{"input": "(1, 2)", "expected_output": "3"}]
    }
    code_py_valid = """
def add_nums(a, b):
    return a + b
    """
    print(f"Test 4: Valid Python Code ... ", end="")
    res = evaluate_question(q_py_valid, code_py_valid)
    print(f"[{res['grade']}] Score: {res['score']}")

    # -------------------------------------------------------------
    # SCENARIO 5: Theory Input (Python)
    # -------------------------------------------------------------
    print(f"Test 5: Theory/Text Input (Python) ... ", end="")
    code_py_theory = "This is just a comment, not code."
    res = evaluate_question(q_py_valid, code_py_theory)
    if res['score'] == 0 and res.get('error'):
         print(f"[PASS] Correctly identified error: {res['error'][:50]}...")
    else:
         print(f"[FAIL] Unexpected result: {res}")
    
    print("\n----------------------------------------------------------------")
    print("ALL TESTS COMPLETED")
    print("----------------------------------------------------------------")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
