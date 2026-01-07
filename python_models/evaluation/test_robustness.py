import sys
import os
import json

# Add current directory to path so we can import evaluate_code
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_code import evaluate_question

def run_test(name, question, user_code, expected_status="evaluated", expected_grade_min=None):
    print(f"\n[{name}] Testing...")
    try:
        result = evaluate_question(question, user_code)
        
        # Check status (error vs evaluated)
        if expected_status == "error":
            # Expecting unsupported language or system-level error
            if result.get("error") or result.get("status") == "error":
                print(f"✅ Passed (Expected Error caught): {result.get('error')}")
            else:
                print(f"❌ Failed (Expected Error, got valid result): {result}")
                
        else: # Expected "evaluated" - meaning system handled it (even if user code failed)
            # If there's an error but score exists, it's gracefully handled
            if result.get("error") and "score" in result:
                score = result.get("score", 0)
                grade = result.get("grade", "F")
                print(f"✅ Passed (Graceful Error Handling). Score: {score}, Grade: {grade}")
                print(f"   Error: {result.get('error')[:80]}...")
            elif result.get("error"):
                # Error without score means system failure
                print(f"❌ Failed (System Error): {result.get('error')}")
            else:
                score = result.get("score", 0)
                grade = result.get("grade", "F")
                print(f"✅ Passed. Score: {score}, Grade: {grade}")
                if expected_grade_min and score < expected_grade_min:
                     print(f"⚠️ Warning: Score {score} lower than expected {expected_grade_min}")

    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Test crashed with {e}")

def main():
    print("==================================================")
    print("      ROBUSTNESS TEST SUITE FOR EVALUATOR")
    print("==================================================")

    # ----------------------------------------------------
    # 1. JS: HEAVY COMMENTS & TYPES (User Scenario)
    # ----------------------------------------------------
    q_js_dirty = {
        "_id": "js_types_01",
        "question_type": "coding",
        "coding_instructions": {"language": "javascript", "function_signature": "add(a, b)"},
        "test_cases": [{"input": "1, 2", "expected_output": "3"}]
    }
    code_js_dirty = """
    // This is a solution
    /* Multi-line 
       comment here */
    function add(a: any, b: number) -> number {
        return a + b; // inline comment
    }
    """
    run_test("JS with Types & Comments", q_js_dirty, code_js_dirty, expected_status="evaluated", expected_grade_min=100)

    # ----------------------------------------------------
    # 2. JS: SYNTAX ERROR
    # ----------------------------------------------------
    code_js_broken = "function add(a, b) { return a + "
    run_test("JS Syntax Error", q_js_dirty, code_js_broken, expected_status="evaluated") 
    # Logic note: With the new VM wrapper, this should return score 0 and a clean 'error' field like "SyntaxError: Unexpected end of input". 
    # It counts as "evaluated" because the system handled it gracefully.

    # ----------------------------------------------------
    # 3. PYTHON: WITH TYPES & COMMENTS
    # ----------------------------------------------------
    q_py_types = {
        "_id": "py_types_01",
        "question_type": "coding",
        "coding_instructions": {"language": "python", "function_signature": "is_positive(n: int) -> bool"},
        "test_cases": [{"input": "5", "expected_output": "True"}]
    }
    code_py_types = """
# This is a comment
def is_positive(n: int) -> bool:
    return n > 0
    """
    run_test("Python with Types & Comments", q_py_types, code_py_types, expected_status="evaluated", expected_grade_min=100)

    # ----------------------------------------------------
    # 4. PYTHON: INVALID INDENT
    # ----------------------------------------------------
    q_py = {
        "_id": "py_01",
        "question_type": "coding",
        "coding_instructions": {"language": "python", "function_signature": "foo(x)"},
        "test_cases": [{"input": "5", "expected_output": "5"}]
    }
    code_py_bad = """
def foo(x):
return x
    """
    run_test("Python Indentation Error", q_py, code_py_bad, expected_status="evaluated") 

    # ----------------------------------------------------
    # 4. UNSUPPORTED LANGUAGE (JAVA)
    # ----------------------------------------------------
    q_java = {
        "_id": "java_01",
        "question_type": "coding",
        "coding_instructions": {"language": "java", "function_signature": "Solution.main()"},
        "test_cases": []
    }
    code_java = "public class Solution { ... }"
    run_test("Java (Unsupported)", q_java, code_java, expected_status="error")

    # ----------------------------------------------------
    # 5. JS: MALFORMED INPUTS (String instead of code)
    # ----------------------------------------------------
    code_text = "I dont know how to code sorry"
    run_test("JS Text Garbage", q_js_dirty, code_text, expected_status="evaluated") # Score 0, SyntaxError

    # ----------------------------------------------------
    # 6. JS: Arrow Function (Implicit Return) - NOW SUPPORTED
    # ----------------------------------------------------
    code_js_arrow = "(a, b) => a + b" 
    run_test("JS Arrow Function", q_js_dirty, code_js_arrow, expected_status="evaluated", expected_grade_min=100)

if __name__ == "__main__":
    main()
