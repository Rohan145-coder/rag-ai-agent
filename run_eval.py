import re
import time
from rag_pipeline import ask
from eval_questions import eval_set

correct = 0
total = len(eval_set)
results_log = []

def extract_numbers_in_millions(text):
    """
    Finds numbers in the answer text and normalizes them to a
    single unit (millions) where a unit word is present, so
    '$87.6 billion' and '87,604 million' can both be compared
    on the same scale. Plain numbers (like percentages) are
    returned as-is.
    """
    text = text.replace("\u202f", " ").replace("\u00a0", " ").replace(",", "")
    numbers = []

    for match in re.finditer(r'(\d+\.?\d*)\s*billion', text, re.IGNORECASE):
        numbers.append(float(match.group(1)) * 1000)

    for match in re.finditer(r'(\d+\.?\d*)\s*million', text, re.IGNORECASE):
        numbers.append(float(match.group(1)))

    for match in re.finditer(r'(\d+\.?\d*)\s*%?', text):
        try:
            numbers.append(float(match.group(1)))
        except ValueError:
            pass

    return numbers

def check_answer(expected, answer_text):
    answer_clean = answer_text.replace("\u202f", " ").replace("\u00a0", " ")

    # Text answers (names) - plain substring match, case-insensitive
    if not expected.replace(".", "").isdigit():
        return expected.lower() in answer_clean.lower()

    # Numeric answers - tolerance-based match (within 1%)
    expected_num = float(expected)
    candidates = extract_numbers_in_millions(answer_text)
    for num in candidates:
        if expected_num == 0:
            continue
        if abs(num - expected_num) / expected_num < 0.01:  # 1% tolerance
            return True
    return False

for item in eval_set:
    question = item["question"]
    expected = item["expected_answer"]

    answer = ask(question)
    print(f"    [RAW ANSWER REPR]: {repr(answer)}")

    passed = check_answer(expected, answer)
    if passed:
        correct += 1

    results_log.append({
        "question": question,
        "expected": expected,
        "answer": answer,
        "passed": passed
    })

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {question}")
    if not passed:
        print(f"    Expected to find: '{expected}'")
        print(f"    Got: {answer[:150]}...")

    time.sleep(3)  # avoid hitting Groq's tokens-per-minute rate limit

accuracy = (correct / total) * 100
print(f"\n=== RESULTS: {correct}/{total} correct ({accuracy:.1f}% accuracy) ===")

# CI quality gate
MIN_ACCURACY = 90.0

if accuracy < MIN_ACCURACY:
    print(f"❌ RAG evaluation failed: accuracy {accuracy:.1f}% is below {MIN_ACCURACY}%")
    raise SystemExit(1)

print(f"✅ RAG evaluation passed: accuracy {accuracy:.1f}% meets the {MIN_ACCURACY}% threshold")