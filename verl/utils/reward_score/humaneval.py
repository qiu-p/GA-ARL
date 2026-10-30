import json
import re
import pickle
import zlib
import base64

_CODE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def extract_code_from_completion(completion: str, prompt: str) -> str:
    ms = _CODE_RE.findall(completion)
    code_str = ""
    if len(ms) > 0:
        code_str = ms[0]
    else:
        code_str = completion
    if code_str.startswith(prompt):
        code_str = code_str[len(prompt) :]
    return code_str


def evaluate_humaneval(
    completion: str,
    humaneval_item: dict,
    answer_reward: float = 1.0,
    format_reward: float = 0.0,
):
    """
    Evaluate a single HumanEval example.

    Returns:
        score: float
        log: str
    """
    prompt = humaneval_item["prompt"]
    test_code = humaneval_item["test"]
    entry_point = humaneval_item["entry_point"]
    
    # 1. Extract code (model usually outputs only function body)
    solution_code = extract_code_from_completion(completion, prompt)

    # 2. Construct full executable program
    full_code = (
        prompt
        + solution_code
        + "\n\n"
        + test_code
        + f"\n\ncheck({entry_point})"
    )

    # 3. Execute in sandboxed namespace
    exec_globals = {}
    try:
        exec(full_code, exec_globals)
    except Exception as e:
        return format_reward

    return format_reward + answer_reward

def compute_score(solution_str, ground_truth, format_reward=0.0, answer_reward=1.0):
    # extract test cases
    try:
        in_outs = json.loads(ground_truth)
    except Exception as e:
        print(f"Error loading test cases: {e}")
        in_outs = json.loads(pickle.loads(zlib.decompress(base64.b64decode(ground_truth.encode("utf-8")))))

    score = evaluate_humaneval(solution_str, in_outs)

    return {"score": score, "acc": score}
