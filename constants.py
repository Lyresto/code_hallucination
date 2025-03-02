from tools import load_json, load_jsonl_with_all_models

label_map = {
    'CEJava': load_json('../label_result/CEJava.json'),
    'CEPython': load_json('../label_result/CEPython.json'),
    'HumanEval': load_json('../label_result/HumanEval.json')
}

data_map = {
    'CEJava': load_jsonl_with_all_models('../result/CEJava.jsonl'),
    'CEPython': load_jsonl_with_all_models('../result/CEPython.jsonl'),
    'HumanEval': load_jsonl_with_all_models('../result/HumanEval.jsonl')
}

schedule = load_json('../label_result/schedule.json')

factors = ['P1', 'P2', 'P3', 'P4', 'M1', 'M2']
types = ['R1', 'R21', 'R22', 'R23', 'K1', 'K2', 'K31', 'K32', 'K33', 'C1', 'C2', 'C3', 'C4', 'C5']
affections = ['A1', 'A2', 'A3', 'A4', 'A5']

models_formal_name = ['DeepSeek-Coder-1.3B', 'DeepSeek-Coder-7B', 'CodeLlama-7B', 'GPT-4', 'DeepSeek-R1']

models_simple_name = ['deepseek-coder-1.3b', 'deepseek-coder-7b', 'codellama-7b', 'gpt-4', 'deepseek-r1']

datasets = ['CEJavaRaw', 'CEPythonRaw', 'HumanEval']

datasets2desc = {
    'HumanEval': 'HumanEval (Python)',
    'CEJava': 'CoderEval (Java)',
    'CEPython': 'CoderEval (Python)'
}

type2desc = {
    'R1': 'Functional Conflicting',
    'R21': 'Non-functional Conflicting',
    'R22': 'Non-functional Conflicting',
    'R23': 'Non-functional Conflicting',
    'C1': 'Undefined Variables',
    'C2': 'Useless Statements (executed without effect)',
    'C3': 'Fragmented Logics',
    'C4': 'Inconsistent Libraries',
    'C5': 'Useless Statements (unexecuted)',
    'K1': 'Common Sense',
    'K2': 'Mathematics & Natural Science',
    'K31': 'Algorithm',
    'K32': 'Library/Project',
    'K33': 'Computer Theory'
}
excluded_desc = {'Non-functional\nConflicting', 'Common\nSense', 'Inconsistent\nLibraries'}

factor2desc = {
    'P1': 'Ambiguous',
    'P2': 'Incomplete',
    'P3': 'Overly complex',
    'P4': 'Lengthy',
    'M1': 'Poor Reasoning Ability',
    'M2': 'Lack of Relevant Knowledge'
}

affection2desc = {
    'A1': 'Incorrect Functionality',
    'A2': 'Low Execution Efficiency',
    'A3': 'Extra Memory Footprint',
    'A4': 'Poor Readability',
    'A5': 'Poor Maintainability<br>and Scalability'
}


def desc2type(__desc):
    for __k, __v in type2desc.items():
        if __v == __desc:
            return __k



