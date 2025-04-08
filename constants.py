models_formal_name = ['DeepSeek-Coder-1.3B', 'DeepSeek-Coder-7B', 'CodeLlama-7B', 'GPT-4', 'DeepSeek-R1']

models_simple_name = ['deepseek-coder-1.3b', 'deepseek-coder-7b', 'codellama-7b', 'gpt-4', 'deepseek-r1']

datasets = ['CEJavaRaw', 'CEPythonRaw', 'HumanEval']

datasets2desc = {
    'HumanEval': 'HumanEval (Python)',
    'CEJava': 'CoderEval (Java)',
    'CEPython': 'CoderEval (Python)'
}

desc2top_type = {
    'Behavior Conflicting': 'R',
    'Data Conflicting': 'R',
    'Undefined Variables': 'C',
    'Useless Statements (executed without effect)': 'C',
    'Fragmented Logics': 'C',
    'Inconsistent Libraries': 'C',
    'Useless Statements (unexecuted)': 'C',
    'Common Sense': 'K',
    'Mathematics & Natural Science': 'K',
    'Algorithm': 'K',
    'Library/Project': 'K',
    'Computer Theory': 'K'
}

excluded_desc = {'Common Sense', 'Inconsistent Libraries'}

factors_color = {
    'Ambiguous or Incomplete': '#AF7AC5',
    'Model-related Causes': '#85929E',
    'Lack of Domain-Specific Knowledge': '#7D3C98'
}

affections_color = {
    'Incorrect Functionality': '#F7DC6F',
    'Low Efficiency': '#E59866',
    'Poor Readability': '#F0B27A',
    'Poor Maintainability and Scalability': '#F8C471'
}


