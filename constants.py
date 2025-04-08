factors = ['P1', 'P2', 'P3', 'P4', 'M1', 'M2']
types = ['R1', 'R21', 'R22', 'R23', 'K1', 'K2', 'K31', 'K32', 'K33', 'C1', 'C2', 'C3', 'C4', 'C5']
affections = ['A1', 'A2', 'A3', 'A4', 'A5']

models_formal_name = ['DeepSeek-Coder-1.3B', 'DeepSeek-Coder-7B', 'CodeLlama-7B', 'GPT-4', 'DeepSeek-R1']

models_simple_name = ['deepseek-coder-1.3b', 'deepseek-coder-7b', 'codellama-7b', 'gpt-4', 'deepseek-r1']
# models_simple_name = ['deepseek-coder-1.3b-cot2', 'deepseek-coder-1.3b-refine']
# models_simple_name = ['deepseek-coder-1.3b-RAG']

datasets = ['CEJavaRaw', 'CEPythonRaw', 'HumanEval']

datasets2desc = {
    'HumanEval': 'HumanEval (Python)',
    'CEJava': 'CoderEval (Java)',
    'CEPython': 'CoderEval (Python)'
}

type2desc = {
    # 'R1': 'Functional Conflicting',
    'R11': 'Behavior Conflicting',
    'R12': 'Data Conflicting',
    # 'R21': 'Non-functional Conflicting',
    # 'R22': 'Non-functional Conflicting',
    # 'R23': 'Non-functional Conflicting',
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
desc2top_type = {desc: tp[0] for tp, desc in type2desc.items()}

excluded_desc = {'Non-functional Conflicting', 'Common Sense', 'Inconsistent Libraries'}


# class Types:
#     FunctionalConflicting = 'Functional Conflicting'
#     NonFunctionalConflicting = 'Non-Functional Conflicting'
#     UndefinedVariables = 'Undefined Variables'
#     UselessStatementsExecuted = 'Useless Statements (executed without effect)'
#     FragmentedLogics = 'Fragmented Logics'
#     InconsistentLibraries = 'Inconsistent Libraries'
#     UselessStatementsUnexecuted = 'Useless Statements (unexecuted)'
#     CommonSense = 'Common Sense'
#     MathematicsAndNaturalScience = 'Mathematics & Natural Science'
#     Algorithm = 'Algorithm'
#     LibraryProject = 'Library/Project'
#     ComputerTheory = 'Computer Theory'


factor2desc = {
    'P1': 'Ambiguous or Incomplete',
    'P2': 'Ambiguous or Incomplete',
    'P3': 'Model-related Causes',
    'P4': 'Model-related Causes',
    'M1': 'Model-related Causes',
    'M2': 'Lack of Domain-Specific Knowledge'
}

factors_color = {
    'P1': '#AF7AC5',
    'P2': '#AF7AC5',
    'P3': '#85929E',
    'P4': '#85929E',
    'M1': '#85929E',
    'M2': '#7D3C98'
}
factors_color = {factor2desc[k]: v for k, v in factors_color.items()}

affection2desc = {
    'A1': 'Incorrect Functionality',
    'A2': 'Low Efficiency',
    'A3': 'Low Efficiency',
    'A4': 'Poor Readability',
    'A5': 'Poor Maintainability and Scalability'
}

affections_color = {
    'A1': '#F7DC6F',
    'A2': '#E59866',
    'A3': '#E59866',
    'A4': '#F0B27A',
    'A5': '#F8C471'
}
affections_color = {affection2desc[k]: v for k, v in affections_color.items()}

text_complexity_mapper = {
    '1': 1,
    '2': 2,
    '3': 2,
    '4': 3,
    '5': 3
}


def desc2type(__desc):
    for __k, __v in type2desc.items():
        if __v == __desc:
            return __k



