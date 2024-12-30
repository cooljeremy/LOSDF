import logging
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
import re

# Setup Log
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_em_f1(predictions, labels):
    """
    Calculate Exact Match (EM) and F1 scores.

    Args.
        predictions: list of predicted answers.
        labels: list of true answers.

    Returns.
        A dictionary with EM and F1 scores.
    """
    em = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"em": em, "f1": f1}

def calculate_bleu(predictions, references):
    """
    Calculate the BLEU score.

    Args.
        predictions: list of predicted sentences.
        references: list of references, each predicted sentence can have multiple references.

    Returns: BLEU score.
        BLEU score.
    """
    bleu_scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.split()
        ref_counts = [Counter(ref.split()) for ref in refs]
        total_count = Counter()
        for ref_count in ref_counts:
          total_count |= ref_count
        
        match_count = 0
        for token in pred_tokens:
          if total_count[token] > 0:
            match_count += 1
            total_count[token] -= 1
        
        bleu_scores.append(match_count / len(pred_tokens) if len(pred_tokens) > 0 else 0)

    return np.mean(bleu_scores)

def calculate_rouge(predictions, references):
    """
    Calculate the ROUGE-L score.

    Args.
        predictions: list of predicted sentences.
        references: list of references, each predicted sentence can have multiple references.

    Returns: ROUGE-L score.
        ROUGE-L score.
    """
    rouge_scores = []
    for pred, refs in zip(predictions, references):
        pred_tokens = pred.split()
        scores = []
        for ref in refs:
            ref_tokens = ref.split()

            # Compute the longest common subsequence
            dp = [[0] * (len(ref_tokens) + 1) for _ in range(len(pred_tokens) + 1)]
            for i in range(1, len(pred_tokens) + 1):
                for j in range(1, len(ref_tokens) + 1):
                    if pred_tokens[i - 1] == ref_tokens[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            lcs = dp[len(pred_tokens)][len(ref_tokens)]
            
            precision = lcs / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = lcs / len(ref_tokens) if len(ref_tokens) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            scores.append(f1)

        rouge_scores.append(max(scores))

    return np.mean(rouge_scores)

def normalize_answer(s):
    """
    Some simple standardization of answers.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))