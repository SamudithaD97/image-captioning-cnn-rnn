from nltk.translate.bleu_score import sentence_bleu

def calculate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())