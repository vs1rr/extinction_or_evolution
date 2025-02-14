import pandas as pd
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau
import numpy as np
import matplotlib.cm as cm

class SimilarityCalculator:
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings1 = None
        self.embeddings2 = None

    def compute_embeddings(self, list1, list2):
        self.embeddings1 = self.semantic_model.encode(list1, convert_to_tensor=True).cpu().numpy()
        self.embeddings2 = self.semantic_model.encode(list2, convert_to_tensor=True).cpu().numpy()

    def compute_jaccard_similarity(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0

    def compute_precision_recall_f1(self, list1, list2):
        set1, set2 = set(list1), set(list2)
        precision = len(set1 & set2) / len(set2) if set2 else 0
        recall = len(set1 & set2) / len(set1) if set1 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    def compute_cosine_similarity(self, list1, list2):
        text1 = ' '.join(list1)
        text2 = ' '.join(list2)
        vectorizer = CountVectorizer().fit([text1, text2])
        vectors = vectorizer.transform([text1, text2]).toarray()
        return cosine_similarity(vectors)[0, 1]

    def compute_semantic_similarity(self):
        mean_embedding1 = self.embeddings1.mean(axis=0)
        mean_embedding2 = self.embeddings2.mean(axis=0)
        return cosine_similarity(mean_embedding1.reshape(1, -1), mean_embedding2.reshape(1, -1))[0, 0]

    def compute_rank_biased_overlap(self, list1, list2, p=0.9):
        def rbo(S, T, p):
            S_set, T_set = set(), set()
            overlap = 0
            rbo_ext = 0
            for i in range(1, max(len(S), len(T)) + 1):
                if i <= len(S):
                    S_set.add(S[i - 1])
                if i <= len(T):
                    T_set.add(T[i - 1])
                overlap = len(S_set & T_set)
                rbo_ext += (overlap / i) * (p ** (i - 1))
            return (1 - p) * rbo_ext
        return rbo(list1, list2, p)

    def compute_mean_pairwise_cosine_similarity(self):
        pairwise_similarities = cosine_similarity(self.embeddings1, self.embeddings2)
        return np.mean(pairwise_similarities)

    def compute_max_pairwise_cosine_similarity(self):
        pairwise_similarities = cosine_similarity(self.embeddings1, self.embeddings2)
        return np.max(pairwise_similarities)

    def compute_kendalls_tau(self, list1, list2):
        rank1 = {word: idx for idx, word in enumerate(list1)}
        rank2 = {word: idx for idx, word in enumerate(list2)}
        common_items = set(rank1.keys()).intersection(set(rank2.keys()))
        rank_vector1 = [rank1[item] for item in common_items]
        rank_vector2 = [rank2[item] for item in common_items]
        if len(rank_vector1) < 2:
            return 0  # Not enough data for meaningful Kendall's Tau
        tau, _ = kendalltau(rank_vector1, rank_vector2)
        return tau

    def normalize_kendalls_tau(self, tau):
        return (tau + 1) / 2

    def compute_all_metrics(self, list1, list2):
        self.compute_embeddings(list1, list2)
        precision, recall, f1 = self.compute_precision_recall_f1(list1, list2)
        return {
            "JS": self.compute_jaccard_similarity(list1, list2),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "CS": self.compute_cosine_similarity(list1, list2),
            "SS": self.compute_semantic_similarity(),
            "RBO": self.compute_rank_biased_overlap(list1, list2),
            "MeanPCS": self.compute_mean_pairwise_cosine_similarity(),
            "MaxPCS": self.compute_max_pairwise_cosine_similarity(),
            "KT": self.compute_kendalls_tau(list1, list2),
            "NKT": self.normalize_kendalls_tau(self.compute_kendalls_tau(list1, list2))
        }


def process_files(file1, file2, label):
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    all_synonyms1, all_synonyms2 = [], []
    
    for word in df1['original_word']:
        if word in df2['original_word'].values:
            synonyms1 = eval(df1[df1['original_word'] == word]['synonyms'].values[0])
            synonyms2 = eval(df2[df2['original_word'] == word]['synonyms'].values[0])
            all_synonyms1.extend(synonyms1)
            all_synonyms2.extend(synonyms2)
    
    sim_calc = SimilarityCalculator()
    metrics = sim_calc.compute_all_metrics(all_synonyms1, all_synonyms2)
    metrics['Type'] = label
    return metrics

def plot_results(results, save_path="results/figures"):
    os.makedirs(save_path, exist_ok=True)
    
    for result in results:
        label = result["Type"]  # Extract label
        df = pd.DataFrame([result])  # Convert dictionary to DataFrame
        df.set_index('Type', inplace=True)

        # Create a unique filename for each label
        csv_file_path = os.path.join(save_path, f"adjs_similarity_metrics_results_{label.replace(' ', '_')}.csv")
        df.to_csv(csv_file_path)
        print(f"Results saved to CSV: {csv_file_path}")

# Process files and compare
low_keyness_gpt = "./results/llms/gpt/diachronic_gpt/adjs_low_keyness_words.tsv"
high_keyness_gpt = "./results/llms/gpt/diachronic_gpt/adjs_high_keyness_words.tsv"
low_keyness_wordsapi = "./results/thesaurus/merriam_webster/diachronic_wordsAPI/adjs_low_keyness_words_thesaurus.tsv"
high_keyness_wordsapi = "./results/thesaurus/merriam_webster/diachronic_wordsAPI/adjs_high_keyness_words_thesaurus.tsv"

results = []
results.append(process_files(low_keyness_gpt, low_keyness_wordsapi, "Low Keyness"))
results.append(process_files(high_keyness_gpt, high_keyness_wordsapi, "High Keyness"))

# Generate plot
plot_results(results)
