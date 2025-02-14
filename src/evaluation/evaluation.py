import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import seaborn as sns
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.cm as cm  # For colormap
import matplotlib.colors as mcolors  # For normalizing values to the colormap

class SimilarityCalculator:
    def __init__(self):
        # Initialize the SentenceTransformer model for semantic similarity
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2') #TODO TO CHANGE
        self.embeddings1 = None
        self.embeddings2 = None

    def compute_embeddings(self, list1, list2):
        """
        Precomputes embeddings for the two lists and stores them.
        """
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
        """
        Computes semantic similarity using precomputed embeddings.
        """
        mean_embedding1 = self.embeddings1.mean(axis=0)
        mean_embedding2 = self.embeddings2.mean(axis=0)
        return cosine_similarity(mean_embedding1.reshape(1, -1), mean_embedding2.reshape(1, -1))[0, 0]

    def compute_rank_biased_overlap(self, list1, list2, p=0.9):
        """
        Computes Rank-Biased Overlap (RBO) between two ranked lists.
        """
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
        """
        Computes the mean pairwise cosine similarity between embeddings.
        """
        pairwise_similarities = cosine_similarity(self.embeddings1, self.embeddings2)
        return np.mean(pairwise_similarities)

    def compute_max_pairwise_cosine_similarity(self):
        """
        Computes the maximum pairwise cosine similarity between embeddings.
        """
        pairwise_similarities = cosine_similarity(self.embeddings1, self.embeddings2)
        return np.max(pairwise_similarities)

    def compute_kendalls_tau(self, list1, list2):
        """
        Computes Kendall's Tau correlation for ranked lists.
        """
        # Create rank dictionaries for each list
        rank1 = {word: idx for idx, word in enumerate(list1)}
        rank2 = {word: idx for idx, word in enumerate(list2)}

        # Combine rankings for common items
        common_items = set(rank1.keys()).intersection(set(rank2.keys()))
        rank_vector1 = [rank1[item] for item in common_items]
        rank_vector2 = [rank2[item] for item in common_items]

        if len(rank_vector1) < 2:
            return 0  # Not enough data for meaningful Kendall's Tau

        # Compute Kendall's Tau
        tau, _ = kendalltau(rank_vector1, rank_vector2)
        return tau

    def normalize_kendalls_tau(self, tau):
        """
        Normalizes Kendall's Tau to [0, 1].
        """
        return (tau + 1) / 2

    def compute_all_metrics(self, list1, list2):
        # Precompute embeddings to ensure consistency
        self.compute_embeddings(list1, list2)

        jaccard = self.compute_jaccard_similarity(list1, list2)
        precision, recall, f1 = self.compute_precision_recall_f1(list1, list2)
        cosine = self.compute_cosine_similarity(list1, list2)
        semantic = self.compute_semantic_similarity()
        rbo = self.compute_rank_biased_overlap(list1, list2)
        mean_pairwise = self.compute_mean_pairwise_cosine_similarity()
        max_pairwise = self.compute_max_pairwise_cosine_similarity()
        kendalls_tau = self.compute_kendalls_tau(list1, list2)
        normalized_kendalls_tau = self.normalize_kendalls_tau(kendalls_tau)
        
        return {
            "Jaccard Similarity": jaccard,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Cosine Similarity": cosine,
            "Semantic Similarity": semantic,
            "Rank-Biased Overlap (RBO)": rbo,
            "Mean Pairwise Cosine Similarity": mean_pairwise,
            "Max Pairwise Cosine Similarity": max_pairwise,
            "Kendall's Tau (Normalized)": normalized_kendalls_tau,
        }

    @staticmethod
    def normalize_metrics(metrics):
        """
        Normalize metrics to [0, 1].
        """
        normalized_metrics = metrics.copy()
        return normalized_metrics

    
    def save_results_figure(self, metrics, save_path="results/figures"):
        """
        Saves a visually enhanced bar chart of metrics to the specified directory.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Normalize metrics
        normalized_metrics = self.normalize_metrics(metrics)
        
        labels = list(normalized_metrics.keys())
        values = list(normalized_metrics.values())
        
        # Normalize the values to [0, 1] for the colormap
        norm = cm.colors.Normalize(vmin=min(values), vmax=max(values))
        cmap = cm.get_cmap("plasma")  # Changed colormap for variety
        colors = [cmap(norm(value)) for value in values]
        
        # Create the bar chart
        plt.figure(figsize=(max(12, len(labels) * 1.5), 8))  # Dynamic sizing
        
        bars = plt.bar(range(len(labels)), values, color=colors, edgecolor='black')
        
        plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right', fontsize=14, fontweight='bold')
        plt.yticks(fontsize=12)
        
        # Add value annotations
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=height + 0.02,
                s=f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )
        
        # Add title and labels
        plt.title("Normalized Similarity Metrics Comparison", fontsize=18, fontweight='bold', pad=20)
        plt.ylabel("Normalized Score", fontsize=14, fontweight='bold')
        plt.xlabel("Metrics", fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Generate a timestamped filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"normalized_similarity_metrics_{timestamp}.png"
        save_full_path = os.path.join(save_path, file_name)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_full_path, dpi=300, bbox_inches="tight")  # High DPI for better quality
        plt.close()
        print(f"Figure saved at: {save_full_path}")


def compare_tsv_files(file1, file2):
    # Load the TSV files
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')

    # Combine all synonyms into aggregated lists
    all_synonyms1 = []
    all_synonyms2 = []

    for word in df1['original_word']:
        if word in df2['original_word'].values:
            synonyms1 = eval(df1[df1['original_word'] == word]['synonyms'].values[0])
            synonyms2 = eval(df2[df2['original_word'] == word]['synonyms'].values[0])
            all_synonyms1.extend(synonyms1)
            all_synonyms2.extend(synonyms2)

    # Extract folder names for llms and thesaurus
    llms_folder = os.path.basename(os.path.dirname(file2))  # Folder name after llms
    thesaurus_folder = os.path.basename(os.path.dirname(file1))  # Folder name after thesaurus

    # Determine the type (simple or complex) from the file paths
    word_type = "simple" if "simple" in file1.lower() or "simple" in file2.lower() else "complex"

    # Define the base save directory
    base_save_path = "./chat-gpt_thesaurus/results/figures"
    save_path = os.path.join(base_save_path, f"{llms_folder}_{thesaurus_folder}")
    os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist

    # Initialize the similarity calculator
    sim_calc = SimilarityCalculator()

    # Compute overall similarity metrics
    metrics = sim_calc.compute_all_metrics(all_synonyms1, all_synonyms2)

    # Save overall results as a figure with adjusted naming
    figure_name = f"{llms_folder}_{thesaurus_folder}_{word_type}.png"
    sim_calc.save_results_figure(metrics, save_path=save_path)

    # Save aggregated results to CSV
    csv_name = f"aggregated_similarity_results_{llms_folder}_{thesaurus_folder}_{word_type}.csv"
    results_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
    results_df.to_csv(os.path.join(save_path, csv_name))
    print(f"Aggregated results saved to {os.path.join(save_path, csv_name)}")

    # Display the results for the user
    print("Aggregated Similarity Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Example usage
if __name__ == "__main__":

    ############# COMPLEX AND SIMPLE ###########################

    # # SIMPLE WORDS API + GPT
    # file1 = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI/simplest_words_thesaurus.tsv"
    # file2 = "./results/llms/gpt/complex_simple_gpt/synonyms_simple.tsv"
    # compare_tsv_files(file1, file2)
    # # COMPLEX WORDS API + GPT
    # file3 = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI/most_complex_words_thesaurus.tsv"
    # file4 = "./results/llms/gpt/complex_simple_gpt/synonyms_complex.tsv"
    # compare_tsv_files(file3, file4)

    # SIMPLE MW + GPT
    # file1 = "./results/thesaurus/merriam_webster/complex_simple/simplest_words_merriam_webster.tsv"
    # file2 = "./results/llms/gpt/complex_simple_gpt/synonyms_simple.tsv"
    # compare_tsv_files(file1, file2)
    # # COMPLEX MW + GPT
    # file3 = "./results/thesaurus/merriam_webster/complex_simple/most_complex_words_merriam_webster.tsv"
    # file4 = "./results/llms/gpt/complex_simple_gpt/synonyms_complex.tsv"
    # compare_tsv_files(file3, file4)

    #MW + LLAMA
    # file3 = "./results/thesaurus/merriam_webster/complex_simple/simplest_words_merriam_webster.tsv"
    # file4 = "./results/llms/llama/complex_simple_llama/synonyms_simple.tsv"
    # compare_tsv_files(file3, file4)
    
    # file5 = "./results/thesaurus/merriam_webster/complex_simple/most_complex_words_merriam_webster.tsv"
    # file6 = "./results/llms/llama/complex_simple_llama/synonyms_complex.tsv"
    # compare_tsv_files(file5, file6)
    
    # WORDS + LLAMA
    file7 = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI/simplest_words_thesaurus.tsv"
    file8 = "./results/llms/llama/complex_simple_llama/synonyms_simple.tsv"
    compare_tsv_files(file7, file8)
    ##COMPLEX
    file1 = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI/most_complex_words_thesaurus.tsv"
    file2 = "./results/llms/llama/complex_simple_llama/synonyms_complex.tsv"
    compare_tsv_files(file1, file2)

    ############# DIACHRONIC ###########################

    
    
