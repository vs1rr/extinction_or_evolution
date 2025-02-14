import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class Benchmark:
    def __init__(self, thesaurus_file, llm_file):
        self.thesaurus_file = thesaurus_file
        self.llm_file = llm_file
        self.similarity_calculator = SimilarityCalculator()
        
    def load_data(self):
        """
        Loads the thesaurus and LLM-generated synonyms.
        """
        thesaurus_df = pd.read_csv(self.thesaurus_file, sep='\t')
        llm_df = pd.read_csv(self.llm_file, sep='\t')

        return thesaurus_df, llm_df
    
    def evaluate(self):
        """
        Computes benchmark scores for different similarity metrics.
        """
        thesaurus_df, llm_df = self.load_data()
        
        all_synonyms1 = []
        all_synonyms2 = []

        for word in thesaurus_df['original_word']:
            if word in llm_df['original_word'].values:
                synonyms1 = eval(thesaurus_df[thesaurus_df['original_word'] == word]['synonyms'].values[0])
                synonyms2 = eval(llm_df[llm_df['original_word'] == word]['synonyms'].values[0])
                all_synonyms1.extend(synonyms1)
                all_synonyms2.extend(synonyms2)

        metrics = self.similarity_calculator.compute_all_metrics(all_synonyms1, all_synonyms2)
        return metrics

    def generate_benchmark_report(self, save_path="benchmark_results"):
        """
        Runs the benchmark and generates a report with visualizations.
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Compute similarity metrics
        metrics = self.evaluate()
        
        # Save results to CSV
        results_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Score'])
        csv_path = os.path.join(save_path, "benchmark_scores.csv")
        results_df.to_csv(csv_path)
        print(f"Benchmark results saved to {csv_path}")

        # Plot the results
        self.plot_results(metrics, save_path)

    def plot_results(self, metrics, save_path):
        """
        Generates a bar plot of similarity scores for comparison.
        """
        labels = list(metrics.keys())
        values = list(metrics.values())

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values, color='skyblue', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Score")
        plt.title("Benchmark: LLM vs. Thesaurus Similarity Scores")
        plt.ylim(0, 1)

        # Save figure
        fig_path = os.path.join(save_path, "benchmark_plot.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Benchmark plot saved to {fig_path}")

# Example usage
if __name__ == "__main__":
    # Specify paths to thesaurus and LLM-generated synonyms
    thesaurus_file = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI/simplest_words_thesaurus.tsv"
    llm_file = "./results/llms/gpt/complex_simple_gpt/synonyms_simple.tsv"

    benchmark = Benchmark(thesaurus_file, llm_file)
    benchmark.generate_benchmark_report()
