import pandas as pd
import os
import spacy
from collections import defaultdict

class WordComplexityProcessor:
    def __init__(self, input_file, output_dir, custom_stopwords=None):
        """
        Initialize the processor with the input file path, output directory, and custom stopwords.

        :param input_file: Path to the input TSV file.
        :param output_dir: Directory where the output files will be saved.
        :param custom_stopwords: List of additional stopwords to exclude.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.data = None

        # Load SpaCy's language model
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(self.nlp.Defaults.stop_words)

        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update([word.lower() for word in custom_stopwords])

    def load_file(self):
        """
        Load the TSV file into a pandas DataFrame and preprocess it.
        """
        self.data = pd.read_csv(self.input_file, sep='\t', header=None, names=['word', 'score'])

        # Lowercase all words
        self.data['word'] = self.data['word'].str.lower()

        # Remove stopwords
        self.data = self.data[~self.data['word'].isin(self.stopwords)]

        # Remove single-character words
        self.data = self.data[self.data['word'].str.len() > 2]

    def remove_proper_nouns(self):
        """
        Remove proper nouns (e.g., names, places) based on part-of-speech tagging.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_file() first.")

        # Identify and filter out proper nouns
        non_proper_nouns = []
        for _, row in self.data.iterrows():
            word = row['word']
            score = row['score']

            # Use SpaCy to get the part-of-speech tag
            doc = self.nlp(word)
            if doc[0].pos_ != "PROPN":  # Keep only non-proper nouns
                non_proper_nouns.append({'word': word, 'score': score})

        # Update the DataFrame with non-proper-noun words
        self.data = pd.DataFrame(non_proper_nouns)

    def calculate_percentile_threshold(self):
        """
        Calculate lower and upper thresholds using percentiles.
        """
        lower_threshold = self.data['score'].quantile(0.10)
        upper_threshold = self.data['score'].quantile(0.90)
        return lower_threshold, upper_threshold

    def lemmatize_and_aggregate(self):
        """
        Lemmatize words and aggregate scores for the same lemma.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_file() first.")

        # Create a dictionary to store aggregated scores
        lemma_scores = defaultdict(list)

        # Lemmatize each word and store its score
        for _, row in self.data.iterrows():
            word = row['word']
            score = row['score']

            # Lemmatize the word
            doc = self.nlp(word)
            lemma = doc[0].lemma_

            # Add the score to the lemma
            lemma_scores[lemma].append(score)

        # Aggregate scores for each lemma
        aggregated_data = []
        for lemma, scores in lemma_scores.items():
            # Take the mean score
            aggregated_score = sum(scores) / len(scores)
            aggregated_data.append({'lemma': lemma, 'score': aggregated_score})

        # Convert to DataFrame
        self.data = pd.DataFrame(aggregated_data)

    def select_words(self):
        """
        Dynamically select words based on automatically calculated thresholds.

        :return: Two DataFrames, one for the simplest words and one for the most complex.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_file() first.")

        # Calculate automatic thresholds
        lower_threshold, upper_threshold = self.calculate_percentile_threshold()

        # Filter words within thresholds
        simplest_words = self.data[self.data['score'] <= lower_threshold]
        most_complex_words = self.data[self.data['score'] >= upper_threshold]

        return simplest_words, most_complex_words

    def save_to_files(self, simplest_words, most_complex_words):
        """
        Save the simplest and most complex words to separate files.

        :param simplest_words: DataFrame containing the simplest words.
        :param most_complex_words: DataFrame containing the most complex words.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        simplest_words.to_csv(os.path.join(self.output_dir, 'simplest_words.tsv'), sep='\t', index=False, header=False)
        most_complex_words.to_csv(os.path.join(self.output_dir, 'most_complex_words.tsv'), sep='\t', index=False, header=False)

    def process(self):
        """
        Perform the entire process of loading, lemmatizing, aggregating, selecting, and saving words.
        """
        self.load_file()
        self.remove_proper_nouns()
        self.lemmatize_and_aggregate()
        simplest_words, most_complex_words = self.select_words()
        self.save_to_files(simplest_words, most_complex_words)

if __name__ == "__main__": 
    # Example usage
    input_file = './corpus/old_words/'
    output_dir = './corpus/truncation_result/complex_simple'

    # Add your custom stopwords here
    custom_stopwords = ['jul','feb','xii','hmmm','vii','iii','hmm','jan','hrs','sec','www','dec','amy','hehe','hey','yep','wow','fri','apr']

    processor = WordComplexityProcessor(input_file, output_dir, custom_stopwords=custom_stopwords)
    processor.process()
    print("Processing complete.")
