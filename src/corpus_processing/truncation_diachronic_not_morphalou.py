import pandas as pd
import os
import json
import spacy
from collections import defaultdict


class JSONWordProcessor:
    def __init__(self, input_file, output_dir, custom_stopwords=None):
        """
        Initialize the processor with the input file path, output directory, and custom stopwords.

        :param input_file: Path to the input JSON file.
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

    def load_json(self):
        """
        Load the JSON data into a pandas DataFrame.
        """
        # Load JSON data from the file
        with open(self.input_file, 'r') as file:
            input_json = json.load(file)
        
        # Convert JSON to a DataFrame
        self.data = pd.DataFrame.from_dict(input_json, orient='index')
        self.data.reset_index(inplace=True)
        self.data.rename(columns={'index': 'word'}, inplace=True)

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
            raise ValueError("Data not loaded. Call load_json() first.")

        # Identify and filter out proper nouns
        non_proper_nouns = []
        for _, row in self.data.iterrows():
            word = row['word']
            k_value = row['k']

            # Use SpaCy to get the part-of-speech tag
            doc = self.nlp(word)
            if doc[0].pos_ != "PROPN":  # Keep only non-proper nouns
                non_proper_nouns.append({'word': word, 'k': k_value})

        # Update the DataFrame with non-proper-noun words
        self.data = pd.DataFrame(non_proper_nouns)

    def calculate_percentile_threshold(self):
        """
        Calculate lower and upper thresholds using percentiles.
        """
        lower_threshold = self.data['k'].quantile(0.10)
        upper_threshold = self.data['k'].quantile(0.90)
        return lower_threshold, upper_threshold

    def select_words(self):
        """
        Dynamically select words based on automatically calculated thresholds.

        :return: Two DataFrames, one for the words with low keyness and one for high keyness.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_json() first.")

        # Calculate automatic thresholds
        lower_threshold, upper_threshold = self.calculate_percentile_threshold()

        # Filter words within thresholds
        low_keyness = self.data[self.data['k'] <= lower_threshold]
        high_keyness = self.data[self.data['k'] >= upper_threshold]

        return low_keyness, high_keyness

    def save_to_files(self, low_keyness, high_keyness, file_prefix):
        """
        Save the low and high keyness words to separate files.

        :param low_keyness: DataFrame containing the words with low keyness.
        :param high_keyness: DataFrame containing the words with high keyness.
        :param file_prefix: Prefix to use for output filenames.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        low_keyness.to_csv(os.path.join(self.output_dir, f'{file_prefix}_low_keyness_words.tsv'), sep='\t', index=False, header=False)
        high_keyness.to_csv(os.path.join(self.output_dir, f'{file_prefix}_high_keyness_words.tsv'), sep='\t', index=False, header=False)

    def process(self):
        """
        Perform the entire process of loading, aggregating, selecting, and saving words.
        """
        self.load_json()
        self.remove_proper_nouns()
        low_keyness, high_keyness = self.select_words()
        file_prefix = os.path.splitext(os.path.basename(self.input_file))[0]
        self.save_to_files(low_keyness, high_keyness, file_prefix)

if __name__ == "__main__":
    # Input directory containing JSON files
    input_dir = './corpus/filtered_corpus'
    output_dir = './corpus/truncation_result/diachronic'

    # Add your custom stopwords here
    custom_stopwords = ['jul', 'feb', 'xii', 'hmmm', 'vii', 'iii', 'hmm', 'jan', 'hrs', 'sec', 'www', 'dec', 'amy', 'hehe', 'hey', 'yep', 'wow', 'fri', 'apr']

    # Get a list of all JSON files in the input directory
    json_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.json')]

    # Process each JSON file
    for json_file in json_files:
        processor = JSONWordProcessor(json_file, output_dir, custom_stopwords=custom_stopwords)
        processor.process()
        print(f"Processing complete for {json_file}. Results saved.")
