import os
import pandas as pd
import time
import requests

"""
BE CAREFUL ABOUT THE CHANGEHERE COMMENTS!
"""

class OpenThesaurusGenerator:
    def __init__(self, output_dir):
        """
        Initialize the synonym generator with an open-access thesaurus API and output directory.

        :param output_dir: Directory where the output files will be saved.
        """
        #CHANGEHERE
        self.output_dir = output_dir
        # self.api_url = "https://api.datamuse.com/words?ml={word}"  # API for ENGLISH
        self.api_url = "https://api.datamuse.com/words?ml={word}&v=fr" # API for FRENCH

    def generate_synonyms(self, word):
        """
        Generate synonyms for a given word using the Datamuse API.

        :param word: The word for which synonyms are to be generated.
        :return: A list of synonyms.
        """
        url = self.api_url.format(word=word)
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            # Extract the words from the response
            return [entry["word"] for entry in data]
        return ["No synonyms found"]  # Default if no synonyms are available

    def process_file(self, input_file, output_file):
        """
        Process an input file to generate synonyms for each word and save the output.

        :param input_file: Path to the input TSV file containing words and their scores.
        :param output_file: Path to save the output file with synonyms.
        """
        # Load the input file
        data = pd.read_csv(input_file, sep="\t", header=None, names=["word", "score"])
        
        results = []
        for _, row in data.iterrows():
            word = row["word"]
            start_time = time.time()
            synonyms = self.generate_synonyms(word)
            elapsed_time = time.time() - start_time
            results.append([word, synonyms, row["score"], elapsed_time])

        # Save the results to a new TSV file
        results_df = pd.DataFrame(results, columns=["original_word", "synonyms", "score", "time_elapsed"])
        results_df.to_csv(output_file, sep="\t", index=False)

    def process_directory(self, input_dir, file_suffix="_thesaurus.tsv"):
        """
        Process all TSV files in a directory and save the output for each file.

        :param input_dir: Path to the directory containing input TSV files.
        :param file_suffix: Suffix to append to output files.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Process each file in the directory
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".tsv"):
                input_file = os.path.join(input_dir, file_name)
                output_file = os.path.join(self.output_dir, file_name.replace(".tsv", file_suffix))
                print(f"Processing file: {input_file}")
                self.process_file(input_file, output_file)
                print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Directories for input and output
    
    #CHANGEHERE
    input_dir = "./corpus/truncation_result/diachronic"
    #input_dir = "./corpus/truncation_result/complex_simple/" 

    output_dir = "./results/thesaurus/wordsAPI/diachronic_wordsAPI"
    #output_dir = "./results/thesaurus/wordsAPI/complex_simple_wordsAPI"

    # Initialize OpenThesaurus generator
    thesaurus_generator = OpenThesaurusGenerator(output_dir)
    print("Initialized!")

    # Process all files in the directory
    thesaurus_generator.process_directory(input_dir)
    print("All files processed!")