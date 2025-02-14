import os
import pandas as pd
import time
import requests
import logging
from dotenv import load_dotenv

class MerriamWebsterSynonymGenerator:
    def __init__(self, output_dir, api_key):
        """
        Initialize the synonym generator with Merriam-Webster API key and output directory.

        :param output_dir: Directory where the output files will be saved.
        :param api_key: Merriam-Webster API key for thesaurus integration.
        """
        self.output_dir = output_dir
        self.api_key = api_key
        self.api_url = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json/{word}?key={api_key}"
        
        # Setup logging
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "synonym_generation.log"),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def generate_synonyms(self, word):
        """
        Generate synonyms for a given word using the Merriam-Webster Thesaurus API.

        :param word: The word for which synonyms are to be generated.
        :return: A list of synonyms or a "No synonyms found" flag.
        """
        url = self.api_url.format(word=word, api_key=self.api_key)
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                data = response.json()
                # Check if data contains valid synonym information
                if isinstance(data, list) and len(data) > 0 and "meta" in data[0]:
                    return data[0]["meta"]["syns"][0]  # List of synonyms
                
                # Log and return "No synonyms found" when no synonyms are available
                logging.info(f"No synonyms found for: {word}")
                return ["No synonyms found"]
            except ValueError:
                logging.error(f"Invalid JSON response for word '{word}': {response.text}")
                return ["Invalid response format"]
        else:
            logging.error(f"API request failed for '{word}' - Status code {response.status_code}")
            return ["API request failed"]

    def process_file(self, input_file, output_file):
        """
        Process an input file to generate synonyms for each word and save the output.

        :param input_file: Path to the input TSV file containing words and their scores.
        :param output_file: Path to save the output file with synonyms.
        """
        # Load the input file
        data = pd.read_csv(input_file, sep="\t", header=None, names=["word", "score"])
        
        results = []
        missing_synonyms_count = 0  # Counter for words without synonyms

        for _, row in data.iterrows():
            word = row["word"]
            start_time = time.time()
            synonyms = self.generate_synonyms(word)
            elapsed_time = time.time() - start_time

            # Track the number of missing synonyms
            if synonyms == ["No synonyms found"]:
                missing_synonyms_count += 1

            results.append([word, synonyms, row["score"], elapsed_time])

        # Log the percentage of missing synonyms
        missing_percentage = (missing_synonyms_count / len(data)) * 100
        logging.info(f"Processed {len(data)} words. Missing synonyms for {missing_synonyms_count} words ({missing_percentage:.2f}%).")

        # Save the results to a new TSV file
        results_df = pd.DataFrame(results, columns=["original_word", "synonyms", "score", "time_elapsed"])
        results_df.to_csv(output_file, sep="\t", index=False)

    def process_directory(self, input_dir, file_suffix="_merriam_webster.tsv"):
        """
        Process all TSV files in a directory and save the output for each file.

        :param input_dir: Path to the directory containing input TSV files.
        :param file_suffix: Suffix to append to output files.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for file_name in os.listdir(input_dir):
            if file_name.endswith(".tsv"):
                input_file = os.path.join(input_dir, file_name)
                output_file = os.path.join(self.output_dir, file_name.replace(".tsv", file_suffix))
                logging.info(f"Processing file: {input_file}")
                self.process_file(input_file, output_file)
                logging.info(f"Output saved to: {output_file}")

if __name__ == "__main__":
    # Load environment variables from .env
    load_dotenv()

    # Fetch the API key from .env
    mw_api_key = os.getenv("MW_API_KEY")
    if not mw_api_key:
        raise ValueError("Merriam-Webster API key not found in .env file.")

    # Directories for input and output
    input_dir = "./corpus/truncation_result/complex_simple"
    output_dir = "./results/thesaurus/merriam_webster/complex_simple"

    # Initialize Merriam-Webster Synonym Generator
    thesaurus_generator = MerriamWebsterSynonymGenerator(output_dir, mw_api_key)
    logging.info("Initialized Merriam-Webster Synonym Generator.")

    # Process all files in the directory
    thesaurus_generator.process_directory(input_dir)
    logging.info("All files processed.")
