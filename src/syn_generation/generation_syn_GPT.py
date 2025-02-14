import pandas as pd
import time
import os
import logging
from dotenv import load_dotenv
import openai

class SynonymGenerator:
    def __init__(self, output_dir):
        """
        Initialize the synonym generator with GPT API key and output directory.

        :param output_dir: Directory where the output files will be saved.
        """
        self.output_dir = output_dir
        
        # Load environment variables
        load_dotenv()

        # Load GPT API key from environment variables
        self.gpt_api_key = os.getenv("OPENAI_API_KEY")
        if not self.gpt_api_key:
            raise ValueError("GPT_API_KEY not found in environment variables.")

        openai.api_key = self.gpt_api_key

    def generate_synonyms_gpt(self, word):
        """
        Generate synonyms for a given word using the GPT model.

        :param word: The word for which synonyms are to be generated.
        :return: A list of synonyms.
        """
        try:
            # Prepare the messages for the chat-based model
            messages = [
                {"role": "system", "content": "You are a helpful assistant who provides synonyms for words."},
                {"role": "user", "content": f"Provide a plain, clean list of synonyms for the word '{word}'. Do not include any newlines, bullet points, special characters, or formatting. Just output the synonyms as a simple, comma-separated list of words without any extra spaces or symbols."}
            ]

            # Make the API call
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )

            # Extract the generated text
            generated_text = response.choices[0].message['content'].strip()

            # Parse the response
            synonyms = [syn.strip() for syn in generated_text.split(",") if syn.strip()]
            return synonyms
        except Exception as e:
            logging.error(f"Error generating synonyms with GPT for word '{word}': {e}")
            return []
        
    def process_file(self, input_file, output_file, max_rows=None):
        """
        Process an input file to generate synonyms for each word and save the output.
        :param input_file: Path to the input TSV file containing words and their scores.
        :param output_file: Path to save the output file with synonyms.
        :param max_rows: Maximum number of rows to process from the input file (default is 5).
        """
        # Load the input file
        data = pd.read_csv(input_file, sep="\t", header=None, names=["word", "score"])

        # If max_rows is not specified, process all rows
        if max_rows is None:
            max_rows = data.shape[0]
        
        data = data.head(max_rows)

        results = []
        for _, row in data.iterrows():
            word = row["word"]
            start_time = time.time()
            try:
                synonyms = self.generate_synonyms_gpt(word)
            except Exception as e:
                logging.error(f"Error generating synonyms for word '{word}': {e}")
                synonyms = []
            elapsed_time = time.time() - start_time
            results.append([word, synonyms, row["score"], elapsed_time])

        # Save the results to a new TSV file
        results_df = pd.DataFrame(results, columns=["original_word", "synonyms", "score", "time_elapsed"])
        results_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    ############### COMPLEX SIMPLE ###############

    # Directory setup
    input_dir = os.path.join(os.getcwd(), "corpus/truncation_result/complex_simple")
    output_dir = os.path.join(os.getcwd(), "results/llms/gpt/complex_simple_gpt")

    # Directory setup
    simplest_input = os.path.join(input_dir, "simplest_words.tsv")
    complex_input = os.path.join(input_dir, "most_complex_words.tsv")
    simplest_output_dir = os.path.join(output_dir, "synonyms_simple.tsv")
    complex_output_dir = os.path.join(output_dir, "synonyms_complex.tsv")


    # ############### DIACHRONIC ###############
    # # Directory setup
    # input_dir = os.path.join(os.getcwd(), "corpus/truncation_result/diachronic")
    # output_dir = os.path.join(os.getcwd(), "results/llms/gpt/diachronic_gpt")

    # # ADJ 
    # simplest_input_adj = os.path.join(input_dir, "adjs_high_keyness_words.tsv")
    # complex_input_adj = os.path.join(input_dir, "adjs_low_keyness_words.tsv")
    # simplest_output_dir_adj = os.path.join(output_dir, "adjs_high_keyness_words.tsv")
    # complex_output_dir_adj = os.path.join(output_dir, "adjs_low_keyness_words.tsv")

    # # ADV 
    # simplest_input_adv = os.path.join(input_dir, "advs_high_keyness_words.tsv")
    # complex_input_adv = os.path.join(input_dir, "advs_low_keyness_words.tsv")
    # simplest_output_dir_adv = os.path.join(output_dir, "advs_high_keyness_words.tsv")
    # complex_output_dir_adv = os.path.join(output_dir, "advs_low_keyness_words.tsv")

    # # NOUNS 
    # simplest_input_nouns = os.path.join(input_dir, "nouns_high_keyness_words.tsv")
    # complex_input_nouns = os.path.join(input_dir, "nouns_low_keyness_words.tsv")
    # simplest_output_dir_nouns = os.path.join(output_dir, "nouns_high_keyness_words.tsv")
    # complex_output_dir_nouns = os.path.join(output_dir, "nouns_low_keyness_words.tsv")

    # #VEBRS 
    # simplest_input_verbs = os.path.join(input_dir, "verbs_high_keyness_words.tsv")
    # complex_input_verbs = os.path.join(input_dir, "verbs_low_keyness_words.tsv")
    # simplest_output_dir_verbs = os.path.join(output_dir, "verbs_high_keyness_words.tsv")
    # complex_output_dir_verbs = os.path.join(output_dir, "verbs_low_keyness_words.tsv")
    
    # Initialize SynonymGenerator
    
    try:
        generator = SynonymGenerator(output_dir=output_dir)
        #### COMPLEX SIMPLE
        logging.info("Processing simplest words...")
        generator.process_file(simplest_input, simplest_output_dir)
        logging.info("Processing complex words...")
        generator.process_file(complex_input, complex_output_dir)
        logging.info("Processing simplest words...")

        # Process files and generate synonyms
        # logging.info("Processing simplest words...")
        # generator.process_file(simplest_input_adj, simplest_output_dir_adj)
        # logging.info("Processing complex words...")
        # generator.process_file(complex_input_adj, complex_output_dir_adj)
        # logging.info("Processing simplest words...")
        # generator.process_file(simplest_input_nouns, simplest_output_dir_nouns)
        # logging.info("Processing complex words...")
        # generator.process_file(complex_input_nouns, complex_output_dir_nouns)
        # logging.info("Processing simplest words...")
        # generator.process_file(simplest_input_adv, simplest_output_dir_adv)
        # logging.info("Processing complex words...")
        # generator.process_file(complex_input_adv, complex_output_dir_adv)
        # logging.info("Processing simplest words...")
        # generator.process_file(simplest_input_verbs, simplest_output_dir_verbs)
        # logging.info("Processing complex words...")
        # generator.process_file(complex_input_verbs, complex_output_dir_verbs)
        # logging.info("Processing simplest words...")
    except Exception as e:
        logging.error(f"Error: {e}")
