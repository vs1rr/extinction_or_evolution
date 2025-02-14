import pandas as pd
import time
import os
import logging
from llama_cpp import Llama
from dotenv import load_dotenv

class SynonymGenerator:
    def __init__(self, output_dir, llama_model_path):
        """
        Initialize the synonym generator with Llama model path and output directory.

        :param output_dir: Directory where the output files will be saved.
        :param llama_model_path: Path to the Llama model file for generating synonyms.
        """
        self.output_dir = output_dir
        self.llama_model_path = llama_model_path

    def generate_synonyms_llama(self, word):
        """
        Generate synonyms for a given word using the Llama model.

        :param word: The word for which synonyms are to be generated.
        :return: A list of synonyms.
        """
        try:
            llm = Llama(model_path=self.llama_model_path)

            # Prompt to generate synonyms
            prompt = f"Donnez une liste simple et propre de synonymes pour le mot '{word}'. N'incluez pas de sauts de ligne, de points, de caractères spéciaux ou de formatage. Donnez simplement les synonymes sous forme de liste de mots séparés par des virgules sans espaces ou symboles supplémentaires."
            output = llm(prompt, max_tokens=50, temperature=0.7, top_k=50, top_p=0.9)
            generated_text = output["choices"][0]["text"].strip()

            # Parse the response
            synonyms = [syn.strip() for syn in generated_text.split(",") if syn.strip()]
            return synonyms
        except Exception as e:
            logging.error(f"Error generating synonyms with Llama for word '{word}': {e}")
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
                synonyms = self.generate_synonyms_llama(word)
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

    ################### LLAMA LOCAL ###################
    llama_model_path = "./chat-gpt_thesaurus/meta-llama/Meta-Llama-3.1-8B-Instruct/original/meta-llama-Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf"


    ################### COMPLEX SIMPLE ###################
    # input_dir = os.path.join(os.getcwd(), "corpus/truncation_result/complex_simple")
    # output_dir = os.path.join(os.getcwd(), "results/llms/llama/complex_simple_llama")

    # simplest_input = os.path.join(input_dir, "simplest_words.tsv")
    # complex_input = os.path.join(input_dir, "most_complex_words.tsv")
    # simplest_output_dir = os.path.join(output_dir, "synonyms_simple.tsv")
    # complex_output_dir = os.path.join(output_dir, "synonyms_complex.tsv")

    ################### DIACHRONIC ###################

    # Directory setup
    input_dir = os.path.join(os.getcwd(), "corpus/truncation_result/diachronic")
    output_dir = os.path.join(os.getcwd(), "results/llms/llama/diachronic_llama")

    # ADJ 
    simplest_input_adj = os.path.join(input_dir, "adjs_high_keyness_words.tsv")
    complex_input_adj = os.path.join(input_dir, "adjs_low_keyness_words.tsv")
    simplest_output_dir_adj = os.path.join(output_dir, "adjs_high_keyness_words.tsv")
    complex_output_dir_adj = os.path.join(output_dir, "adjs_low_keyness_words.tsv")

    # ADV 
    simplest_input_adv = os.path.join(input_dir, "advs_high_keyness_words.tsv")
    complex_input_adv = os.path.join(input_dir, "advs_low_keyness_words.tsv")
    simplest_output_dir_adv = os.path.join(output_dir, "advs_high_keyness_words.tsv")
    complex_output_dir_adv = os.path.join(output_dir, "advs_low_keyness_words.tsv")

    # NOUNS 
    simplest_input_nouns = os.path.join(input_dir, "nouns_high_keyness_words.tsv")
    complex_input_nouns = os.path.join(input_dir, "nouns_low_keyness_words.tsv")
    simplest_output_dir_nouns = os.path.join(output_dir, "nouns_high_keyness_words.tsv")
    complex_output_dir_nouns = os.path.join(output_dir, "nouns_low_keyness_words.tsv")

    #VEBRS 
    simplest_input_verbs = os.path.join(input_dir, "verbs_high_keyness_words.tsv")
    complex_input_verbs = os.path.join(input_dir, "verbs_low_keyness_words.tsv")
    simplest_output_dir_verbs = os.path.join(output_dir, "verbs_high_keyness_words.tsv")
    complex_output_dir_verbs = os.path.join(output_dir, "verbs_low_keyness_words.tsv")


    # Initialize SynonymGenerator
    generator = SynonymGenerator(output_dir=output_dir, llama_model_path=llama_model_path)

    # Process files and generate synonyms
    try:
        #################### COMPLEX SIMPLE ########################
        # logging.info("Processing simplest words...")
        # generator.process_file(simplest_input, simplest_output_dir)
        # logging.info("Processing complex words...")
        # generator.process_file(complex_input, complex_output_dir)

        #################### DIACHRONIC ########################
        logging.info("Processing simplest words...")
        generator.process_file(simplest_input_adj, simplest_output_dir_adj)
        logging.info("Processing complex words...")
        generator.process_file(complex_input_adj, complex_output_dir_adj)
        logging.info("Processing simplest words...")
        generator.process_file(simplest_input_nouns, simplest_output_dir_nouns)
        logging.info("Processing complex words...")
        generator.process_file(complex_input_nouns, complex_output_dir_nouns)
        logging.info("Processing simplest words...")
        generator.process_file(simplest_input_adv, simplest_output_dir_adv)
        logging.info("Processing complex words...")
        generator.process_file(complex_input_adv, complex_output_dir_adv)
        logging.info("Processing simplest words...")
        generator.process_file(simplest_input_verbs, simplest_output_dir_verbs)
        logging.info("Processing complex words...")
        generator.process_file(complex_input_verbs, complex_output_dir_verbs)
        logging.info("Processing simplest words...")
    except Exception as e:
        logging.error(f"Error processing files: {e}")

