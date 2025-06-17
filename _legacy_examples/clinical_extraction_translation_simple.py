import os
import requests # Added
import re # Added
import hashlib # Added for generating case IDs
from typing import Optional # Added
import argparse # Added
from datasets import load_dataset, Dataset, IterableDataset # Added Dataset, IterableDataset
from pydantic import BaseModel, Field
from chainette import Step, Chain, Apply, register_engine, SamplingParams
from transformers import AutoTokenizer

MAX_LEN = 8192 * 4  # Set a maximum length for the text, adjusted for simplicity

# Schemas
class Article(BaseModel):
    text: str = Field(..., description="The full text of the scientific article")
    pmid: Optional[str] = Field(default=None, description="PubMed ID of the article, if available") # Corrected typing
    authors: list[str] = Field(default_factory=list, description="List of authors")
    license: str = Field(default="", description="License information")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms")

class ExtractedPair(BaseModel):
    case_id: str = Field(..., description="Unique identifier for linking with translations")
    clinical_case: str = Field(..., description="The extracted clinical case passage")
    insights: str = Field(..., description="The extracted insights passage corresponding to the clinical case")
    pmid: Optional[str] = Field(default=None, description="PubMed ID of the article, if available") # Optional field for PMID
    authors: list[str] = Field(default_factory=list, description="List of authors of the article")
    license: str = Field(default="", description="License information of the article")
    mesh_terms: list[str] = Field(default_factory=list, description="MeSH terms associated with the article")

class ExtractedPairList(BaseModel):
    extracted_pairs: list[ExtractedPair] = Field(..., description="A list of clinical case and insights pairs")

class TranslatedExtractedPairFrench(BaseModel):
    case_id: str = Field(..., description="Unique identifier linking to original English case")
    clinical_case_en: str = Field(..., description="The original English clinical case passage")
    insights_en: str = Field(..., description="The original English insights passage")
    clinical_case_fr: str = Field(..., description="The translated French clinical case passage")
    insights_fr: str = Field(..., description="The translated French insights passage")
    pmid: Optional[str] = Field(default=None, description="PubMed ID of the article, if available") # Optional field for PMID
    authors: list[str] = Field(default_factory=list, description="List of authors of the article")
    license: str = Field(default="", description="License information of the article")
    mesh_terms: list[str] = Field(default_factory=list, description="Translated MeSH terms associated with the article")

# Engine Registration
register_engine(
    name="llama3_3_70b_instruct",
    model="google/medgemma-27b-text-it", # Using model from clinical_insights_qa.py
    dtype="bfloat16",
    gpu_memory_utilization=0.9, # Adjusted from 0.95 for potentially wider compatibility
    enable_reasoning=False,
    max_model_len=MAX_LEN, # Reduced from 32768 for a simpler example
    tensor_parallel_size=1, # Reduced from 4
)

# Prompt Templates
EXTRACT_SYSTEM_PROMPT = """
You are given a scientific article as plain text. Your task is to extract every "Clinical Case" passage together with its matching "Insights" passage, ensuring each Insights refers only to that specific Clinical Case.

Instructions:
1.  Identify Pairs: Locate all passages describing a single patient's case and the corresponding passages that interpret or reflect exclusively on that specific case.
2.  Extract Verbatim: Copy the exact text for each "clinical_case" and its matching "insights".
3.  Generate Unique ID: For each pair, create a unique case_id using this format: "{pmid}_{case_number}" where case_number starts from 1 (e.g., "12345678_1", "12345678_2"). If no PMID, use "UNKNOWN_{hash}" where hash is first 8 characters of SHA-256 of clinical_case text.
4.  Return Empty If None Found: If the article contains no clear clinical cases or insights pairs, return an empty list of extracted pairs.

Definitions:
- Clinical Case: A detailed description of a single patient's medical condition, symptoms, and history etc.
- Insights: Interpretations, analyses, or conclusions drawn from the clinical case.

Also include the following metadata for each extracted pair:
- pmid: The PubMed ID of the article, if available.
- authors: A list of authors of the article.
- license: The license information of the article.
- mesh_terms: A list of MeSH terms associated with the article. If there is no mesh_terms, return "N/A".
"""

TRANSLATE_FRENCH_SYSTEM_PROMPT = """\
You are a professional translator. Your task is to translate the provided clinical case and its insights into French.
Ensure both the 'clinical_case' and 'insights' fields are accurately and fluently translated into French.
Translate also the metadata fields: mesh_terms. If there is no mesh_terms, return "N/A".
Keep authors and license information in English, as they are not to be translated.
"""

TRANSLATE_FRENCH_USER_PROMPT = """\
Original Clinical Case:
{{flatten_extracted_pairs.clinical_case}}

Original Insights:
{{flatten_extracted_pairs.insights}}

case_id:
{{flatten_extracted_pairs.case_id}}

pmid:
{{flatten_extracted_pairs.pmid}}

authors:
{{flatten_extracted_pairs.authors}}

license:
{{flatten_extracted_pairs.license}}

mesh_terms:
{{flatten_extracted_pairs.mesh_terms}}
"""

# Custom Flattening Function
def flatten_extracted_pairs(pair_list_container: ExtractedPairList, on_field: str = "extracted_pairs") -> list[ExtractedPair]:
    individual_pairs = []
    if hasattr(pair_list_container, on_field):
        items = getattr(pair_list_container, on_field)
        if isinstance(items, list):
            for i, item in enumerate(items):
                # Ensure case_id is set if missing
                if not hasattr(item, 'case_id') or not item.case_id:
                    pmid = getattr(item, 'pmid', None)
                    clinical_case = getattr(item, 'clinical_case', '')
                    if pmid:
                        item.case_id = f"{pmid}_{i+1}"
                    else:
                        # Generate hash from clinical case text
                        case_hash = hashlib.sha256(clinical_case.encode()).hexdigest()[:8]
                        item.case_id = f"UNKNOWN_{case_hash}"
                individual_pairs.append(item)
    return individual_pairs

# Added function to fetch PubMed metadata (place before Steps or where appropriate)
def fetch_pubmed_metadata(pmid: Optional[str], email: str = "user@example.com") -> tuple[list[str], str, list[str]]:
    """Fetch authors, license, and MeSH terms from PubMed and PMC APIs using E-utilities."""
    authors = []
    license_info = "Copyright restrictions may apply" # Default license
    mesh_terms = []
    
    if not pmid:
        return authors, license_info, mesh_terms

    try:
        # First, fetch from PubMed to get authors, MeSH terms, and PMC ID
        pubmed_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&email={email}"
        response = requests.get(pubmed_url, timeout=15)
        response.raise_for_status()
        xml_data = response.text

        # Extract authors: <contrib contrib-type="author"> <name> <surname>X</surname> <given-names>Y</given-names> </name> </contrib>
        # or <AuthorList><Author><LastName>X</LastName><ForeName>Y</ForeName></Author></AuthorList>
        author_contrib_matches = re.findall(r"<contrib contrib-type=\"author\"[^>]*>([\s\S]*?)</contrib>", xml_data, re.DOTALL)
        if author_contrib_matches:
            for contrib_xml in author_contrib_matches:
                name_match = re.search(r"<name>([\s\S]*?)</name>", contrib_xml)
                if name_match:
                    name_xml = name_match.group(1)
                    surname_match = re.search(r"<surname>([^<]+)</surname>", name_xml)
                    given_names_match = re.search(r"<given-names>([^<]+)</given-names>", name_xml)
                    if surname_match and given_names_match:
                        authors.append(f"{given_names_match.group(1).strip()} {surname_match.group(1).strip()}")
                    elif surname_match: # Fallback if only surname is found
                        authors.append(surname_match.group(1).strip())
        else: # Fallback to <AuthorList>
            author_list_match = re.search(r"<AuthorList[^>]*>([\s\S]*?)</AuthorList>", xml_data)
            if author_list_match:
                author_matches = re.findall(r"<Author[^>]*>([\s\S]*?)</Author>", author_list_match.group(1))
                for author_xml in author_matches:
                    lastname_match = re.search(r"<LastName>([^<]+)</LastName>", author_xml)
                    forename_match = re.search(r"<ForeName>([^<]+)</ForeName>", author_xml)
                    if lastname_match and forename_match:
                        authors.append(f"{forename_match.group(1).strip()} {lastname_match.group(1).strip()}")
                    elif lastname_match:
                         authors.append(lastname_match.group(1).strip())


        # Extract MeSH terms: <MeshHeadingList><MeshHeading><DescriptorName>...</DescriptorName></MeshHeading></MeshHeadingList>
        mesh_list_match = re.search(r"<MeshHeadingList>([\s\S]*?)</MeshHeadingList>", xml_data)
        if mesh_list_match:
            mesh_matches = re.findall(r"<DescriptorName[^>]*>([^<]+)</DescriptorName>", mesh_list_match.group(1))
            mesh_terms = [term.strip() for term in mesh_matches]

        # Extract PMC ID for license lookup: <ArticleId IdType="pmc">PMC88923</ArticleId>
        pmc_id = None
        pmc_match = re.search(r'<ArticleId IdType="pmc">([^<]+)</ArticleId>', xml_data)
        if pmc_match:
            pmc_id = pmc_match.group(1).strip()
            print(f"  Found PMC ID: {pmc_id}")
        
        # If we have a PMC ID, fetch license information from PMC database
        if pmc_id:
            try:
                pmc_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmc_id}&retmode=xml&email={email}"
                pmc_response = requests.get(pmc_url, timeout=15)
                pmc_response.raise_for_status()
                pmc_xml = pmc_response.text
                
                # Extract license from <permissions> section
                permissions_match = re.search(r"<permissions>([\s\S]*?)</permissions>", pmc_xml, re.DOTALL)
                if permissions_match:
                    permissions_content = permissions_match.group(1)
                    
                    # Look for license information within permissions
                    license_tag_match = re.search(r"<license[^>]*>([\s\S]*?)</license>", permissions_content, re.DOTALL)
                    if license_tag_match:
                        license_full_content = license_tag_match.group(0)  # Full license tag
                        license_inner_content = license_tag_match.group(1)  # Content inside license tag
                        
                        # Extract href from any xlink:href attribute in the license tag
                        href_match = re.search(r'xlink:href="([^"]+)"', license_full_content)
                        href = href_match.group(1).strip() if href_match else ""
                        
                        # Extract license type
                        license_type_match = re.search(r'license-type="([^"]+)"', license_full_content)
                        license_type = license_type_match.group(1).strip() if license_type_match else ""
                        
                        # Extract text content from <license-p> tags (PMC format)
                        text_content = ""
                        license_p_match = re.search(r"<license-p>([\s\S]*?)</license-p>", license_inner_content, re.DOTALL)
                        if license_p_match:
                            text_content = re.sub(r'<[^>]+>', '', license_p_match.group(1).strip()) # Clean HTML tags
                            text_content = " ".join(text_content.split()) # Normalize whitespace
                        
                        # Build license info
                        if href:
                            license_info = href
                            if license_type:
                                license_info += f" ({license_type})"
                            if text_content:
                                # Take first sentence or a short summary
                                first_sentence = text_content.split('.')[0]
                                if len(first_sentence) < 150:  # Only add if reasonably short
                                    license_info += f" - {first_sentence}"
                        elif license_type:
                            license_info = f"License type: {license_type}"
                            if text_content:
                                first_sentence = text_content.split('.')[0]
                                if len(first_sentence) < 150:
                                    license_info += f" - {first_sentence}"
                        elif text_content:
                            first_sentence = text_content.split('.')[0]
                            license_info = first_sentence if len(first_sentence) < 150 else "License terms available in article"
                    
                    # Fallback to copyright statement if no license tag found
                    elif re.search(r"<copyright-statement>", permissions_content):
                        copyright_match = re.search(r"<copyright-statement>([\s\S]*?)</copyright-statement>", permissions_content, re.DOTALL)
                        if copyright_match:
                            license_info = copyright_match.group(1).strip()
                            license_info = re.sub(r'<[^>]+>', '', license_info) # Clean HTML
                            license_info = " ".join(license_info.split()) # Normalize whitespace
                            # Decode HTML entities
                            license_info = license_info.replace('&#169;', '©').replace('&copy;', '©')
                
            except requests.exceptions.RequestException as e:
                print(f"Warning: Error fetching PMC data for {pmc_id}: {e}")
            except Exception as e:
                print(f"Warning: Error parsing PMC data for {pmc_id}: {e}")

        # Extract license information: <license xlink:href="URL"><p>text</p></license> or <copyright-statement>
        license_tag_match = re.search(r"<license[^>]*xlink:href=\"([^\"]+)\"[^>]*>([\s\S]*?)</license>", xml_data, re.DOTALL)
        if license_tag_match:
            href = license_tag_match.group(1).strip()
            text_content = ""
            p_match = re.search(r"<p>([\s\S]*?)</p>", license_tag_match.group(2), re.DOTALL)
            if p_match:
                text_content = re.sub(r'<[^>]+>', '', p_match.group(1).strip()) # Clean HTML tags
                text_content = " ".join(text_content.split()) # Normalize whitespace
            
            license_info = f"{href}"
            if text_content:
                # Take first sentence or a short summary from the <p> tag
                first_sentence = text_content.split('.')[0]
                license_info += f" ({first_sentence})"

        elif "<?properties open_access?>" in xml_data or "<license license-type=\"open-access\">" in xml_data:
             license_info = "Open Access (details likely in article)"
        else:
            copyright_match = re.search(r"<copyright-statement>([\s\S]*?)</copyright-statement>", xml_data, re.DOTALL)
            if copyright_match:
                license_info = copyright_match.group(1).strip()
                license_info = re.sub(r'<[^>]+>', '', license_info) # Clean HTML
                license_info = " ".join(license_info.split()) # Normalize whitespace
        
    except requests.exceptions.RequestException as e:
        print(f"Warning: Error fetching metadata for PMID {pmid}: {e}")
    except Exception as e:
        print(f"Warning: Error parsing metadata for PMID {pmid}: {e}")
        
    return authors, license_info, mesh_terms

# Steps
extract_step = Step(
    input_model=Article,
    output_model=ExtractedPairList,
    engine_name="llama3_3_70b_instruct",
    id="clinical_case_extraction",
    name="Clinical Case Extraction",
    emoji="📄",
    sampling=SamplingParams(temperature=0.0, max_tokens=8192),
    system_prompt=EXTRACT_SYSTEM_PROMPT,
    user_prompt="""
    Input text: 
    {{chain_input.text}}

    pmid:
    {{chain_input.pmid}}

    authors:
    {{chain_input.authors}}

    license:
    {{chain_input.license}}

    mesh_terms:
    {{chain_input.mesh_terms}}
    """,
    yield_output=False, # Output will be processed by the Apply step
)

translate_to_french_step = Step(
    input_model=ExtractedPair, # Input is a single ExtractedPair after flattening
    output_model=TranslatedExtractedPairFrench,
    engine_name="llama3_3_70b_instruct",
    id="translate_pair_to_french",
    name="Translate Pair to French",
    emoji="🇫🇷",
    sampling=SamplingParams(temperature=0.1, max_tokens=4096),
    system_prompt=TRANSLATE_FRENCH_SYSTEM_PROMPT,
    user_prompt=TRANSLATE_FRENCH_USER_PROMPT,
)

# Chain Definition
clinical_extraction_translation_chain = Chain(
    name="Clinical Extraction and French Translation",
    emoji="📄🇫🇷",
    steps=[
        extract_step,
        Apply(
            fn=flatten_extracted_pairs,
            id="flatten_extracted_pairs",
            input_model=ExtractedPairList, # Output of extract_step
            output_model=ExtractedPair    # Output type of the function for each item
        ),
        translate_to_french_step,
    ],
    batch_size=100, # Updated batch size to 100
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Clinical Extraction and Translation Chain.")
    parser.add_argument("--debug-metadata", action="store_true", help="Fetch and print article metadata, then exit without running the chain.")
    parser.add_argument("--email", type=str, default="your.email@example.com", help="Email address for NCBI E-utilities API.")
    args = parser.parse_args()

    print("Starting Clinical Extraction and French Translation Chainette example...")
    
    # IMPORTANT: Replace 'your.email@example.com' with your actual email address for NCBI API if not passed via --email.
    ncbi_email = args.email
    if ncbi_email == "your.email@example.com":
        print("WARNING: Using default email 'your.email@example.com' for NCBI API. Please provide a valid email using the --email argument or update the script.")

    tokenizer = AutoTokenizer.from_pretrained("google/medgemma-27b-text-it", trust_remote_code=True)

    data_dir = "rntc/edu3-clinical"

    try:
        # The split argument should give a Dataset (map-style)
        full_dataset_obj = load_dataset(data_dir, split="train") 
    except Exception as e:
        print(f"Failed to load dataset from {data_dir}: {e}. Exiting.")
        exit(1)

    sample_size = 100  # Reduced sample size for simpler processing
    sample_data_from_dataset = []

    if isinstance(full_dataset_obj, Dataset): # Map-style dataset
        actual_sample_size = min(sample_size, len(full_dataset_obj))
        if actual_sample_size > 0:
            sample_data_from_dataset = list(full_dataset_obj.select(range(actual_sample_size)))
        else:
            print("Dataset is empty or sample size is zero.")
    elif isinstance(full_dataset_obj, IterableDataset): # Iterable-style dataset
        print(f"Dataset is Iterable. Taking up to {sample_size} samples.")
        sample_data_from_dataset = list(full_dataset_obj.take(sample_size))
    else:
        print(f"Warning: Unsupported dataset type: {type(full_dataset_obj)}. Attempting to convert to list (may be slow or fail for large datasets).")
        try:
            temp_list = list(full_dataset_obj)
            actual_sample_size = min(sample_size, len(temp_list))
            sample_data_from_dataset = temp_list[:actual_sample_size]
        except Exception as e:
            print(f"Could not process dataset of type {type(full_dataset_obj)}: {e}. Exiting.")
            exit(1)
    
    if not sample_data_from_dataset:
        print("No data sampled from the dataset. Exiting.")
        exit(0)

    if args.debug_metadata:
        print("\n--- Debug Metadata Mode --- ")
        print(f"Loading first 10 items for metadata debugging...")
        debug_items = sample_data_from_dataset[:10]
        
        for i, item in enumerate(debug_items):
            print(f"\nArticle {i+1}:")
            pmid_val = item.get("article_id")
            text_val = item.get("article_text", "")
            
            if pmid_val:
                print(f"  PMID: {pmid_val}")
                authors_list, license_str, mesh_terms_list = fetch_pubmed_metadata(pmid_val, email=ncbi_email)
                print(f"  Authors: {', '.join(authors_list) if authors_list else 'N/A'}")
                print(f"  License: {license_str if license_str else 'N/A'}")
                print(f"  MeSH Terms: {', '.join(mesh_terms_list) if mesh_terms_list else 'N/A'}")
            else:
                print(f"  PMID: N/A")
                print(f"  Text starts with: {text_val[:100]}...")
        
        print("\nMetadata debugging finished. Exiting as per --debug-metadata flag.")
        exit(0)
       
    if not sample_data_from_dataset:
        print("No input data to process. Exiting.")
    else:
        print(f"Processing {len(sample_data_from_dataset)} articles...")
        print(f"Fetching metadata for all articles...")
        
        # Process all articles
        articles = []
        for item in sample_data_from_dataset:
            # Skip item if text more than max_len tokens
            text_val = item.get("article_text", "")
            if text_val:
                token_count = len(tokenizer.encode(text_val))
                if token_count > MAX_LEN:
                    print(f"Skipping article with {token_count} tokens (exceeds max_len of {MAX_LEN})")
                    continue

            # Ensure item is a dictionary, which it should be if from Hugging Face datasets
            if not isinstance(item, dict):
                print(f"Warning: Skipping item of unexpected type: {type(item)}")
                continue

            pmid_val = item.get("article_id") # Use .get() for safety
            text_val = item.get("article_text", "") # Default to empty string if not found

            authors_list, license_str, mesh_terms_list = [], "", []
            if pmid_val:
                print(f"  Fetching metadata for PMID: {pmid_val}...")
                authors_list, license_str, mesh_terms_list = fetch_pubmed_metadata(pmid_val, email=ncbi_email)
                print(f"    -> {len(authors_list)} authors, License: '{license_str}', {len(mesh_terms_list)} MeSH terms")
            else:
                print(f"  No PMID (article_id) found for an item. Text starts with: {text_val[:50]}...")
            
            article_instance = Article(
                text=text_val,
                pmid=pmid_val,
                authors=authors_list,
                license=license_str,
                mesh_terms=mesh_terms_list
            )
            articles.append(article_instance)
        
        if not articles:
            print("No valid articles to process. Exiting.")
            exit(0)
        
        print(f"Loaded {len(articles)} articles with metadata for processing.")
        
        # Run the chain
        output_dir = "/scratch/rtouchen/output_clinical_extraction_translation_test"
        
        result_dataset_dict = clinical_extraction_translation_chain.run(
            articles,
            output_dir=output_dir,
            fmt="jsonl",
            generate_flattened_output=True, # Generates a single file with all final outputs
            max_lines_per_file=100,
            debug=False # Set to True for more verbose logging if needed
        )
        
        print(f"Processing completed. Output saved to {output_dir}")
        if result_dataset_dict: # chain.run now returns a dict of datasets or None
            print(f"Result dataset generated in '{output_dir}' directory.")

    print("Example finished.")
