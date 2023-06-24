import datetime
import json
import os
import re
import tarfile
from typing import Dict, Union

import emoji
import requests
import yaml
from termcolor import colored
from transformers import AutoTokenizer, pipeline

MAX_TOKENS_ACCEPTED_BY_SUMMARIZER = 1024
MIN_TOKENS_TO_SUMMARIZE = 200


def load_configs(file_path: str) -> Dict:
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def remove_existing_file(file_path: str) -> None:
    if os.path.exists(file_path):
        os.remove(file_path)


def preprocess_text(text: str) -> str:
    text = re.sub(r"<[^>]*>", "", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"Copyright.*", "", text)
    text = text.replace("\n", " ")
    text = emoji.demojize(text)
    text = re.sub(r":[a-z_&+-]+:", "", text)
    return text


def summarize_text(text: str, tokenizer, summarizer) -> str:
    tokens = tokenizer(text)["input_ids"]
    total_tokens = len(tokens)
    if total_tokens < MIN_TOKENS_TO_SUMMARIZE:
        return text
    if total_tokens > MAX_TOKENS_ACCEPTED_BY_SUMMARIZER:
        tokens = tokens[: int(MAX_TOKENS_ACCEPTED_BY_SUMMARIZER * 0.75)]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return summarizer(
        text,
        max_length=MAX_TOKENS_ACCEPTED_BY_SUMMARIZER,
        min_length=30,
        do_sample=False,
    )[0]["summary_text"]


def process_sequences(sequences: list, tokenizer, summarizer) -> str:
    for i in range(len(sequences)):
        if not sequences[i].startswith("```"):
            sequences[i] = summarize_text(sequences[i], tokenizer, summarizer)
    return " ".join(sequences)


def download_file(
    url: str, repo_info: dict, tokenizer, summarizer, jsonl_file_name: str
) -> None:
    response = requests.get(url)
    filename = url.split("/")[-1]
    text = response.text

    if text is not None and isinstance(text, str):
        text = preprocess_text(text)
        sequences = re.split(r"(```.*?```)", text)
        summarized_text = process_sequences(sequences, tokenizer, summarizer)
        summarized_text = re.sub(r"\s+", " ", summarized_text)
        summarized_text = summarized_text.strip()

        file_dict = {
            "title": filename,
            "repo_owner": repo_info["owner"],
            "repo_name": repo_info["repo"],
            "text": summarized_text,
        }

        with open(jsonl_file_name, "a") as jsonl_file:
            jsonl_file.write(json.dumps(file_dict) + "\n")
    else:
        print(f"Unexpected response text: {text}")


def process_directory(
    path: str,
    repo_info: Dict,
    headers: Dict,
    tokenizer,
    summarizer,
    jsonl_file_name: str,
) -> None:
    base_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/contents/"
    print(colored(f"Processing directory: {path} of repo: {repo_info['repo']}", "blue"))
    response = requests.get(base_url + path, headers=headers)

    if response.status_code == 200:
        files = response.json()
        for file in files:
            if file["type"] == "file" and (
                file["name"].endswith(".mdx") or file["name"].endswith(".md")
            ):
                print(colored(f"Downloading file: {file['name']}", "green"))
                print(colored(f"Download URL: {file['download_url']}", "cyan"))
                download_file(
                    file["download_url"],
                    repo_info,
                    tokenizer,
                    summarizer,
                    jsonl_file_name,
                )
            elif file["type"] == "dir":
                process_directory(
                    file["path"],
                    repo_info,
                    headers,
                    tokenizer,
                    summarizer,
                    jsonl_file_name,
                )
        print(colored("Successfully retrieved files from the directory.", "green"))
    else:
        print(
            colored(
                "Failed to retrieve files. Please check your GitHub token and the repo details.",
                "red",
            )
        )


def main():
    config = load_configs("text_retriever/config.yaml")
    summarizer_model = config["summarizer_model"]
    summarizer = pipeline("summarization", model=summarizer_model)
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model)
    github_token = os.getenv("GITHUB_TOKEN")

    if github_token is None:
        raise ValueError("GITHUB_TOKEN is not set in the environment variables.")

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3.raw",
    }

    current_date = datetime.date.today().strftime("%Y_%m_%d")
    jsonl_file_name = f"data/docs_en_{current_date}.jsonl"

    create_dir("data/")
    remove_existing_file(jsonl_file_name)

    for repo_info in config["github"]["repos"]:
        process_directory(
            repo_info["path"],
            repo_info,
            headers,
            tokenizer,
            summarizer,
            jsonl_file_name,
        )

    with tarfile.open(f"data/docs_en_{current_date}.tar", "w") as tar:
        tar.add(jsonl_file_name)
        print(colored("Successfully compressed the JSONL file.", "green"))


if __name__ == "__main__":
    main()
