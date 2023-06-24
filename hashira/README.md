# Text Retriever

This subpackage is a utility tool for downloading markdown files from specified GitHub repositories and converting the text content into a `.jsonl` file. It was developed to facilitate retrieval of documentations and blog posts for further data processing.

## How it works

The `text_retriever.py` script uses the GitHub API to access the repository content. It looks for `.md` and `.mdx` files in the specified path, downloads them and stores the content in a `.jsonl` file, each entry being a separate markdown file. Each entry in the `.jsonl` file contains keys for the `title` of the file, `body` of the file, and `source` (which includes owner and repository information). 

## Getting Started

The Text Retriever subpackage uses `poetry` for dependency management. To get started, you need to install the dependencies via:

```shell
poetry install
```

The GitHub token is required for accessing GitHub repositories. It should be stored as an environment variable `GITHUB_TOKEN`.

### Configuring repositories

The `config.yaml` file is used for setting up the repositories to be processed. Each repository is specified by its owner, repo, and path.

Here is a sample configuration:

```yaml
repositories:
  - owner: "huggingface"
    repo: "transformers"
    path: "docs/source/en"
  - owner: "huggingface"
    repo: "blog"
    path: ""
```

Once you have updated the `config.yaml` with the repositories you want to process, run the `text_retriever.py` script (here using `poetry run` from the `hashira` directory):

```shell
poetry run python python text_retriever/text_retriever.py
```

The script will create a `.jsonl` file in the `data/` directory, named `documentation_en_<current_date>.jsonl`, where `<current_date>` is the date when the script is run.

## Important Notes

* Ensure the `data/` directory exists before running the script. If not, the script will attempt to create it.
* If a file with the same date already exists in the data/ directory, the script will delete it before starting the download.