# Multi-Agent QA Framework

This project provides a modular framework for building and evaluating multi-agent reasoning systems on long documents such as novels or stories. It is designed to work with large language models (LLMs) through APIs (OpenAI-compatible, local deployment, etc.), allowing agents to collaborate and debate over document comprehension tasks.

## Features

- Modular and extensible agent design (`agent.py`)
- Dataset loader with preprocessing utilities (`data_utils.py`)
- Multi-round collaborative reasoning pipeline (`main.py`)
- Unified and structured prompting logic (`utils.py`)
- Supports OpenAI API and local LLM deployments
- Automatic retry and logging for robustness

## Directory Structure

```
├── main.py               # Entry point for running experiments
├── agent.py              # Defines agent classes for different backends
├── data_utils.py         # Dataset loading and context preparation
├── utils.py              # Prompt templates, parsing, voting, etc.
├── results/              # Stores experiment outputs
├── logs/                 # Stores runtime logs
├── .env.example          # Sample environment variable configuration
└── requirements.txt      # Python dependencies
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare environment variables

Create a `.env` file:

```env
API_KEY=your-api-key
BASE_URL=http://127.0.0.1:8000/v1
TOKENIZER_PATH=/path/to/llama/tokenizer
```

You can load this via:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. Prepare datasets

Place datasets like `DetectiveQA` or `NovelQA` under the `datasets/` directory. These should be in `.pkl` format with the expected structure.

### 4. Run an experiment

```bash
python main.py \
  --model llama \
  --dataset DetectiveQA \
  --sample_num 100 \
  --agent_num 5 \
  --repetition_num 1
```

### Parameters

- `--model`: model name (e.g., `llama`, `deepseek`)
- `--dataset`: dataset name (`DetectiveQA`, `NovelQA`)
- `--sample_num`: number of examples to process
- `--agent_num`: number of agents in the reasoning group
- `--repetition_num`: how many times to repeat the full run

## Output

- Logs are saved to `logs/`
- Result JSONs are saved to `results/`
- Each decision contains the final prediction and ground truth for evaluation

## Citation

If you use this codebase in your research, please cite appropriately or link back to the repository.

---

Feel free to extend the agents, customize the prompts, or plug in your own LLMs for research or product development.

