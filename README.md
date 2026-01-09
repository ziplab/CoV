# CoV: Chain-of-View Prompting for Spatial Reasoning


## Overview

![CoV Teaser](assets/teaser.jpg)



## Project Structure

```
repo/
├── cov/                      # Main package
├── scripts/                  # Utility scripts
├── tools/                    # Data processing tools
├── main.py                   # Main entry point
├── pixi.toml                 # Pixi environment configuration
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA support (recommended for HabTat Sim)

### Using Pixi

The project uses [Pixi](https://pixi.sh) for dependency management:

```bash
# Install dependencies
pixi install

# Activate the environment
pixi shell
```

Key dependencies:
- `habitat-sim==0.3.3` - 3D environment simulation
- `litellm>=1.80.0` - Multi-LLM interface
- `hydra-core>=1.3.2` - Configuration management
- `jinja2>=3.1.6` - Prompt templating
- `python-dotenv` - Environment variable management

### Setup Environment Variables

Create a `.env` file in the root directory with your API credentials:

```bash
# OpenAI
OPENAI_API_KEY=your_key_here

# OpenRouter
OPENROUTER_API_BASE=https://openrouter.api.com/api/v1
OPENROUTER_API_KEY=your_key_here

# Dashscop
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_API_KEY=your_key_here
```

### Prepare Dataset

1. Download the OpenEQA dataset
2. Place question files in `data/` directory
3. Place scene frames in `data/frames/` directory

## Run evaluation

### Basic Usage

Run the agent on OpenEQA questions:

```bash
python main.py
```

### Configuration

Configuration is managed through Hydra. Create config files in `conf/` or pass parameters directly:

```bash
# Use a specific model
python main.py model=gpt

# Use CoV agent
python main.py agent=cov
```

### Available Models

You can set your own model backend in [cov/config.py](cov/config.py).


### Output

Results are saved to the configured output directory with:
- JSON files containing answers and metadata
- HTML reports showing navigation history and visualizations
- Screenshots of selected views and bird's eye views


## Citation

If you use Chain of View in your research, please cite this work.
