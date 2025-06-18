# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenEvolve is a Python-based evolutionary coding agent that implements the AlphaEvolve system from Google DeepMind's 2025 paper. It uses Large Language Models to iteratively improve code through an evolutionary process, evolving entire code files for optimization problems like symbolic regression, circle packing, and function minimization.

## Development Commands

### Setup and Installation
```bash
# Create virtual environment and install dependencies
make install

# Or install directly
pip install -e .
```

### Code Quality
```bash
# Format code with Black (100-character line length)
make lint

# Run type checking (if needed)
python -m mypy openevolve/
```

### Testing
```bash
# Run all tests
make test

# Or run directly
python -m unittest discover -s tests -p "test_*.py"

# Run specific test
python -m unittest tests.test_valid_configs
```

### Running Examples
```bash
# Basic evolution run
python openevolve-run.py examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 100

# Resume from checkpoint
python openevolve-run.py examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --checkpoint examples/function_minimization/openevolve_output/checkpoints/checkpoint_50 --iterations 50
```

### Visualization
```bash
# Launch web-based evolution tree visualizer
make visualizer

# Or run directly
python scripts/visualizer.py --path examples/
```

### Docker
```bash
# Build image
make docker-build

# Run with Docker
make docker-run
```

## Architecture Overview

### Core Components
- **`openevolve/controller.py`**: Main orchestration logic that coordinates the evolution pipeline
- **`openevolve/database.py`**: Program storage, evolution tracking, and MAP-Elites implementation
- **`openevolve/evaluator.py`**: Asynchronous code evaluation and scoring system
- **`openevolve/llm/`**: LLM integration supporting OpenAI-compatible APIs and ensemble models
- **`openevolve/prompt/`**: Prompt engineering with template variations and context sampling

### Key Design Patterns
- **Asynchronous Pipeline**: Heavy use of async/await for concurrent evaluation and LLM calls
- **Configuration-Driven**: Highly customizable via YAML files in `configs/`
- **Modular Evaluation**: Evaluators return `EvaluationResult` objects with metrics and artifacts
- **Checkpoint System**: Automatic state saving every N iterations with full resume capability

### Evolution Process
1. **Prompt Sampler** creates context-rich prompts with past programs and scores
2. **LLM Ensemble** generates code modifications via multiple language models
3. **Evaluator Pool** tests programs asynchronously and assigns fitness scores
4. **Program Database** stores results and guides future evolution using MAP-Elites

### Artifacts Side-Channel
Programs can return build errors, profiling data, and execution context via the `artifacts` field in `EvaluationResult`. This feedback is included in subsequent prompts to help LLMs learn from failures. Configure with `enable_artifacts` and `include_artifacts` settings.

## Configuration System

Primary config file: `configs/default_config.yaml`

Key configuration areas:
- **LLM Settings**: Model selection, temperature, API endpoints, ensemble weights
- **Evolution Parameters**: Population size, island model, diff vs full rewrites
- **Evaluation**: Timeouts, parallel workers, cascade evaluation
- **Database**: MAP-Elites behavior dimensions, checkpointing intervals

## Project Structure for New Problems

1. **Initial Program**: Mark evolution regions with `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END`
2. **Evaluator**: Return `EvaluationResult` with metrics dict and optional artifacts
3. **Config**: Customize evolution parameters, LLM settings, and evaluation timeouts
4. **Examples**: Reference `examples/` directory for complete problem setups

## Important Development Notes

- **Python 3.9+** required
- **Black formatting** with 100-character line length enforced
- **Type hints** expected (mypy configuration included)
- **Async patterns** throughout - use `await` for LLM calls and evaluations
- **Error handling** via artifacts channel improves evolution quality
- **Checkpointing** preserves complete state - always resumable