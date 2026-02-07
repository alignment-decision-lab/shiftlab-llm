# shiftlab-llm

Research toolkit for diagnosing **domain shift** and testing lightweight **adaptation** strategies for LLM pipelines.

![SHIFT-Lab overview diagram showing a pipeline with domain shift detection and LLM adaptation strategies, including input data transformation, model processing, and output evaluation stages](docs/figures/general_schema.png)

<p align="center">
  <img src="docs/figures/general_schema.png" alt="SHIFT-Lab overview" width="700">
</p>

## Quickstart
```bash
pip install -e .
python -m shiftlab.cli run --config configs/demo.yaml
