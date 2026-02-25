# CoVR-R
Reasoning Aware Composed Video Retrieval

## Quick Start

### Install

```
pip install -r requirements.txt
```

### WebVid8M

```bash
python generate_embeddings.py --config configs/webvid8m.yaml
python evaluate.py --config configs/webvid8m.yaml
```

### Dense WebVid8M

```bash
python generate_embeddings.py --config configs/dense_webvid8m.yaml
python evaluate.py --config configs/dense_webvid8m.yaml
```

### Reasoning WebVid8M

```bash
python generate_embeddings.py --config configs/reasoning_webvid8m.yaml
python evaluate.py --config configs/reasoning_webvid8m.yaml
python evaluate.py --config configs/reasoning_webvid8m.yaml --reasoning_strategy self_consistency
python evaluate.py --config configs/reasoning_webvid8m.yaml --reasoning_strategy single_stage
```
