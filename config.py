import argparse
import sys
import yaml
from dataclasses import dataclass, field, fields, asdict
from typing import Optional


@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    attn_implementation: Optional[str] = "sdpa"
    max_new_tokens: int = 256
    mixed_precision: bool = True
    compile_model: bool = False

    video_dir: str = "./videos"
    label_path: str = ""
    prompt_style: str = "keyword"           
    sample_fps: float = 1.0
    limit: Optional[int] = None

    embedding_source: str = "penultimate"   
    embedding_pooling: str = "weighted_mean"  
    normalize_embeddings: bool = True
    embedding_store: str = "artifacts/test_embeddings.npz"
    output_npz_path: str = "artifacts/test_embeddings.npz"
    save_raw_hidden: bool = True
    weighting_scheme: str = "original"      

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True

    reasoning_strategy: str = "single_stage"  
    reasoning_tokens: int = 128
    self_consistency_samples: int = 5
    self_consistency_temperature: float = 0.9
    synthesis_temperature: float = 0.6
    description_temperature: float = 0.7

    parallelism: str = "dp"                
    batch_size: int = 1
    num_workers: int = 12
    artifact_dir: str = "./artifacts"
    text_log_path: str = "artifacts/text_log.json"
    output_path: Optional[str] = None


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config()
    for key, value in data.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def load_config_from_args(argv=None) -> Config:
    parser = argparse.ArgumentParser(description="COVR Experiment")
    parser.add_argument("--config", required=True, help="Path to YAML config file")

    known, extra = parser.parse_known_args(argv)
    cfg = load_config(known.config)

    i = 0
    while i < len(extra):
        arg = extra[i]
        if arg.startswith("--"):
            key = arg.lstrip("-").replace("-", "_")
            if hasattr(cfg, key):
                fld = {f.name: f for f in fields(Config)}[key]
                if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                    raw = extra[i + 1]
                    i += 2
                else:
                    raw = "true"
                    i += 1
                if fld.type in ("bool", "Optional[bool]") or fld.type is bool:
                    setattr(cfg, key, raw.lower() in ("true", "1", "yes"))
                elif fld.type in ("int", "Optional[int]") or fld.type is int:
                    setattr(cfg, key, int(raw) if raw.lower() != "none" else None)
                elif fld.type in ("float",) or fld.type is float:
                    setattr(cfg, key, float(raw))
                else:
                    setattr(cfg, key, raw)
            else:
                i += 1
        else:
            i += 1

    return cfg
