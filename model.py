import logging
import os
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logger = logging.getLogger(__name__)


STOP_WORDS = {
    # Basic English stop words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "can", "must", "shall", "i", "you", "he", "she", "it",
    "we", "they", "me", "him", "her", "us", "them", "this", "that", "these",
    "those", "here", "there", "where", "when", "why", "how", "very", "quite",
    "rather", "really", "just", "only", "also", "too", "so", "such", "some",
    "any", "all", "every", "each", "both", "either", "neither", "no", "not",
    "as", "than", "like", "while", "during", "before", "after", "since", "until",
    # Video description stop words
    "video", "videos", "clip", "clips", "footage", "scene", "scenes", "frame", "frames",
    "shows", "showing", "displays", "displaying", "depicts", "depicting", "features",
    "featuring", "contains", "containing", "includes", "including", "presents",
    "presenting", "demonstrates", "demonstrating", "illustrates", "illustrating",
    "reveals", "revealing", "captures", "capturing", "records", "recording",
    "documents", "documenting", "describes", "describing", "portrays", "portraying",
    "represents", "representing", "appears", "appearing", "seems", "seeming",
    "looks", "looking", "becomes", "becoming", "begins", "beginning", "starts",
    "starting", "ends", "ending", "continues", "continuing", "occurs", "occurring",
    "happens", "happening", "takes", "taking", "place", "places",
    "content", "contents", "material", "materials", "element", "elements",
    "part", "parts", "section", "sections", "portion", "portions", "segment",
    "segments", "sequence", "sequences", "moment", "moments", "instance",
    "instances", "time", "times", "period", "periods",
    "throughout", "overall", "general", "generally", "typically", "usually",
    "normally", "clearly", "obviously", "apparently", "evidently", "notably",
    "particularly", "especially", "mainly", "mostly", "primarily", "largely",
    "essentially", "basically", "simply",
    "reference", "references", "referencing", "described", "description",
    "descriptions", "detail", "details", "detailed", "specific", "specifically",
    "particular", "observable", "visible", "visual", "visually", "seen", "seeing",
    "view", "viewing", "present", "presence", "tense", "cohesive", "paragraph",
    "bullet", "points",
    # Prompt-echo terms
    "primary", "subjects", "subject", "actions", "action", "transitions",
    "transition", "environment", "environments", "background", "backgrounds",
    "lighting", "color", "colors", "palette", "camera", "motion", "framing",
    "pacing", "events", "event", "atmosphere", "mood", "moods", "vivid",
    "vividly", "focus", "focused", "focusing", "cover", "covers", "covering",
    "noteworthy", "write", "written", "writing",
    # Transitional
    "first", "second", "third", "next", "then", "finally", "initially",
    "subsequently", "meanwhile", "however", "therefore", "thus", "hence",
    "accordingly", "consequently", "furthermore", "moreover", "additionally",
    "likewise", "similarly", "conversely", "nonetheless", "nevertheless",
    "otherwise", "indeed", "certainly", "surely",
    # Generic descriptive
    "thing", "things", "way", "ways", "kind", "kinds", "type", "types",
    "sort", "sorts", "manner", "manners", "form", "forms", "style", "styles",
    "aspect", "aspects", "feature", "quality", "qualities", "characteristic",
    "characteristics",
}

FILLER_WORDS = {"um", "uh", "er", "ah", "well", "you know", "i mean", "sort of", "kind of"}
BASIC_STOP_WORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}


def calculate_token_weights(token_strings: List[str], scheme: str = "original") -> torch.Tensor:
    if scheme == "uniform":
        return torch.ones(len(token_strings), dtype=torch.float32)

    if scheme == "graduated":
        weights = []
        for tok in token_strings:
            c = tok.strip().lower()
            if c.startswith("<") and c.endswith(">"):
                weights.append(0.0)
            elif c in STOP_WORDS:
                weights.append(0.2)
            elif len(c) <= 2:
                weights.append(0.5)
            else:
                weights.append(1.0)
        return torch.tensor(weights, dtype=torch.float32)

    if scheme == "nltk":
        weights = []
        for tok in token_strings:
            c = tok.strip().lower()
            weights.append(0.1 if (c in STOP_WORDS or c.startswith("<")) else 1.0)
        return torch.tensor(weights, dtype=torch.float32)

    if scheme == "basic":
        weights = []
        for tok in token_strings:
            c = tok.strip().lower()
            weights.append(0.1 if (c in BASIC_STOP_WORDS or c.startswith("<")) else 1.0)
        return torch.tensor(weights, dtype=torch.float32)

    weights = []
    for tok in token_strings:
        c = tok.strip().lower()
        if not c:
            weights.append(0.1)
        elif c in ".,!?;:()[]{}\"'-/\\":
            weights.append(0.1)
        elif c in STOP_WORDS:
            weights.append(0.3)
        elif c in FILLER_WORDS:
            weights.append(0.1)
        elif c.isdigit():
            weights.append(0.7)
        elif len(c) <= 2:
            weights.append(0.4)
        elif c.startswith(("▁", "##", "</w>")):
            actual = c.lstrip("▁##").rstrip("</w>").lower()
            weights.append(0.3 if actual in STOP_WORDS else 0.8)
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32)


def pool_hidden_sequences(
    hidden_steps: List[torch.Tensor],
    pooling: str,
    token_strings: Optional[List[str]] = None,
    weighting_scheme: str = "original",
) -> torch.Tensor:
    stacked = torch.stack(hidden_steps, dim=1)
    mode = (pooling or "mean").lower()

    if mode == "mean":
        pooled = stacked.mean(dim=1)
    elif mode == "weighted_mean" and token_strings is not None:
        weights = calculate_token_weights(token_strings, weighting_scheme)
        weights = weights / (weights.sum() + 1e-8)
        weights = weights.unsqueeze(0).unsqueeze(-1)
        pooled = (stacked * weights).sum(dim=1)
    elif mode == "last":
        pooled = stacked[:, -1, :]
    elif mode == "max":
        pooled = stacked.max(dim=1).values
    elif mode == "mean_last":
        pooled = torch.cat([stacked.mean(dim=1), stacked[:, -1, :]], dim=-1)
    elif mode == "mean_max":
        pooled = torch.cat([stacked.mean(dim=1), stacked.max(dim=1).values], dim=-1)
    else:
        pooled = stacked.mean(dim=1)

    return pooled[0]


def hook_penultimate_hidden(module, inputs, output, container):
    if inputs and isinstance(inputs[0], torch.Tensor):
        hidden = inputs[0]
        container["steps"].append(hidden[:, -1, :].detach().float().cpu())
        container.setdefault("raw_sequences", []).append(hidden.detach().float().cpu())


def hook_vision_projection(module, inputs, output, container):
    proj = output
    if isinstance(proj, (tuple, list)):
        proj = proj[0]
    if not isinstance(proj, torch.Tensor):
        return
    with torch.no_grad():
        proj_cpu = proj.detach().cpu().float()
        if proj_cpu.dim() == 3:
            mean_proj = proj_cpu.mean(dim=1)
        elif proj_cpu.dim() == 2:
            mean_proj = proj_cpu
        else:
            return
    container.setdefault("steps", []).append(mean_proj)
    container.setdefault("raw_sequences", []).append(proj_cpu)


class VideoProcessor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None

    def load_model(self, gpu_id=0):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs = {"low_cpu_mem_usage": True, "torch_dtype": dtype}
        if self.cfg.attn_implementation:
            kwargs["attn_implementation"] = self.cfg.attn_implementation

        if self.cfg.parallelism == "mp":
            kwargs["device_map"] = "auto"
            logger.info(f"Loading model with model parallelism: {self.cfg.model_name}")
        else:
            logger.info(f"Loading model on GPU {gpu_id}: {self.cfg.model_name}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.cfg.model_name, **kwargs)

        if self.cfg.parallelism == "dp":
            self.model = self.model.to(f"cuda:{gpu_id}")

        self.processor = AutoProcessor.from_pretrained(self.cfg.model_name)
        logger.info("Model loaded successfully")

    def move_to_device(self, inputs):
        device = next(self.model.parameters()).device
        if isinstance(inputs, list):
            return [{k: v.to(device) if hasattr(v, "to") else v for k, v in inp.items()} for inp in inputs]
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


class EmbeddingExtractor:
    def __init__(self, cfg):
        self.cfg = cfg

    def _register_hook(self, model):
        container = {"steps": []}
        handle = None

        raw_model = model.module if hasattr(model, "module") else model

        try:
            if self.cfg.embedding_source == "vision":
                target = raw_model.visual.merger
                handle = target.register_forward_hook(
                    partial(hook_vision_projection, container=container)
                )
            else:
                target = raw_model.get_output_embeddings()
                handle = target.register_forward_hook(
                    partial(hook_penultimate_hidden, container=container)
                )
        except Exception:
            handle = None

        return container, handle

    def save_raw_hidden(self, raw_sequences, video_path):
        if not self.cfg.save_raw_hidden:
            return
        raw_path = os.path.join(self.cfg.artifact_dir, "raw_hidden", f"{os.path.basename(video_path)}.pt")
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        torch.save(raw_sequences, raw_path)

    def extract_embeddings(
        self,
        inputs,
        video_paths: List[str],
        processor: VideoProcessor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
    ) -> List[Dict]:
        inputs = processor.move_to_device(inputs)
        container, handle = self._register_hook(processor.model)

        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "pad_token_id": processor.processor.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k})
        else:
            gen_kwargs["do_sample"] = False

        if isinstance(inputs, list):
            results = []
            for inp, vpath in zip(inputs, video_paths):
                inp = processor.move_to_device(inp)
                container["steps"] = []
                output_ids = self._generate(processor, inp, gen_kwargs)
                result = self._process_output(output_ids, inp, vpath, processor, container)
                results.append(result)
            if handle:
                handle.remove()
            return results

        model = processor.model.module if hasattr(processor.model, "module") else processor.model
        if self.cfg.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**inputs, **gen_kwargs)
        else:
            output_ids = model.generate(**inputs, **gen_kwargs)

        if handle:
            handle.remove()

        input_length = inputs["input_ids"].shape[1]
        trimmed = output_ids[:, input_length:]
        decoded_texts = processor.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        results = []
        for i, (vpath, text) in enumerate(zip(video_paths, decoded_texts)):
            tokens = trimmed[i].tolist()
            token_strings = [processor.processor.tokenizer.decode([t]) for t in tokens]

            embedding = None
            if container["steps"]:
                ts = token_strings if self.cfg.embedding_pooling == "weighted_mean" else None
                embedding = self._pool_and_normalize(container, ts)
                if "raw_sequences" in container:
                    self.save_raw_hidden(container["raw_sequences"], vpath)

            results.append({
                "video_name": os.path.basename(vpath),
                "decoded_text": text,
                "embedding": embedding,
                "token_mapping": {"tokens": tokens, "token_strings": token_strings, "full_text": text},
            })
        return results

    def _generate(self, processor, inp, gen_kwargs):
        if self.cfg.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return processor.model.generate(**inp, **gen_kwargs)
        return processor.model.generate(**inp, **gen_kwargs)

    def _process_output(self, output_ids, inp, video_path, processor, container):
        input_length = inp["input_ids"].shape[1]
        trimmed = output_ids[:, input_length:]
        text = processor.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        tokens = trimmed[0].tolist()
        token_strings = [processor.processor.tokenizer.decode([t]) for t in tokens]

        embedding = None
        if container["steps"]:
            ts = token_strings if self.cfg.embedding_pooling == "weighted_mean" else None
            embedding = self._pool_and_normalize(container, ts)
            if "raw_sequences" in container:
                self.save_raw_hidden(container["raw_sequences"], video_path)

        return {
            "video_name": os.path.basename(video_path),
            "decoded_text": text,
            "embedding": embedding,
            "token_mapping": {"tokens": tokens, "token_strings": token_strings, "full_text": text},
        }

    def _pool_and_normalize(self, container, token_strings):
        pooled = pool_hidden_sequences(
            container["steps"],
            self.cfg.embedding_pooling,
            token_strings,
            self.cfg.weighting_scheme,
        ).detach()
        if self.cfg.normalize_embeddings:
            pooled = F.normalize(pooled.unsqueeze(0), dim=-1).squeeze(0)
        return pooled.cpu().float().numpy()
