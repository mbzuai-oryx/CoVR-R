import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, load_config_from_args
from data import Record, VideoDataset, collate_fn, load_records
from model import EmbeddingExtractor, VideoProcessor
from prompts import get_prompts

logger = logging.getLogger(__name__)


def load_embedding_store(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as npz:
        return {key: np.asarray(npz[key], dtype=np.float32) for key in npz.files}


def compute_recall(scores_q2t: np.ndarray, gt_indices: Sequence[int]) -> Dict[str, float]:
    ranks = np.zeros(len(gt_indices), dtype=np.float32)

    for query_idx, (row, gt_idx) in enumerate(zip(scores_q2t, gt_indices)):
        sorted_indices = np.argsort(row)[::-1]
        matches = np.where(sorted_indices == gt_idx)[0]
        if matches.size == 0:
            raise ValueError(f"GT index {gt_idx} not found in row {query_idx}")
        ranks[query_idx] = matches[0]

    total = len(ranks)
    tr1 = 100.0 * np.count_nonzero(ranks < 1) / total
    tr5 = 100.0 * np.count_nonzero(ranks < 5) / total
    tr10 = 100.0 * np.count_nonzero(ranks < 10) / total
    tr50 = 100.0 * np.count_nonzero(ranks < 50) / total
    tr1000 = 100.0 * np.count_nonzero(ranks < 1000) / total
    tr2000 = 100.0 * np.count_nonzero(ranks < 2000) / total

    return {
        "R1": round(tr1, 4),
        "R5": round(tr5, 4),
        "R10": round(tr10, 4),
        "R50": round(tr50, 4),
        "R1000": round(tr1000, 4),
        "R2000": round(tr2000, 4),
        "R_mean": round((tr1 + tr5 + tr10) / 3.0, 4),
    }


def _generate_text(processor: VideoProcessor, inputs, max_tokens: int,
                   temperature: float, top_p: float, top_k: int,
                   do_sample: bool) -> str:
    """Run model.generate and return decoded text (no hooks)."""
    inputs = processor.move_to_device(inputs)
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "pad_token_id": processor.processor.tokenizer.eos_token_id,
        "use_cache": True,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs.update({"temperature": temperature, "top_p": top_p, "top_k": top_k})

    model = processor.model.module if hasattr(processor.model, "module") else processor.model
    with torch.no_grad():
        if processor.cfg.mixed_precision:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**inputs, **gen_kwargs)
        else:
            output_ids = model.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[1]
    trimmed = output_ids[:, input_length:]
    text = processor.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return text


def extract_reasoning_and_embedding(
    record: Record,
    processor: VideoProcessor,
    extractor: EmbeddingExtractor,
    cfg: Config,
) -> Optional[Dict[str, Any]]:
    prompts = get_prompts(cfg.prompt_style)
    strategy = cfg.reasoning_strategy

    if strategy == "single_stage":
        prompt = prompts["edit"](record.edit_instruction)
        dataset = VideoDataset([record.video_path], prompt, cfg.sample_fps)
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor.processor))

        for inputs, vpaths in loader:
            results = extractor.extract_embeddings(
                inputs, vpaths, processor,
                temperature=cfg.temperature, top_p=cfg.top_p,
                top_k=cfg.top_k, do_sample=cfg.do_sample,
            )
            if results and results[0]["embedding"] is not None:
                r = results[0]
                return {
                    "record": record, "prompt": prompt,
                    "text_output": r["decoded_text"],
                    "embedding": r["embedding"],
                    "reasoning": None, "reasoning_traces": None,
                    "token_mapping": r.get("token_mapping"),
                }
            break
        return None

    reasoning_prompt = prompts["edit_reasoning"](record.edit_instruction)
    reasoning_dataset = VideoDataset([record.video_path], reasoning_prompt, cfg.sample_fps)
    reasoning_loader = DataLoader(reasoning_dataset, batch_size=1, shuffle=False,
                                  collate_fn=lambda b: collate_fn(b, processor.processor))

    reasoning_traces = []
    num_samples = cfg.self_consistency_samples if strategy == "self_consistency" else 1

    for inputs, _ in reasoning_loader:
        for sample_idx in range(num_samples):
            try:
                trace = _generate_text(
                    processor, inputs, max_tokens=cfg.reasoning_tokens,
                    temperature=cfg.self_consistency_temperature,
                    top_p=cfg.top_p, top_k=cfg.top_k, do_sample=True,
                )
                if trace.strip():
                    reasoning_traces.append(trace.strip())
                    logger.info(f"  Reasoning sample {sample_idx + 1}/{num_samples}: {trace[:100]}...")
            except Exception as e:
                logger.warning(f"  Failed reasoning sample {sample_idx + 1}: {e}")
        break

    if not reasoning_traces:
        return None

    final_reasoning = reasoning_traces[0]
    if len(reasoning_traces) > 1:
        synthesis_prompt = prompts["consistency_synthesis"](record.edit_instruction, reasoning_traces)
        synth_dataset = VideoDataset([record.video_path], synthesis_prompt, cfg.sample_fps)
        synth_loader = DataLoader(synth_dataset, batch_size=1, shuffle=False,
                                  collate_fn=lambda b: collate_fn(b, processor.processor))
        for inputs, _ in synth_loader:
            synthesized = _generate_text(
                processor, inputs, max_tokens=cfg.reasoning_tokens,
                temperature=cfg.synthesis_temperature,
                top_p=cfg.top_p, top_k=cfg.top_k, do_sample=cfg.do_sample,
            )
            if synthesized.strip():
                final_reasoning = synthesized.strip()
            logger.info(f"  Synthesized reasoning: {final_reasoning[:100]}...")
            break

    description_prompt = prompts["edit_description"](record.edit_instruction, final_reasoning)
    desc_dataset = VideoDataset([record.video_path], description_prompt, cfg.sample_fps)
    desc_loader = DataLoader(desc_dataset, batch_size=1, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, processor.processor))

    for inputs, vpaths in desc_loader:
        results = extractor.extract_embeddings(
            inputs, vpaths, processor,
            temperature=cfg.description_temperature,
            top_p=cfg.top_p, top_k=cfg.top_k, do_sample=cfg.do_sample,
        )
        if results and results[0]["embedding"] is not None:
            r = results[0]
            return {
                "record": record, "prompt": description_prompt,
                "text_output": r["decoded_text"],
                "embedding": r["embedding"],
                "reasoning": final_reasoning,
                "reasoning_traces": reasoning_traces if len(reasoning_traces) > 1 else None,
                "token_mapping": r.get("token_mapping"),
            }
        break
    return None


def process_records_on_gpu(gpu_id: int, records: List[Record], cfg: Config, output_file: str):
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    torch.cuda.set_device(gpu_id)

    processor = VideoProcessor(cfg)
    processor.load_model(gpu_id)

    if cfg.compile_model:
        processor.model = torch.compile(processor.model)

    extractor = EmbeddingExtractor(cfg)

    results = []
    for record in tqdm(records, desc=f"GPU {gpu_id}"):
        try:
            result = extract_reasoning_and_embedding(record, processor, extractor, cfg)
            if result:
                results.append(result)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"GPU {gpu_id} failed on {record.video_path}: {e}")

    if results:
        np.savez_compressed(output_file, **{f"result_{i}": r for i, r in enumerate(results)})
        logger.info(f"GPU {gpu_id} saved {len(results)} results")


def save_artifacts(
    artifact_dir: str,
    records: Sequence[Record],
    prompts_used: Sequence[str],
    text_outputs: Sequence[str],
    raw_query_matrix: torch.Tensor,
    normalized_query_matrix: torch.Tensor,
    gt_indices: Sequence[int],
    candidate_keys: Sequence[str],
    scores: np.ndarray,
    reasoning_traces: Optional[Sequence[Optional[str]]] = None,
    all_reasoning_traces: Optional[Sequence[Optional[List[str]]]] = None,
    token_mappings: Optional[Sequence[Optional[Dict]]] = None,
) -> Dict[str, str]:
    os.makedirs(artifact_dir, exist_ok=True)

    embeddings_path = os.path.join(artifact_dir, "query_embeddings.npz")
    np.savez_compressed(
        embeddings_path,
        raw_embeddings=raw_query_matrix.detach().cpu().numpy(),
        normalized_embeddings=normalized_query_matrix.detach().cpu().numpy(),
        gt_indices=np.asarray(gt_indices, dtype=np.int32),
        candidate_keys=np.asarray(candidate_keys),
        scores=np.asarray(scores, dtype=np.float32),
    )

    records_path = os.path.join(artifact_dir, "records.jsonl")
    with open(records_path, "w", encoding="utf-8") as handle:
        for idx, record in enumerate(records):
            payload = asdict(record)
            payload["target_key"] = record.target_key
            payload["prompt"] = prompts_used[idx]
            payload["text_output"] = text_outputs[idx]
            payload["gt_candidate_index"] = int(gt_indices[idx])
            if reasoning_traces and idx < len(reasoning_traces) and reasoning_traces[idx]:
                payload["reasoning"] = reasoning_traces[idx]
            if all_reasoning_traces and idx < len(all_reasoning_traces) and all_reasoning_traces[idx]:
                payload["all_reasoning_traces"] = all_reasoning_traces[idx]
            if token_mappings and idx < len(token_mappings) and token_mappings[idx]:
                payload["token_mapping"] = token_mappings[idx]
            json.dump(payload, handle, ensure_ascii=False)
            handle.write("\n")

    text_path = os.path.join(artifact_dir, "text_outputs.txt")
    with open(text_path, "w", encoding="utf-8") as handle:
        for idx, (record, text) in enumerate(zip(records, text_outputs)):
            handle.write(f"=== Record {idx} (Index: {record.index}) ===\n")
            handle.write(f"Edit: {record.edit_instruction}\n")
            handle.write(f"Ref: {record.reference_token}  Target: {record.target_token}\n")
            if all_reasoning_traces and idx < len(all_reasoning_traces) and all_reasoning_traces[idx]:
                handle.write(f"\nReasoning Traces ({len(all_reasoning_traces[idx])}):\n")
                for ti, trace in enumerate(all_reasoning_traces[idx]):
                    handle.write(f"  Trace {ti + 1}: {trace}\n")
                handle.write("\n")
            if reasoning_traces and idx < len(reasoning_traces) and reasoning_traces[idx]:
                handle.write(f"Synthesized Reasoning:\n{reasoning_traces[idx]}\n\n")
            handle.write(f"Output:\n{text}\n\n")

    return {"query_embeddings": embeddings_path, "records": records_path, "text_outputs": text_path}


def save_summary(summary: Dict[str, Any], artifact_dir: str, output_path: Optional[str] = None) -> str:
    os.makedirs(artifact_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(artifact_dir, "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return output_path


def run_evaluation(cfg: Config) -> Dict[str, Any]:
    start_time = time.time()

    embedding_store = load_embedding_store(cfg.embedding_store)
    logger.info(f"Embedding store: {len(embedding_store)} videos, "
                f"shape: {next(iter(embedding_store.values())).shape}")

    records = load_records(cfg.label_path, cfg.video_dir)
    available = set(embedding_store.keys())
    records = [r for r in records if f"{r.reference_token}.mp4" in available]
    logger.info(f"Found {len(records)} records with available reference embeddings")

    if cfg.limit:
        records = records[: cfg.limit]

    candidate_keys = sorted(embedding_store.keys())
    candidate_index = {key: idx for idx, key in enumerate(candidate_keys)}
    candidate_matrix = torch.from_numpy(np.stack([embedding_store[k] for k in candidate_keys])).float()
    candidate_matrix = F.normalize(candidate_matrix, dim=-1)

    if cfg.parallelism == "dp" and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        chunks = [records[i::num_gpus] for i in range(num_gpus)]
        logger.info(f"Using {num_gpus} GPUs for data parallelism")

        processes, gpu_files = [], {}
        for gid in range(num_gpus):
            if chunks[gid]:
                ofile = os.path.join(cfg.artifact_dir, f"temp_gpu{gid}_results.npz")
                gpu_files[gid] = ofile
                p = mp.Process(target=process_records_on_gpu, args=(gid, chunks[gid], cfg, ofile))
                processes.append(p)
                p.start()
        for p in processes:
            p.join()

        all_results = []
        for gid, ofile in gpu_files.items():
            if os.path.exists(ofile):
                data = np.load(ofile, allow_pickle=True)
                for key in data.files:
                    all_results.append(data[key].item())
                os.remove(ofile)

    else:
        processor = VideoProcessor(cfg)
        processor.load_model(0)
        if cfg.compile_model:
            processor.model = torch.compile(processor.model)
        extractor = EmbeddingExtractor(cfg)

        all_results = []
        for record in tqdm(records, desc="Processing"):
            if not os.path.exists(record.video_path):
                continue
            try:
                result = extract_reasoning_and_embedding(record, processor, extractor, cfg)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error processing {record.video_path}: {e}")

    query_features, gt_indices, ref_indices = [], [], []
    processed_records, prompts_used, text_outputs = [], [], []
    reasoning_list, all_traces_list, token_map_list = [], [], []

    for result in all_results:
        rec = result["record"]
        target_idx = candidate_index.get(rec.target_key)
        reference_idx = candidate_index.get(f"{rec.reference_token}.mp4")
        if target_idx is not None and reference_idx is not None:
            query_features.append(torch.from_numpy(np.asarray(result["embedding"], dtype=np.float32)))
            gt_indices.append(target_idx)
            ref_indices.append(reference_idx)
            processed_records.append(rec)
            prompts_used.append(result["prompt"])
            text_outputs.append(result["text_output"])
            reasoning_list.append(result.get("reasoning"))
            all_traces_list.append(result.get("reasoning_traces"))
            token_map_list.append(result.get("token_mapping"))

    if not query_features:
        raise ValueError("No valid queries processed")

    raw_query_matrix = torch.stack(query_features).float()
    query_matrix = F.normalize(raw_query_matrix, dim=-1) if cfg.normalize_embeddings else raw_query_matrix

    logger.info(f"Query: {query_matrix.shape}, Gallery: {candidate_matrix.shape}")
    scores = torch.matmul(query_matrix, candidate_matrix.T).cpu().numpy()

    for i, ref_idx in enumerate(ref_indices):
        scores[i, ref_idx] = -np.inf

    eval_result = compute_recall(scores, gt_indices)

    artifact_paths = save_artifacts(
        cfg.artifact_dir, processed_records, prompts_used, text_outputs,
        raw_query_matrix, query_matrix, gt_indices, candidate_keys, scores,
        reasoning_list, all_traces_list, token_map_list,
    )

    elapsed = time.time() - start_time

    summary = {
        "metrics": eval_result,
        "processed": len(processed_records),
        "total_records": len(records),
        "elapsed_seconds": round(elapsed, 2),
        "candidate_count": len(candidate_keys),
        "config": asdict(cfg),
        "artifact_paths": artifact_paths,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reasoning_strategy": cfg.reasoning_strategy,
        "self_consistency_samples": cfg.self_consistency_samples,
        "reasoning_samples": sum(1 for r in reasoning_list if r is not None),
    }

    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    cfg = load_config_from_args()

    if cfg.parallelism == "dp":
        mp.set_start_method("spawn", force=True)

    summary = run_evaluation(cfg)

    print("\nRecall metrics:")
    for key, value in summary["metrics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"\nProcessed: {summary['processed']}/{summary['total_records']}")
    print(f"Elapsed: {summary['elapsed_seconds']}s")

    saved = save_summary(summary, cfg.artifact_dir, cfg.output_path)
    print(f"Saved to: {saved}")
    print(f"Artifacts: {cfg.artifact_dir}")


if __name__ == "__main__":
    main()
