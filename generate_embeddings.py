import json
import logging
import multiprocessing as mp
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, load_config_from_args
from data import VideoDataset, collate_fn, list_videos
from model import EmbeddingExtractor, VideoProcessor
from prompts import get_prompts

logging.getLogger("qwen_vl_utils").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_videos_on_gpu(gpu_id: int, video_paths, cfg: Config, output_file: str):
    torch.cuda.set_device(gpu_id)

    processor = VideoProcessor(cfg)
    processor.load_model(gpu_id)
    extractor = EmbeddingExtractor(cfg)

    prompt = get_prompts(cfg.prompt_style)["reference"]()

    results = {}
    text_log_file = None
    if cfg.text_log_path:
        text_log_path = f"{cfg.text_log_path}.gpu{gpu_id}"
        os.makedirs(os.path.dirname(text_log_path) or ".", exist_ok=True)
        text_log_file = open(text_log_path, "w", encoding="utf-8")

    try:
        for video_path in tqdm(video_paths, desc=f"GPU {gpu_id}"):
            try:
                dataset = VideoDataset([video_path], prompt, cfg.sample_fps)
                loader = DataLoader(
                    dataset, batch_size=1, shuffle=False,
                    collate_fn=lambda b: collate_fn(b, processor.processor),
                )

                for inputs, vpaths in loader:
                    batch_results = extractor.extract_embeddings(
                        inputs, vpaths, processor,
                        temperature=cfg.temperature, top_p=cfg.top_p,
                        top_k=cfg.top_k, do_sample=cfg.do_sample,
                    )
                    for r in batch_results:
                        key = r["video_name"]
                        logger.info(f"GPU {gpu_id} [{key}] => {r['decoded_text'][:100]}...")
                        if text_log_file:
                            text_log_file.write(json.dumps({
                                "video": key, "text": r["decoded_text"],
                                "token_mapping": r["token_mapping"],
                            }, ensure_ascii=False) + "\n")
                        if r["embedding"] is not None:
                            results[key] = np.asarray(r["embedding"], dtype=np.float32)

                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"GPU {gpu_id} failed on {video_path}: {e}")
    finally:
        if text_log_file:
            text_log_file.close()
        np.savez_compressed(output_file, **results)
        logger.info(f"GPU {gpu_id} saved {len(results)} embeddings to {output_file}")


def generate_embeddings(cfg: Config):
    video_paths = list_videos(cfg.video_dir)

    folder = "raw_hidden"
    video_paths = [
        vp for vp in video_paths
        if not os.path.exists(os.path.join(cfg.artifact_dir, folder, f"{os.path.basename(vp)}.pt"))
    ]
    logger.info(f"Processing {len(video_paths)} videos")

    os.makedirs(cfg.artifact_dir, exist_ok=True)

    if cfg.parallelism == "mp":
        logger.info("Using model parallelism (single process)")
        processor = VideoProcessor(cfg)
        processor.load_model(0)
        return _generate_single_gpu(cfg, processor, video_paths)

    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        processor = VideoProcessor(cfg)
        processor.load_model(0)
        return _generate_single_gpu(cfg, processor, video_paths)

    chunks = [video_paths[i::num_gpus] for i in range(num_gpus)]
    logger.info(f"Using {num_gpus} GPUs for data parallelism")
    for i, chunk in enumerate(chunks):
        logger.info(f"GPU {i}: {len(chunk)} videos")

    output_files = [f"{cfg.output_npz_path}.gpu{gid}.npz" for gid in range(num_gpus)]
    processes = []
    for gid, chunk in enumerate(chunks):
        if chunk:
            p = mp.Process(target=process_videos_on_gpu, args=(gid, chunk, cfg, output_files[gid]))
            processes.append(p)
            p.start()

    for p in processes:
        p.join()

    time.sleep(3)
    all_results = {}
    for ofile in output_files:
        if os.path.exists(ofile):
            data = np.load(ofile)
            all_results.update(dict(data))

    if all_results:
        np.savez_compressed(cfg.output_npz_path, **all_results)
        logger.info(f"Merged {len(all_results)} embeddings → {cfg.output_npz_path}")

    _post_run_stats(video_paths, all_results)
    _merge_text_logs(cfg, num_gpus)

    return all_results


def _generate_single_gpu(cfg: Config, processor: VideoProcessor, video_paths):
    extractor = EmbeddingExtractor(cfg)
    prompt = get_prompts(cfg.prompt_style)["reference"]()

    dataset = VideoDataset(video_paths, prompt, cfg.sample_fps)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers if cfg.parallelism == "mp" else 0,
        collate_fn=lambda b: collate_fn(b, processor.processor),
        pin_memory=False,
    )

    results = {}
    text_log_file = None
    if cfg.text_log_path:
        os.makedirs(os.path.dirname(cfg.text_log_path) or ".", exist_ok=True)
        text_log_file = open(cfg.text_log_path, "w", encoding="utf-8")

    try:
        for inputs, vpaths in tqdm(loader, desc="Generating embeddings"):
            try:
                batch_results = extractor.extract_embeddings(
                    inputs, vpaths, processor,
                    temperature=cfg.temperature, top_p=cfg.top_p,
                    top_k=cfg.top_k, do_sample=cfg.do_sample,
                )
                for r in batch_results:
                    key = r["video_name"]
                    logger.debug(f"[{key}] => {r['decoded_text'][:100]}...")
                    if text_log_file:
                        text_log_file.write(json.dumps({
                            "video": key, "text": r["decoded_text"],
                            "token_mapping": r["token_mapping"],
                        }, ensure_ascii=False) + "\n")
                    if r["embedding"] is not None:
                        results[key] = np.asarray(r["embedding"], dtype=np.float32)
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Failed on batch: {e}")
                torch.cuda.empty_cache()
    finally:
        if text_log_file:
            text_log_file.close()

    if results:
        np.savez_compressed(cfg.output_npz_path, **results)
        logger.info(f"Saved {len(results)} embeddings → {cfg.output_npz_path}")

    _post_run_stats(video_paths, results)
    return results


def _post_run_stats(video_paths, results):
    logger.info(f"Total input: {len(video_paths)}, processed: {len(results)}, "
                f"success: {len(results) / max(len(video_paths), 1) * 100:.1f}%")
    if results:
        sample = next(iter(results.values()))
        logger.info(f"Embedding dim: {sample.shape[0]}, "
                     f"total size: {sum(e.nbytes for e in results.values()) / 1024 / 1024:.1f} MB")


def _merge_text_logs(cfg: Config, num_gpus: int):
    if not cfg.text_log_path:
        return
    combined = {}
    for gid in range(num_gpus):
        path = f"{cfg.text_log_path}.gpu{gid}"
        if os.path.exists(path):
            with open(path, "r") as f:
                for line in f:
                    data = json.loads(line.strip())
                    combined[data["video"]] = data["text"]
    if combined:
        with open(cfg.text_log_path, "w") as f:
            for video, text in combined.items():
                f.write(json.dumps({"video": video, "text": text}) + "\n")
        logger.info(f"Merged {len(combined)} text entries → {cfg.text_log_path}")


def main():
    cfg = load_config_from_args()

    if cfg.parallelism == "dp":
        mp.set_start_method("spawn", force=True)

    start = time.time()
    logger.info("Starting video embedding generation")
    generate_embeddings(cfg)
    logger.info(f"Total time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
