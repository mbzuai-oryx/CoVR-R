import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def list_videos(video_dir: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(video_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                paths.append(os.path.join(root, f))
    return paths


class VideoDataset(Dataset):
    def __init__(self, video_paths: List[str], prompt: str, sample_fps: float):
        self.video_paths = video_paths
        self.prompt = prompt
        self.sample_fps = sample_fps

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        return {
            "video_path": self.video_paths[idx],
            "prompt": self.prompt,
            "sample_fps": self.sample_fps,
        }


def collate_fn(batch, processor):
    video_paths = [item["video_path"] for item in batch]
    prompt = batch[0]["prompt"]
    fps = batch[0]["sample_fps"]

    all_texts, all_images, all_videos = [], [], []
    for vp in video_paths:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"file://{os.path.abspath(vp)}", "fps": float(fps)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        images, videos, _ = process_vision_info(messages, return_video_kwargs=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_texts.append(text)
        all_images.extend(images or [])
        all_videos.extend(videos or [])

    inputs = processor(
        text=all_texts,
        images=all_images if all_images else None,
        videos=all_videos if all_videos else None,
        return_tensors="pt",
        padding=True,
    )
    return inputs, video_paths


@dataclass
class Record:
    index: int
    reference_token: str
    target_token: str
    edit_instruction: str
    video_path: str
    video_extension: str = ".mp4"

    @property
    def target_key(self) -> str:
        return f"{self.target_token}{self.video_extension}"

    @property
    def reference_key(self) -> str:
        return f"{self.reference_token}{self.video_extension}"


def load_records(label_path: str, video_dir: str, limit: Optional[int] = None) -> List[Record]:
    ext = os.path.splitext(label_path)[1].lower()
    if ext == ".json":
        return _load_json_records(label_path, video_dir, limit)
    else:
        return _load_csv_records(label_path, video_dir, limit)


def _load_csv_records(path: str, video_dir: str, limit: Optional[int]) -> List[Record]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            if limit is not None and len(records) >= limit:
                break
            pth1 = (row.get("pth1") or "").strip()
            pth2 = (row.get("pth2") or "").strip()
            edit = (row.get("edit") or "").strip()

            ref_token = os.path.basename(pth1).split(".")[0]
            tgt_token = os.path.basename(pth2).split(".")[0]
            video_path = os.path.join(video_dir, f"{ref_token}.mp4")

            records.append(Record(
                index=int(row.get("index", row_idx)),
                reference_token=ref_token,
                target_token=tgt_token,
                edit_instruction=edit,
                video_path=video_path,
            ))
    return records


def _load_json_records(path: str, video_dir: str, limit: Optional[int]) -> List[Record]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect merged format: [{"webvid": [...]}, {"ss2": [...]}]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        first_keys = set(data[0].keys())
        if first_keys & {"webvid", "ss2"}:
            return _load_merged_json_records(data, video_dir, limit)

    if isinstance(data, dict):
        data = list(data.values())

    records = []
    for row_idx, row in enumerate(data):
        if limit is not None and len(records) >= limit:
            break
        src = str(row.get("video_source") or "").strip()
        tgt = str(row.get("video_target") or "").strip()
        edit = (row.get("modification_text") or "").strip()

        ref_token = os.path.basename(src).split(".")[0]
        tgt_token = os.path.basename(tgt).split(".")[0]
        ext = row.get("video_extension", ".mp4")
        if not ext.startswith("."):
            ext = f".{ext}"
        video_path = os.path.join(video_dir, f"{ref_token}{ext}")

        records.append(Record(
            index=int(row.get("index", row_idx)),
            reference_token=ref_token,
            target_token=tgt_token,
            edit_instruction=edit,
            video_path=video_path,
            video_extension=ext,
        ))
    return records


def _load_merged_json_records(data: list, video_dir: str, limit: Optional[int]) -> List[Record]:
    records = []
    for section in data:
        for section_key, entries in section.items():
            if section_key == "webvid":
                ext = ".mp4"
            elif section_key == "ss2":
                ext = ".webm"
            else:
                ext = ".mp4"

            for row_idx, row in enumerate(entries):
                if limit is not None and len(records) >= limit:
                    return records

                src = str(row.get("video_source", "")).strip()
                tgt = str(row.get("video_target", "")).strip()
                edit = (row.get("modification_text") or "").strip()

                ref_token = os.path.basename(src).split(".")[0]
                tgt_token = os.path.basename(tgt).split(".")[0]
                video_path = os.path.join(video_dir, f"{ref_token}{ext}")

                records.append(Record(
                    index=int(row.get("id", row_idx)),
                    reference_token=ref_token,
                    target_token=tgt_token,
                    edit_instruction=edit,
                    video_path=video_path,
                    video_extension=ext,
                ))
    return records
