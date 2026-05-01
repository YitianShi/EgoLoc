#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
HOI_PYTHON = Path.home() / "anaconda3" / "envs" / "hoi" / "bin" / "python"


def env_default(name, default):
    return os.environ.get(name, default)


def default_python():
    return str(HOI_PYTHON) if HOI_PYTHON.is_file() else sys.executable


def natural_key(path):
    text = str(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def require_egoloc(egoloc_dir):
    if not egoloc_dir.is_dir():
        raise FileNotFoundError(f"EgoLoc directory not found: {egoloc_dir}")

    runner = egoloc_dir / "egoloc2d_long.py"
    if not runner.is_file():
        raise FileNotFoundError(f"EgoLoc runner not found: {runner}")


def resolve_credentials_path(credentials, egoloc_dir):
    credentials_path = Path(credentials).expanduser()
    if credentials_path.is_absolute():
        return credentials_path
    return egoloc_dir / credentials_path


def read_env_file_value(path, key):
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        value = value.strip().strip("\"'")
        return value or None
    return None


def require_openai_api_key(args):
    if os.environ.get("OPENAI_API_KEY"):
        print("[EgoLoc] Found OPENAI_API_KEY in environment.")
        return

    credentials_path = resolve_credentials_path(args.credentials, args.egoloc_dir)
    if credentials_path.is_file():
        api_key = read_env_file_value(credentials_path, "OPENAI_API_KEY")
        if api_key:
            print(f"[EgoLoc] Found OPENAI_API_KEY in credentials file: {credentials_path}")
            return

    raise RuntimeError(
        "OPENAI_API_KEY is required. Export it in the environment or set it in "
        f"the credentials file: {credentials_path}"
    )


def resolve_video_path(target, data_root, camera_video):
    target_path = Path(target)
    if str(target).isdigit():
        return data_root / str(target) / camera_video
    if target_path.is_dir():
        return target_path / camera_video
    return target_path


def default_output_dir(video_path, camera_video, output_dirname):
    video_dir = video_path.parent.resolve()
    if video_path.name == camera_video:
        return video_dir / output_dirname
    return video_dir / "egoloc_output"


def default_summary_path(output_dir, summary_filename):
    return output_dir / summary_filename


def speed_json_name(video_path, pattern):
    return pattern.format(video_stem=video_path.stem, video_name=video_path.name)


def resolve_speed_json(args, video_path, output_dir):
    if args.speed_json:
        speed_json = Path(args.speed_json).expanduser().resolve()
        if speed_json.is_dir():
            return speed_json / speed_json_name(video_path, args.speed_json_pattern)
        return speed_json

    if args.speed_json_dir:
        return Path(args.speed_json_dir).expanduser().resolve() / speed_json_name(
            video_path, args.speed_json_pattern
        )

    candidates = [
        output_dir / speed_json_name(video_path, args.speed_json_pattern),
        video_path.parent / "speed" / speed_json_name(video_path, args.speed_json_pattern),
        video_path.parent.parent / "speed" / speed_json_name(video_path, args.speed_json_pattern),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def normalize_speed_json_for_long_runner(speed_json, output_dir, video_name, write=True):
    if speed_json is None:
        return None

    with speed_json.open() as f:
        speed_data = json.load(f)

    if isinstance(speed_data, list):
        if all(isinstance(item, list) and len(item) == 2 for item in speed_data):
            return speed_json
        raise ValueError(f"Unsupported speed JSON list format: {speed_json}")

    if not isinstance(speed_data, dict):
        raise ValueError(f"Unsupported speed JSON format: {speed_json}")

    frame_speed_pairs = [(int(frame), speed) for frame, speed in speed_data.items()]
    frame_ids = [frame for frame, _ in frame_speed_pairs]
    frame_offset = 1 if frame_ids and min(frame_ids) == 0 else 0
    normalized = [
        [frame + frame_offset, speed]
        for frame, speed in sorted(frame_speed_pairs)
    ]

    normalized_path = output_dir / f"{video_name}_speed_egoloc_long.json"
    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        with normalized_path.open("w") as f:
            json.dump(normalized, f, indent=2)
        print(f"[EgoLoc] Converted speed JSON for long runner: {normalized_path}")
    else:
        print(f"[EgoLoc] Would convert speed JSON for long runner: {normalized_path}")
    return normalized_path


def discover_sample_dirs(data_root):
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    return sorted((path for path in data_root.iterdir() if path.is_dir()), key=natural_key)


def collect_jobs(args):
    jobs = []
    if not args.targets:
        for sample_dir in discover_sample_dirs(args.data_root):
            video_path = sample_dir / args.camera_video
            output_dir = sample_dir / args.output_dirname
            jobs.append((video_path, output_dir))
        return jobs

    for target in args.targets:
        video_path = resolve_video_path(target, args.data_root, args.camera_video)
        output_dir = default_output_dir(video_path, args.camera_video, args.output_dirname)
        jobs.append((video_path, output_dir))
    return jobs


def copy_summary(result_path, summary_path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(result_path, summary_path)
    print(f"[EgoLoc] Contact summary saved: {summary_path}")


def build_egoloc_command(args, video_path, output_dir, speed_json):
    command = [
        args.python,
        "egoloc2d_long.py",
        "--video_path",
        str(video_path),
        "--output_dir",
        str(output_dir),
        "--credentials",
        args.credentials,
        "--device",
        args.device,
        "--encoder",
        args.encoder,
        "--grid_size",
        str(args.grid_size),
        "--max_feedback",
        str(args.max_feedbacks),
        "--video_type",
        args.video_type,
    ]
    if speed_json is not None:
        command.extend(["--speed_json", str(speed_json)])
    return command


def print_dry_run(args, command, result_path, summary_path):
    command_text = " ".join(
        [
            f"cd {shlex.quote(str(args.egoloc_dir))}",
            "&&",
            f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)}",
            shlex.join(command),
        ]
    )
    print(command_text)
    print(f"cp {shlex.quote(str(result_path))} {shlex.quote(str(summary_path))}")


def run_video(args, video_path, output_dir):
    if not video_path.is_file():
        print(f"[EgoLoc] Missing video, skipping: {video_path}")
        return

    video_path = video_path.resolve()
    output_dir = output_dir.resolve()
    video_name = video_path.stem
    result_path = output_dir / f"{video_name}_result.json"
    summary_path = default_summary_path(output_dir, args.summary_filename)
    raw_speed_json = resolve_speed_json(args, video_path, output_dir)

    print()
    print(f"===== EgoLoc long: {video_path} =====")
    print(f"Output: {output_dir}")
    print(f"Contact summary: {summary_path}")
    if raw_speed_json is not None:
        print(f"Speed JSON: {raw_speed_json}")
    else:
        print("[EgoLoc] No precomputed speed JSON found; EgoLoc will compute speed from video.")

    if not args.overwrite and result_path.is_file():
        print(f"[EgoLoc] Existing result, skipping: {result_path}")
        copy_summary(result_path, summary_path)
        return

    speed_json = normalize_speed_json_for_long_runner(
        raw_speed_json,
        output_dir,
        video_name,
        write=not args.dry_run,
    )
    command = build_egoloc_command(args, video_path, output_dir, speed_json)

    if args.dry_run:
        print_dry_run(args, command, result_path, summary_path)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    subprocess.run(command, cwd=args.egoloc_dir, env=env, check=True)

    if not result_path.is_file():
        raise FileNotFoundError(f"Expected result missing: {result_path}")

    copy_summary(result_path, summary_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EgoLoc long-video localization on HOMimic camera videos."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="sample ids, sample directories, or direct video paths. If empty, runs every sample under DATA_ROOT.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(env_default("DATA_ROOT", str(Path.home() / "Datasets/HOMimic_recorded_data"))),
        help="dataset root containing sample folders",
    )
    parser.add_argument("--camera-video", default=env_default("CAMERA_VIDEO", "camera_static.mp4"))
    parser.add_argument(
        "--egoloc-dir",
        type=Path,
        default=Path(env_default("EGOLOC_DIR", str(REPO_ROOT))),
    )
    parser.add_argument("--output-dirname", default=env_default("OUTPUT_DIRNAME", "egoloc"))
    parser.add_argument("--summary-filename", default=env_default("SUMMARY_FILENAME", "vlm_hoi_egoloc.txt"))
    parser.add_argument("--python", default=env_default("PYTHON", default_python()))
    parser.add_argument("--cuda-visible-devices", default=env_default("CUDA_VISIBLE_DEVICES", "1"))
    parser.add_argument("--device", default=env_default("DEVICE", "cuda"))
    parser.add_argument(
        "--credentials",
        default=env_default("CREDENTIALS", "auth.env"),
        help="fallback .env credentials file; OPENAI_API_KEY environment variable takes precedence",
    )
    parser.add_argument("--speed-json", default=env_default("SPEED_JSON", None))
    parser.add_argument("--speed-json-dir", default=env_default("SPEED_JSON_DIR", None))
    parser.add_argument(
        "--speed-json-pattern",
        default=env_default("SPEED_JSON_PATTERN", "{video_stem}_speed.json"),
        help="filename pattern used with --speed-json-dir and automatic speed JSON discovery",
    )
    parser.add_argument("--encoder", choices=["vits", "vitl"], default=env_default("ENCODER", "vits"))
    parser.add_argument("--video-type", choices=["short", "long"], default=env_default("VIDEO_TYPE", "long"))
    parser.add_argument("--grid-size", type=int, default=int(env_default("GRID_SIZE", "2")))
    parser.add_argument("--max-feedbacks", type=int, default=int(env_default("MAX_FEEDBACKS", "1")))
    parser.add_argument("--overwrite", action="store_true", default=env_default("OVERWRITE", "0") == "1")
    parser.add_argument("--dry-run", action="store_true", default=env_default("DRY_RUN", "0") == "1")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=env_default("CONTINUE_ON_ERROR", "0") == "1",
        help="continue to the next video if one sample fails",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.egoloc_dir = args.egoloc_dir.resolve()
    args.data_root = args.data_root.resolve()

    require_egoloc(args.egoloc_dir)
    require_openai_api_key(args)

    jobs = collect_jobs(args)
    print(f"[EgoLoc] Found {len(jobs)} video job(s).")

    if args.dry_run:
        for video_path, output_dir in jobs:
            run_video(args, video_path, output_dir)
        return

    for video_path, output_dir in jobs:
        try:
            run_video(args, video_path, output_dir)
        except Exception:
            if not args.continue_on_error:
                raise
            print(f"[EgoLoc] Failed on {video_path}; continuing.")
            traceback.print_exc()


if __name__ == "__main__":
    main()
