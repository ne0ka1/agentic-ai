from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
from dataclasses import dataclass

import cv2
from langchain_core.messages import HumanMessage

try:
    from langchain_ollama import ChatOllama
except ImportError as exc:
    raise ImportError(
        "This script requires langchain-ollama. Install with: pip install langchain-ollama"
    ) from exc


PROMPT = (
    "Look at this surveillance frame. Is at least one human person visible anywhere "
    "(full body or partial: head, torso, limbs)? Answer generously: smooth skin, "
    "simple shapes, or video compression artifacts are still people unless it is "
    "clearly a poster, tiny toy, or an obvious store mannequin on a stand. "
    "Reply with ONLY one JSON object, no markdown or extra text. Keys: person_present "
    "(boolean), confidence (0.0–1.0), reason (short string). "
    "Set person_present to false only if you are confident there is no real person."
)


@dataclass
class DetectionEvent:
    event: str  # "enter" | "exit"
    time_seconds: float


def prepare_frame(frame, max_side: int, min_side: int):
    """Resize: upscale tiny frames (distant people), downscale huge ones for speed."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if min_side > 0 and longest < min_side:
        scale = min_side / longest
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        h, w = frame.shape[:2]
        longest = max(h, w)
    if max_side > 0 and longest > max_side:
        scale = max_side / longest
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return frame


def to_data_url(frame_bgr, jpeg_quality: int) -> str:
    ok, buf = cv2.imencode(
        ".jpg",
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    mime = mimetypes.types_map.get(".jpg", "image/jpeg")
    return f"data:{mime};base64,{b64}"


def parse_person_present(raw: str) -> tuple[bool, float, str, str]:
    """Parse model output into (person_present, confidence, reason, parse_note)."""
    text = raw.strip()

    def from_obj(obj: dict) -> tuple[bool, float, str] | None:
        try:
            present = obj.get("person_present")
            if present is None:
                return None
            if isinstance(present, str):
                pl = present.strip().lower()
                if pl in ("true", "yes", "1"):
                    present = True
                elif pl in ("false", "no", "0"):
                    present = False
                else:
                    return None
            conf = float(obj.get("confidence", 0.5))
            reason = str(obj.get("reason", ""))
            return bool(present), max(0.0, min(1.0, conf)), reason
        except Exception:
            return None

    try:
        got = from_obj(json.loads(text))
        if got is not None:
            p, c, r = got
            return p, c, r, "json"
    except Exception:
        pass

    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if m:
        blob = m.group(0).replace("'", '"')
        try:
            got = from_obj(json.loads(blob))
            if got is not None:
                p, c, r = got
                return p, c, r, "json-substring"
        except Exception:
            pass

    lower = text.lower()
    # Strong negatives first (LLaVA often answers in plain language).
    if re.search(
        r"\b(no person|nobody|no one|no human|not visible|can't see anyone|cannot see anyone|"
        r"no people|empty (scene|room|frame)|no individuals)\b",
        lower,
    ):
        return False, 0.6, "", "heuristic-negative"

    if re.search(
        r"\b(yes\b|there (is|are) (a |one |some )?(person|people|human|someone|figure|individual)|"
        r"\ba person\b|\bperson (is |in |visible|present|standing|walking)|"
        r"\bsomeone\b|\bpeople\b|\bhuman (figure|body|silhouette))\b",
        lower,
    ):
        return True, 0.55, "", "heuristic-positive"

    if re.search(r'"person_present"\s*:\s*true\b', text):
        return True, 0.65, "", "regex-field"
    if re.search(r'"person_present"\s*:\s*false\b', text):
        return False, 0.65, "", "regex-field"

    return False, 0.4, "", "heuristic-fallback"


def format_ts(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    h = total_ms // 3_600_000
    total_ms %= 3_600_000
    m = total_ms // 60_000
    total_ms %= 60_000
    s = total_ms // 1000
    ms = total_ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def detect_events(
    video_path: str,
    model: str,
    base_url: str,
    frame_stride: int,
    enter_frames: int,
    exit_frames: int,
    temperature: float,
    request_timeout: float,
    num_predict: int,
    max_side: int,
    min_side: int,
    jpeg_quality: int,
    debug_raw: bool,
) -> tuple[list[DetectionEvent], float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0.0:
        cap.release()
        raise RuntimeError("Video FPS unavailable or invalid")

    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=request_timeout,
        num_predict=num_predict,
    )

    frame_idx = 0
    sampled_count = 0
    true_streak = 0
    false_streak = 0
    in_scene = False
    events: list[DetectionEvent] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        sampled_count += 1
        t = frame_idx / fps
        frame = prepare_frame(frame, max_side=max_side, min_side=min_side)
        data_url = to_data_url(frame, jpeg_quality=jpeg_quality)

        msg = HumanMessage(
            content=[
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
        reply = llm.invoke([msg])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        present, conf, reason, how = parse_person_present(text)

        if debug_raw:
            preview = text.replace("\n", " ")[: 400]
            print(f"  raw ({how}): {preview!r}")

        status = "PERSON" if present else "NO_PERSON"
        reason_out = reason or "—"
        print(f"[{format_ts(t)}] {status} conf={conf:.2f} [{how}] {reason_out}")

        if present:
            true_streak += 1
            false_streak = 0
            if (not in_scene) and true_streak >= enter_frames:
                in_scene = True
                events.append(DetectionEvent(event="enter", time_seconds=t))
        else:
            false_streak += 1
            true_streak = 0
            if in_scene and false_streak >= exit_frames:
                in_scene = False
                events.append(DetectionEvent(event="exit", time_seconds=t))

        frame_idx += 1

    duration_seconds = frame_idx / fps if fps > 0 else 0.0
    cap.release()

    if in_scene:
        events.append(DetectionEvent(event="exit", time_seconds=duration_seconds))

    return events, duration_seconds, sampled_count


def write_events(events: list[DetectionEvent], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        if not events:
            f.write("No enter/exit events found.\n")
            return
        for e in events:
            f.write(f"{e.event.upper()} {format_ts(e.time_seconds)} ({e.time_seconds:.3f}s)\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frame-by-frame person enter/exit detection using LLaVA via Ollama"
    )
    parser.add_argument("--video", default="video.mp4", help="Path to input video")
    parser.add_argument("--model", default="llava", help="Ollama model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Analyze every Nth frame (1 = all frames)",
    )
    parser.add_argument(
        "--enter-frames",
        type=int,
        default=1,
        help="Consecutive positive detections required for ENTER",
    )
    parser.add_argument(
        "--exit-frames",
        type=int,
        default=2,
        help="Consecutive negative detections required for EXIT",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature for deterministic classification",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=60.0,
        help="Per-frame LLaVA request timeout in seconds",
    )
    parser.add_argument(
        "--output",
        default="person_events.txt",
        help="Output file for enter/exit timestamps",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Max new tokens from Ollama (raise if JSON gets cut off)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Downscale frame if longer side exceeds this (0 = no downscale)",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=512,
        help="Upscale frame if longer side is below this (0 = no upscale; helps tiny people)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG quality 1–100 for frames sent to the model",
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print model reply snippet and parse path each frame",
    )
    args = parser.parse_args()

    if args.frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")
    if args.enter_frames < 1 or args.exit_frames < 1:
        raise ValueError("--enter-frames and --exit-frames must be >= 1")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be > 0")
    if args.num_predict < 8:
        raise ValueError("--num-predict must be >= 8")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be 1–100")
    if args.max_side < 0 or args.min_side < 0:
        raise ValueError("--max-side and --min-side must be >= 0 (use 0 to disable)")

    video_path = os.path.abspath(args.video)
    output_path = os.path.abspath(args.output)

    print(f"Video:  {video_path}")
    print(f"Model:  {args.model}")
    print(f"Output: {output_path}")

    events, duration, sampled = detect_events(
        video_path=video_path,
        model=args.model,
        base_url=args.base_url,
        frame_stride=args.frame_stride,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
        temperature=args.temperature,
        request_timeout=args.request_timeout,
        num_predict=args.num_predict,
        max_side=args.max_side,
        min_side=args.min_side,
        jpeg_quality=args.jpeg_quality,
        debug_raw=args.debug_raw,
    )

    write_events(events, output_path)

    print("\n=== Summary ===")
    print(f"Duration: {format_ts(duration)} ({duration:.3f}s)")
    print(f"Sampled frames: {sampled}")
    if not events:
        print("No person entry/exit events found.")
    else:
        for e in events:
            print(f"{e.event.upper():5s} at {format_ts(e.time_seconds)}")
        print(f"\nEvents written to: {output_path}")


if __name__ == "__main__":
    main()