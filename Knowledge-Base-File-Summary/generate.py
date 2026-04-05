#!/usr/bin/env python3
"""
Knowledge Base Summary Generator.

Pipeline capabilities:
- Reproducible run metadata (manifest + source digest)
- File-level retry rounds for partial failures
- Versioned artifacts per run ID
"""

import argparse
import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Union

import httpx
import tiktoken

from backend.config import settings

# Large file splitting configuration
MAX_SUMMARY_TOKENS = 10
MAX_CHUNK_TOKENS = 3000
MIN_CHUNK_TOKENS = 1000
TOKEN_DISPLAY_INTERVAL = 100
GENERATOR_VERSION = "2.0.0"

SummaryValue = Union[str, List[dict]]


@dataclass
class PipelineOptions:
    max_retries: int = 5
    retry_rounds: int = 3
    request_delay_seconds: float = 0.5
    run_id: str = ""
    artifacts_dir: str = ""
    cache_file: str = ""
    strict: bool = False
    verbose_prompts: bool = False


class SummaryGenerator:
    def __init__(self, options: PipelineOptions):
        self.options = options
        self.base_path = Path(settings.knowledge_base)
        self.chunks_dir = Path(settings.knowledge_base_chunks)
        self.output_file = Path(settings.knowledge_base_file_summary)

        # Stable cache for resume across runs
        default_cache = self.output_file.parent / "summary_cache.json"
        self.cache_file = Path(options.cache_file) if options.cache_file else default_cache

        # Versioned artifacts directory
        default_artifacts_root = self.output_file.parent / "artifacts"
        self.artifacts_root = (
            Path(options.artifacts_dir) if options.artifacts_dir else default_artifacts_root
        )
        self.run_id = options.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.artifacts_root / self.run_id

        # Initialize tokenizer
        self.encoder = tiktoken.get_encoding("cl100k_base")

        # Get LLM configuration
        self.provider = settings.api_provider
        self.config = settings.get_provider_config(self.provider)

        # API configuration
        self.base_url = self.config.get("base_url", "")
        if "/chat/completions" in self.base_url:
            self.api_url = self.base_url
        else:
            self.api_url = f"{self.base_url}/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            **self.config.get("headers", {}),
        }
        if self.config.get("api_key"):
            self.headers["Authorization"] = f"Bearer {self.config['api_key']}"

        # Load cache
        self.cache = self._load_cache()

    async def _generate_file_summary(self, file_path: Path, relative_path: str) -> SummaryValue | None:
        """
        Generate summary for a single file using LLM.

        Small files: return string summary.
        Large files: return list[dict] chunk summaries.
        """
        if relative_path in self.cache:
            print(f"✅ Using cached summary: {relative_path}")
            return self.cache[relative_path]

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            total_tokens = self._count_tokens(content)

            if total_tokens > MAX_CHUNK_TOKENS:
                lines = content.split("\n")
                num_chunks = (total_tokens // MAX_CHUNK_TOKENS) + 1

                lines_with_info: List[str] = []
                cumulative_tokens = 0
                next_mark = 0
                for i, line in enumerate(lines, 1):
                    show_count = cumulative_tokens >= next_mark
                    prefix = f"{i}({cumulative_tokens})" if show_count else str(i)
                    lines_with_info.append(f"{prefix} {line}")
                    if show_count:
                        next_mark = (
                            ((cumulative_tokens // TOKEN_DISPLAY_INTERVAL) + 1)
                            * TOKEN_DISPLAY_INTERVAL
                        )
                    cumulative_tokens += self._count_tokens(line)

                content_with_lines = "\n".join(lines_with_info)

                prompt = f"""- Task: Split the following large file into {num_chunks} small files and provide a content summary for each
- Goal: When users ask questions, let the LLM understand the general content of the file to decide whether to read all the content of that file among many file content summaries
- Length: {MIN_CHUNK_TOKENS} < small file token count < {MAX_CHUNK_TOKENS}, content summary token count < {MAX_SUMMARY_TOKENS}
- Large file: {relative_path}
- Output format: Single-line JSON without code blocks, format like [{{"start": start_line_number, "end": end_line_number, "summary": "content_summary"}}]
- Content summary: Forbidden to include words from file path, low information density words (such as the, it, this, outline, document)
- Important! Do not brutally truncate semantically complete paragraphs and sentences!!!
- Important! Try to put key information directly in the content summary!!!

> Below is the large file content, each line's left number is line number, number in parentheses is cumulative token count

{content_with_lines}

""".strip()
            else:
                lines = []
                prompt = f"""- Task: Provide a file content summary
- Goal: When users ask questions, let the LLM understand the general content of the file to decide whether to read all the content of that file among many file content summaries
- Length: Content summary token count < {MAX_SUMMARY_TOKENS}
- File path: {relative_path}
- Content summary: Forbidden to include words from file path, low information density words (such as the, it, this, outline, document)
- Important! Try to put key information directly in the content summary!!!

> Below is the file content

{content}

""".strip()

            if self.options.verbose_prompts:
                print(f"\n[Prompt for {relative_path}]\n{prompt}\n")

            messages = [{"role": "user", "content": prompt}]

            for attempt in range(self.options.max_retries):
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        payload = {
                            "model": self.config["model"],
                            "messages": messages,
                            "temperature": 0,
                            "stream": False,
                        }

                        if "gpt-5" in self.config["model"] or "o1" in self.config["model"]:
                            payload["max_completion_tokens"] = settings.max_tokens
                        else:
                            payload["max_tokens"] = settings.max_tokens

                        response = await client.post(self.api_url, json=payload, headers=self.headers)
                        response.raise_for_status()

                        data = response.json()
                        if "choices" not in data or len(data["choices"]) == 0:
                            print(f"⚠️  Empty response for {relative_path}")
                            return None

                        result = data["choices"][0]["message"].get("content", "").strip()

                        if total_tokens > MAX_CHUNK_TOKENS:
                            chunks = self._parse_chunk_payload(result)
                            if chunks is None:
                                print(f"⚠️  Failed to parse JSON chunks for {relative_path}")
                                return None

                            self._save_chunk_files(file_path, chunks, lines)
                            self.cache[relative_path] = chunks
                            self._save_cache()
                            print(
                                f"✅ Split and summarized ({total_tokens} tokens -> {len(chunks)} chunks): {relative_path}"
                            )
                            return chunks

                        summary = " ".join(result.replace("\n", " ").replace("\r", " ").split())
                        self._copy_small_file(file_path)
                        self.cache[relative_path] = summary
                        self._save_cache()
                        print(f"✅ Generated summary ({total_tokens} tokens): {relative_path}")
                        return summary

                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    wait_time = (2**attempt) * 0.5
                    print(
                        f"⚠️  Attempt {attempt + 1}/{self.options.max_retries} failed for {relative_path}: {e}"
                    )
                    if attempt < self.options.max_retries - 1:
                        print(f"   Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"❌ Max retries reached for {relative_path}")
                        return None

        except Exception as e:
            print(f"❌ Error processing {relative_path}: {e}")
            return None

    async def _process_files_with_delay(self, files: List[Tuple[Path, str]]) -> Dict[str, SummaryValue]:
        summaries: Dict[str, SummaryValue] = {}
        tasks = []

        for i, (file_path, relative_path) in enumerate(files):
            async def delayed_task(fp: Path, rp: str, delay: float):
                await asyncio.sleep(delay)
                return rp, await self._generate_file_summary(fp, rp)

            task = delayed_task(file_path, relative_path, i * self.options.request_delay_seconds)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"❌ Task failed: {result}")
                continue

            if result:
                rel_path, summary = result
                if summary:
                    summaries[rel_path] = summary

        return summaries

    def _parse_chunk_payload(self, raw_text: str) -> List[dict] | None:
        result = raw_text.strip()

        if result.startswith("```"):
            result = result.split("\n", 1)[1]
        if result.endswith("```"):
            result = result.rsplit("\n", 1)[0]

        start = result.find("[")
        end = result.rfind("]")
        if start == -1 or end == -1 or end < start:
            return None

        try:
            parsed = json.loads(result[start : end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, list):
            return None

        return parsed

    def _count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))

    def _save_chunk_files(self, file_path: Path, chunks: List[dict], lines: List[str]):
        file_stem = file_path.stem
        parent_path = file_path.parent.relative_to(self.base_path)
        chunk_base_dir = self.chunks_dir / parent_path / file_stem
        chunk_base_dir.mkdir(parents=True, exist_ok=True)

        for chunk in chunks:
            start = int(chunk["start"])
            end = int(chunk["end"])
            chunk_lines = lines[start - 1 : end]
            chunk_content = "\n".join(chunk_lines)
            chunk_file = chunk_base_dir / f"{start}-{end}.md"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk_content)

    def _copy_small_file(self, file_path: Path):
        parent_path = file_path.parent.relative_to(self.base_path)
        target_dir = self.chunks_dir / parent_path
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / file_path.name

        with open(file_path, "r", encoding="utf-8") as src:
            content = src.read()
        with open(target_file, "w", encoding="utf-8") as dst:
            dst.write(content)

    def _collect_all_files(self) -> List[Tuple[Path, str]]:
        files: List[Tuple[Path, str]] = []
        for md_file in self.base_path.rglob("*.md"):
            relative_path = str(md_file.relative_to(self.base_path))
            files.append((md_file, relative_path))
        return sorted(files, key=lambda item: item[1])

    def _scan_directory(
        self, path: Path, prefix: str = "", summaries: Dict[str, SummaryValue] | None = None
    ) -> List[str]:
        lines: List[str] = []
        summaries = summaries or {}
        if not path.exists():
            return lines

        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        for i, item in enumerate(items):
            if item.name.startswith("."):
                continue

            is_last_item = i == len(items) - 1
            connector = "└──" if is_last_item else "├──"

            if item.is_dir():
                lines.append(f"{prefix}{connector} {item.name}")
                extension = "   " if is_last_item else "│  "
                sub_lines = self._scan_directory(item, prefix + extension, summaries)
                lines.extend(sub_lines)
                if not is_last_item and sub_lines:
                    lines.append(f"{prefix}│")
                continue

            file_relative = str(item.relative_to(self.base_path))
            file_summary = summaries.get(file_relative, "")
            if isinstance(file_summary, list):
                lines.append(f"{prefix}{connector} {item.stem}")
                chunk_extension = "   " if is_last_item else "│  "
                for chunk_idx, chunk in enumerate(file_summary):
                    is_last_chunk = chunk_idx == len(file_summary) - 1
                    chunk_connector = "└──" if is_last_chunk else "├──"
                    start = chunk.get("start", "?")
                    end = chunk.get("end", "?")
                    summary = chunk.get("summary", "")
                    lines.append(
                        f"{prefix}{chunk_extension}{chunk_connector} {start}-{end}.md：{summary}"
                    )
            elif file_summary:
                lines.append(f"{prefix}{connector} {item.name}：{file_summary}")
            else:
                lines.append(f"{prefix}{connector} {item.name}")

        return lines

    def _load_cache(self) -> Dict[str, SummaryValue]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        return loaded
            except Exception as e:
                print(f"⚠️  Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def _compute_source_digest(self, files: List[Tuple[Path, str]]) -> str:
        digest = hashlib.sha256()
        for file_path, relative_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                digest.update(relative_path.encode("utf-8"))
                digest.update(file_hash.encode("utf-8"))
            except Exception as e:
                digest.update(f"{relative_path}:read_error:{e}".encode("utf-8"))
        return digest.hexdigest()

    def _write_artifacts(
        self,
        output_content: str,
        total_files: int,
        summarized_files: int,
        total_kb_tokens: int,
        output_tokens: int,
        elapsed: float,
        failed_paths: List[str],
        source_digest: str,
    ):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

        run_summary_file = self.run_dir / "summary.txt"
        run_cache_file = self.run_dir / "summary_cache.json"
        run_manifest_file = self.run_dir / "manifest.json"

        run_summary_file.write_text(output_content, encoding="utf-8")
        with open(run_cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

        if failed_paths:
            failed_file = self.run_dir / "failed_files.txt"
            failed_file.write_text("\n".join(failed_paths) + "\n", encoding="utf-8")

        manifest = {
            "generator_version": GENERATOR_VERSION,
            "run_id": self.run_id,
            "generated_at": datetime.now().isoformat(),
            "provider": self.provider,
            "model": self.config.get("model"),
            "knowledge_base_path": str(self.base_path),
            "summary_output_file": str(self.output_file),
            "cache_file": str(self.cache_file),
            "source_digest": source_digest,
            "options": {
                "max_retries": self.options.max_retries,
                "retry_rounds": self.options.retry_rounds,
                "request_delay_seconds": self.options.request_delay_seconds,
                "strict": self.options.strict,
            },
            "stats": {
                "total_files": total_files,
                "summarized_files": summarized_files,
                "failed_files": len(failed_paths),
                "total_kb_tokens": total_kb_tokens,
                "output_tokens": output_tokens,
                "compression_ratio": (
                    (output_tokens / total_kb_tokens * 100) if total_kb_tokens > 0 else 0.0
                ),
                "duration_seconds": elapsed,
            },
        }
        with open(run_manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        latest_file = self.artifacts_root / "latest.json"
        latest_file.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "manifest": str(run_manifest_file),
                    "summary": str(run_summary_file),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    async def generate(self) -> bool:
        print("=" * 60)
        print("Knowledge Base Summary Generator")
        print("=" * 60)
        print(f"📁 Knowledge Base Path: {self.base_path}")
        print(f"📄 Output File: {self.output_file}")
        print(f"📦 Artifacts Root: {self.artifacts_root}")
        print(f"🧾 Run ID: {self.run_id}")
        print(f"🤖 LLM Provider: {self.provider}")
        print(f"🎯 Model: {self.config['model']}")
        print("=" * 60)

        print("\n📂 Collecting files...")
        files = self._collect_all_files()
        total_kb_tokens = 0
        for file_path, _ in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                total_kb_tokens += self._count_tokens(content)
            except Exception as e:
                print(f"⚠️  Failed to read {file_path}: {e}")

        source_digest = self._compute_source_digest(files)

        print(f"🔍 Found {len(files)} files")
        print(f"📊 Knowledge Base total tokens: {total_kb_tokens:,}")
        print(f"🔒 Source digest: {source_digest}")

        print("\n🔄 Generating file summaries...")
        start_time = time.time()
        file_summaries: Dict[str, SummaryValue] = {}
        pending_files = files

        for round_idx in range(1, self.options.retry_rounds + 1):
            if not pending_files:
                break

            print(
                f"\n🧪 Round {round_idx}/{self.options.retry_rounds} - pending: {len(pending_files)} files"
            )
            round_summaries = await self._process_files_with_delay(pending_files)
            file_summaries.update(round_summaries)
            pending_files = [
                (file_path, rel_path)
                for file_path, rel_path in pending_files
                if rel_path not in round_summaries
            ]

        elapsed = time.time() - start_time
        print(f"🎉 Generated {len(file_summaries)}/{len(files)} summaries in {elapsed:.1f}s")

        print("\n🌲 Building tree structure...")
        lines = ["."]
        lines.extend(self._scan_directory(self.base_path, summaries=file_summaries))
        output_content = "\n".join(lines)

        print(f"\n💾 Writing to {self.output_file}...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.output_file.write_text(output_content, encoding="utf-8")

        output_tokens = self._count_tokens(output_content)
        failed_paths = [rel_path for _, rel_path in pending_files]
        self._write_artifacts(
            output_content=output_content,
            total_files=len(files),
            summarized_files=len(file_summaries),
            total_kb_tokens=total_kb_tokens,
            output_tokens=output_tokens,
            elapsed=elapsed,
            failed_paths=failed_paths,
            source_digest=source_digest,
        )

        print("=" * 60)
        print("😊 Summary generation completed!")
        print(f"📊 Total files: {len(files)}")
        print(f"📊 File summaries: {len(file_summaries)}")
        print(f"📄 Output: {self.output_file}")
        print(f"📦 Artifact run dir: {self.run_dir}")
        print(f"📊 Output file tokens: {output_tokens:,}")
        if total_kb_tokens > 0:
            print(f"📊 Compression ratio: {(output_tokens / total_kb_tokens * 100):.2f}%")
        if failed_paths:
            print(f"⚠️  Failed files ({len(failed_paths)}):")
            for failed in failed_paths:
                print(f"   - {failed}")
        print("=" * 60)

        if self.options.strict and failed_paths:
            print("❌ Strict mode enabled and some files failed.")
            return False
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate knowledge base summary artifacts.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max HTTP retries for each file summary request.",
    )
    parser.add_argument(
        "--retry-rounds",
        type=int,
        default=3,
        help="How many rounds to retry pending files.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.5,
        help="Delay between launching concurrent file summary tasks.",
    )
    parser.add_argument("--run-id", type=str, default="", help="Custom artifact run ID.")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="",
        help="Artifacts root directory (default: Knowledge-Base-File-Summary/artifacts).",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="",
        help="Stable cache file path (default: Knowledge-Base-File-Summary/summary_cache.json).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when some files fail to summarize.",
    )
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Print raw prompts for troubleshooting.",
    )
    return parser.parse_args()


async def async_main() -> int:
    args = parse_args()
    options = PipelineOptions(
        max_retries=args.max_retries,
        retry_rounds=args.retry_rounds,
        request_delay_seconds=args.request_delay_seconds,
        run_id=args.run_id,
        artifacts_dir=args.artifacts_dir,
        cache_file=args.cache_file,
        strict=args.strict,
        verbose_prompts=args.verbose_prompts,
    )
    generator = SummaryGenerator(options)
    success = await generator.generate()
    return 0 if success else 1


def main():
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
