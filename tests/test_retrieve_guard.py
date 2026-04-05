import asyncio
import importlib.util
import tempfile
import unittest
from pathlib import Path


HAS_AIOFILES = importlib.util.find_spec("aiofiles") is not None


@unittest.skipUnless(HAS_AIOFILES, "aiofiles is required for knowledge base tests")
class RetrieveGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        from backend.config import settings

        self.settings = settings
        self.original_values = {
            "allow_full_retrieval": settings.allow_full_retrieval,
            "max_retrieve_paths": settings.max_retrieve_paths,
            "max_retrieve_files": settings.max_retrieve_files,
            "max_retrieve_chars": settings.max_retrieve_chars,
        }

        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        (self.base_path / "safe.md").write_text("# safe", encoding="utf-8")

        settings.allow_full_retrieval = False
        settings.max_retrieve_paths = 10
        settings.max_retrieve_files = 10
        settings.max_retrieve_chars = 20000

    def tearDown(self) -> None:
        for key, value in self.original_values.items():
            setattr(self.settings, key, value)
        self.temp_dir.cleanup()

    def test_rejects_path_escape(self) -> None:
        from backend.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(base_path=str(self.base_path))
        with self.assertRaises(ValueError):
            asyncio.run(kb.retrieve_files(["../../etc/passwd"]))

    def test_rejects_full_retrieval_when_disabled(self) -> None:
        from backend.knowledge_base import KnowledgeBase

        kb = KnowledgeBase(base_path=str(self.base_path))
        with self.assertRaises(ValueError):
            asyncio.run(kb.retrieve_files(["/"]))


if __name__ == "__main__":
    unittest.main()
