import importlib.util
import unittest


HAS_AIOFILES = importlib.util.find_spec("aiofiles") is not None


@unittest.skipUnless(HAS_AIOFILES, "aiofiles is required for prompts import")
class ReactParserTest(unittest.TestCase):
    def test_parse_structured_action_payload(self) -> None:
        from backend.prompts import parse_react_response

        text = (
            "<|Thought|> Need details.\n"
            '<|Action|> {"tool":"retrieve_files","input":{"file_paths":["a.md","b.md"]}}'
        )

        action, action_input, has_action = parse_react_response(text)
        self.assertTrue(has_action)
        self.assertEqual(action, "retrieve_files")
        self.assertEqual(action_input, {"file_paths": ["a.md", "b.md"]})

    def test_parse_legacy_action_payload(self) -> None:
        from backend.prompts import parse_react_response

        text = (
            "<|Thought|> Need details.\n"
            "<|Action|> retrieve_files\n"
            '<|Action Input|> {"file_paths":["legacy.md"]}'
        )

        action, action_input, has_action = parse_react_response(text)
        self.assertTrue(has_action)
        self.assertEqual(action, "retrieve_files")
        self.assertEqual(action_input, {"file_paths": ["legacy.md"]})

    def test_parse_legacy_action_payload_with_observation_json(self) -> None:
        from backend.prompts import parse_react_response

        text = (
            "<|Thought|> Need details.\n"
            "<|Action|> retrieve_files\n"
            '<|Action Input|> {"file_paths":["legacy.md"]}\n'
            '<|Observation|> {"status":"ok"}'
        )

        action, action_input, has_action = parse_react_response(text)
        self.assertTrue(has_action)
        self.assertEqual(action, "retrieve_files")
        self.assertEqual(action_input, {"file_paths": ["legacy.md"]})


if __name__ == "__main__":
    unittest.main()
