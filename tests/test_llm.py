import unittest
from unittest.mock import Mock, patch

from observatoire.llm import OllamaChatClient, _parse_json_content, make_llm_client


class LLMClientTest(unittest.TestCase):
    def test_ollama_client_calls_local_chat_endpoint(self):
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"message": {"content": '{"claims": []}'}}
        fake_response.text = ""

        with patch("observatoire.llm.requests.post", return_value=fake_response) as post:
            client = make_llm_client(
                provider="ollama",
                model="mistral:7b",
                base_url="http://localhost:11434",
            )
            result = client.complete_json("system", "user")

        self.assertEqual(result, {"claims": []})
        post.assert_called_once()
        request_url = post.call_args.args[0]
        request_payload = post.call_args.kwargs["json"]
        self.assertEqual(request_url, "http://localhost:11434/api/chat")
        self.assertEqual(request_payload["model"], "mistral:7b")
        self.assertEqual(request_payload["format"], "json")
        self.assertFalse(request_payload["stream"])

    def test_ollama_alias_uses_open_source_default_model(self):
        client = make_llm_client(provider="olama", model="gpt-4.1-mini")

        self.assertIsInstance(client, OllamaChatClient)
        self.assertEqual(client.model, "llama3.1:8b")

    def test_ollama_client_accepts_generation_options_from_env(self):
        fake_response = Mock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"message": {"content": '{"discursive_card": null}'}}
        fake_response.text = ""

        with patch.dict(
            "os.environ",
            {
                "OLLAMA_NUM_CTX": "2048",
                "OLLAMA_NUM_PREDICT": "512",
                "OLLAMA_NUM_THREAD": "8",
                "OLLAMA_KEEP_ALIVE": "30m",
            },
        ):
            with patch("observatoire.llm.requests.post", return_value=fake_response) as post:
                client = make_llm_client(
                    provider="ollama",
                    model="llama3.1:8b",
                    base_url="http://localhost:11434",
                )
                client.complete_json("system", "user")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["options"]["num_ctx"], 2048)
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertEqual(payload["options"]["num_thread"], 8)
        self.assertEqual(payload["keep_alive"], "30m")

    def test_parse_json_content_tolerates_fenced_json(self):
        payload = _parse_json_content('```json\n{"claims": []}\n```')

        self.assertEqual(payload, {"claims": []})


if __name__ == "__main__":
    unittest.main()
