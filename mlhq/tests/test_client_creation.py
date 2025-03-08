import unittest
from  mlhq.backend.openai import (
    Client, 
    OLLAMA_BACKEND, 
    LOCAL_BACKEND, 
    HF_CLIENT_BACKEND, 
    MODELS, 
    HF_MODELS, 
    OLLAMA_MODELS,
    DEFAULT_MODEL, 
    DEFAULT_BACKEND,
) 
# --------------------------------------------------------------------|-------:
class TestClientCreation(unittest.TestCase):
    def setUp(self):
        # This runs before each test
        # Reset any module-level state if needed
        pass 
        
    def tearDown(self):
        # This runs after each test
        # Clean up any resources
        pass

    def test_aff_empty_client(self): 
        client = Client()
        self.assertEqual(client._model, DEFAULT_MODEL)
        self.assertEqual(client._backend, DEFAULT_BACKEND)

    def test_aff_local_hf_client(self): 
        model = "meta-llama/Llama-3.1-8B-Instruct"
        client = Client(model)
        self.assertEqual(client._model, model)
        self.assertEqual(client._backend, LOCAL_BACKEND) 
    
    def tests_aff_ollama_client(self):
        model = "llama3.2:1b"
        client = Client(model)
        self.assertEqual(client._model, model)
        self.assertEqual(client._backend, OLLAMA_BACKEND)
# --------------------------------------------------------------------|-------:
if __name__ == '__main__':
    unittest.main()
