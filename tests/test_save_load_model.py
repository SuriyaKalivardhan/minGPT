import unittest
import torch
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import dill as pickle

class TestSaveLoadModel(unittest.TestCase):

    def test_save_tensor_dill(self):
        model_type = 'gpt2'
        model = GPT.from_pretrained(model_type)
        with open('/tmp/models/mingpt.pkl', 'wb') as file:
            pickle.dump(model, file)

    def test_load_tensor_dill(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open('/tmp/models/mingpt.pkl', 'rb') as file:
            model = pickle.load(file)
        model.to(device)
        model.eval()

        tokenizer = BPETokenizer()
        prompt = "Hello World"
        x = tokenizer(prompt).to(device)
        logits1, loss = model(x)
        y = model.generate(x, max_new_tokens=20, do_sample=False)[0]

        # convert indices to strings
        out = tokenizer.decode(y.cpu().squeeze())
        print(out)

if __name__ == '__main__':
    unittest.main()
