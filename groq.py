from PIL import Image
from monorepo import GroqLLM, load_api_keys

load_api_keys()

client = GroqLLM(model_id="meta-llama/llama-4-scout-17b-16e-instruct")

image = Image.open("figure/env.png")

risposta = client.ask(prompt="Cosa vedi in questa immagine?", images=[image])
print(risposta)