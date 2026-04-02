from PIL import Image
from monorepo import GroqLLM, load_api_keys, GROQ_MULTIMODAL_MODEL_ID

# Carica la API key da ~/.env.ml
load_api_keys()

# Crea il client con il modello multimodale
client = GroqLLM(model_id=GROQ_MULTIMODAL_MODEL_ID)

image = Image.open("figure/env.png")

# Fai una domanda con un'immagine
risposta = client.ask(prompt="Cosa vedi in questa immagine?", images=[image])
print(risposta)