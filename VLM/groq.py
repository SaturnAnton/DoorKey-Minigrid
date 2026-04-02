from PIL import Image
from monorepo import GroqLLM, load_api_keys, GROQ_MULTIMODAL_MODEL_ID

load_api_keys()

client = GroqLLM(model_id=GROQ_MULTIMODAL_MODEL_ID)

image = Image.open("../figure/Verona_Logo.png")

risposta = client.ask(prompt="Cosa vedi in questa immagine?", images=[image])
print(risposta)