from configparser import ConfigParser
import torch
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer


class InternVL:
    def __init__(self, config_path):
        self.config = ConfigParser()
        self.config.read(config_path)
        self.model_path = self.config.get('Model', 'path')
        
        self.model = AutoModel.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                        device_map='auto',load_in_8bit=True).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model_path)
        self.generation_config = dict(
                        num_beams=int(self.config.get('Model', 'num_beams')),
                        max_new_tokens=int(self.config.get('Model', 'max_new_tokens')),
                        do_sample=False,
                    )
        
        
    def predict(self, image, prompt):
        self.img = Image.fromarray(image)
        self.img = self.img.resize((448, 448))
        pixel_values = self.image_processor(images=self.img, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        response = self.model.chat(self.tokenizer, pixel_values, prompt, self.generation_config)
        
        return response
