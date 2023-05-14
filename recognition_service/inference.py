import PIL
import torch
from torchvision import models


class EfficientnetB0Model:
    def __init__(self) -> None:
        model = models.efficientnet_b0(pretrained=True)
        model = model.eval()

        #model.fc = torch.nn.Sequential()
        model.classifier = torch.nn.Sequential()
        
        #weights = ResNet50_Weights.DEFAULT
        weights = models.EfficientNet_B0_Weights.DEFAULT
        preprocess = weights.transforms()

        self.model = model
        self.preprocess = preprocess
    
    def extract_features(self, image: PIL.Image) -> list:
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor[None, :]
        
        with torch.no_grad():
            f = self.model(image_tensor).detach().numpy()
            #f = f / f.shape[1]
            f = f.tolist()
        
        return f[0]


Model = EfficientnetB0Model


