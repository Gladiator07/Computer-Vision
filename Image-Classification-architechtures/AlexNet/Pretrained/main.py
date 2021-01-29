from torchvision import models, transforms
import torch
from PIL import Image


alexnet = models.alexnet(pretrained=True)
print(alexnet)

tranform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img = Image.open("dog.jpg")

img_t = tranform(img)
batch_t = torch.unsqueeze(img_t, 0)

# using pre-trained model to classify dog image
alexnet.eval()

result = alexnet(batch_t)
print(out.shape)

with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)
percentage = torch.nn.functiona.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)

out = [(labels[idx], percentage[idx].item() for idx in indices[0][:5])]