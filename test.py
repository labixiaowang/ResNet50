import torch
from torchvision import transforms
from resnet50 import ResNet,Bottleneck
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
from PIL import Image
model=ResNet(Bottleneck,[3,4,6,3],4).to(device)

# 加载模型权重
model.load_state_dict(torch.load("./resnet_50.pth"))
model.eval()  # 设置模型为评估模式
# 假设你有一个预处理后的输入图像 `input_image`
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_path = "./test_picture.jpg"
image = Image.open(image_path).convert('RGB')

# 应用转换
input_image = transformations(image)
input_image = input_image.unsqueeze(0)  # 增加一个维度作为批次大小
input_image = input_image.to(device)  # 将输入图像移动到与模型相同的设备

# 使用模型进行预测
with torch.no_grad():  # 不需要计算梯度
    output = model(input_image)

# 获取预测结果
predicted_class = output.argmax(dim=1).item()
print(f"Predicted class: {predicted_class}")
