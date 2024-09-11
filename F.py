from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from resnet50 import ResNet, Bottleneck

# 初始化 Flask 应用
app = Flask(__name__)

# 检查是否有 GPU 并设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model = ResNet(Bottleneck, [3, 4, 6, 3], 4).to(device)
model.load_state_dict(torch.load("./resnet_50.pth"))
model.eval()  # 设置模型为评估模式

# 定义图像转换流程
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 预测函数
def predict_image_class(image_path):
    image = Image.open(image_path).convert('RGB')
    input_image = transformations(image)
    input_image = input_image.unsqueeze(0).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        output = model(input_image)

    # 获取预测结果
    predicted_class = output.argmax(dim=1).item()
    return predicted_class


# Flask 路由，用于上传和分类图像
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # 将上传的文件保存到本地
    image_path = "./uploaded_image.jpg"
    file.save(image_path)

    # 获取预测结果
    predicted_class = predict_image_class(image_path)

    # 返回预测结果
    return jsonify({'prediction': predicted_class})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)