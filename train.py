import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from resnet50 import ResNet,Bottleneck
from torchvision import transforms
from Customdata import CustomImageDataset
import logging
# 配置日志记录器
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 在程序中记录日志
logging.info('程序开始执行')
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用GPU
    print(device)
    # 定义预处理步骤（例如，将图像转换为张量并归一化）
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建并实例化自定义数据集
    custom_dataset = CustomImageDataset("./valid.txt", transform=transformations)
    # 定义训练集和测试集的比例
    total_length = len(custom_dataset)
    train_len = int(total_length*0.7)
    test_len = total_length - train_len
    # from torch.utils.data.dataloader import default_collate
    # def safe_collate(batch):
    #     batch = list(filter(lambda x: x[0] is not None and x[1] is not None, batch))  # 过滤掉包含None的样本
    #     return default_collate(batch)

    # 划分测试集和训练集
    train_data, test_data = torch.utils.data.random_split(custom_dataset, [train_len, test_len])
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle = True)#,collate_fn=safe_collate)

    model=ResNet(Bottleneck,[3,4,6,3],4).to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    train_num_sum = 0
    train_num = 0
    test_num = 0
    # 添加tensorboard
    writer = SummaryWriter("./logs_resnet_model")

    for epoch in range(2):
        # 训练
        print("---------第{}轮训练开始-------".format(epoch+1))
        for data in train_loader:

            img_train,lable_train = data
            img_train = img_train.to(device)
            lable_train = lable_train.to(device)
            output_train = model(img_train)
            loss_train = loss_function(output_train,lable_train)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            train_num_sum += 1
            if train_num_sum % 100 == 0:
                print("训练次数:{},Loss:{}".format(train_num_sum, loss_train.item()))  # item将tensor转换成python的数字
                writer.add_scalar("train_loss", loss_train.item(), train_num_sum)
    # 测试
        total_accurancy = 0  # 准确率
        total_test_loss = 0 # 总误差
        with torch.no_grad():
            print("------  第{}轮测试开始-------".format(epoch+1))
            for data in test_loader:
                img_test,lable_test = data
                img_test = img_test.to(device)
                lable_test = lable_test.to(device)
                output_test = model(img_test)
                loss_test = loss_function(output_test,lable_test)
                total_test_loss+=loss_test.item()
                total_accurancy += (output_test.argmax(1) == lable_test).sum().item()
            test_num += 1
            print("第{}轮的loss:{}".format(epoch+1,total_test_loss))
            print("第{}轮的accurancy:{}".format(epoch+1,total_accurancy/total_length))
            writer.add_scalar("total_test_loss",total_test_loss,test_num)
            writer.add_scalar("total_accurancy",total_accurancy/total_length,test_num)
    torch.save(model.state_dict(),"./resnet_50.pth")
    print("模型数据已保存")
    writer.close()
except Exception as e:
    logging.error(f'程序发生异常: {e}', exc_info=True)
# 在程序结束时记录日志
logging.info('程序执行结束')