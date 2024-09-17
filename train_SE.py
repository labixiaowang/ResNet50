import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from Resnet50_SE import ResNet, Bottleneck
from torchvision import transforms
from Customdata import CustomImageDataset
import logging
import wandb
from argparse import Namespace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
print(device)

# Define sweep configuration
sweep_config = {
    "method": "random",  # or "grid", "bayes"
    "metric": {
        "name": "total_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "values": [0.0001, 0.001, 0.01]
        },
        "epochs": {
            "values": [1, 5, 10]
        },
        "batch_size": {
            "values": [16, 32, 64, 128]
        },
        "optimizer": {
            "values": ["SGD", "Adam"]
        },
        "weight_decay": {
            "values": [0, 0.001, 0.01, 0.1]
        }
    }
}

def train(config=None):
    # Initialize wandb with the config passed from the sweep agent
    with wandb.init(config=config):
        config = wandb.config
        # Configure the logger
        logging.basicConfig(filename='training.log', level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('Program started.')
        try:
            # Define data transformations
            transformations = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            # Create and split the dataset
            custom_dataset = CustomImageDataset("./valid.txt", transform=transformations)
            total_length = len(custom_dataset)
            train_len = int(total_length * 0.7)
            test_len = total_length - train_len
            train_data, test_data = torch.utils.data.random_split(custom_dataset, [train_len, test_len])

            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=config.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=config.batch_size, shuffle=True)

            # Initialize the model, loss function, and optimizer
            model = ResNet(Bottleneck, [3, 4, 6, 3], 4).to(device)
            loss_function = nn.CrossEntropyLoss().to(device)
            if config.optimizer == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=config.learning_rate,
                                            weight_decay=config.weight_decay)
            elif config.optimizer == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(),
                                             lr=config.learning_rate,
                                             weight_decay=config.weight_decay)
            else:
                raise ValueError(f"Optimizer {config.optimizer} not recognized.")

            train_num_sum = 0
            test_num = 0
            # Add wandb.watch
            wandb.watch(model, log="all")
            # Add TensorBoard writer
            writer = SummaryWriter("./logs_resnet_model")
            init_img = torch.Tensor(1, 3, 224, 224).to(device)
            writer.add_graph(model, init_img)

            for epoch in range(config.epochs):
                # Training
                print(f"--------- Epoch {epoch + 1} Training Start -------")
                model.train()
                for data in train_loader:
                    img_train, label_train = data
                    img_train = img_train.to(device)
                    label_train = label_train.to(device)
                    output_train = model(img_train)
                    loss_train = loss_function(output_train, label_train)
                    optimizer.zero_grad()
                    loss_train.backward()
                    optimizer.step()
                    train_num_sum += 1
                    if train_num_sum % 100 == 0:
                        print(f"Training Step: {train_num_sum}, Loss: {loss_train.item()}")
                        writer.add_scalar("train_loss", loss_train.item(), train_num_sum)
                        wandb.log({"train_loss": loss_train.item(), "step": train_num_sum})

                        for name, param in model.named_parameters():
                            writer.add_histogram(f"{name}_weights", param, train_num_sum)
                            if param.grad is not None:
                                writer.add_histogram(f"{name}_grads", param.grad, train_num_sum)
                # Testing
                total_accuracy = 0  # Accuracy
                total_test_loss = 0  # Total loss
                model.eval()
                with torch.no_grad():
                    print(f"------ Epoch {epoch + 1} Testing Start -------")
                    for data in test_loader:
                        img_test, label_test = data
                        img_test = img_test.to(device)
                        label_test = label_test.to(device)
                        output_test = model(img_test)
                        loss_test = loss_function(output_test, label_test)
                        total_test_loss += loss_test.item()
                        total_accuracy += (output_test.argmax(1) == label_test).sum().item()
                    test_num += 1
                    accuracy = total_accuracy / total_length
                    print(f"Epoch {epoch + 1} Loss: {total_test_loss}")
                    print(f"Epoch {epoch + 1} Accuracy: {accuracy}")
                    writer.add_scalar("total_test_loss", total_test_loss, test_num)
                    writer.add_scalar("total_accuracy", accuracy, test_num)
                    wandb.log({
                        "total_test_loss": total_test_loss,
                        "total_accuracy": accuracy,
                        "epoch": epoch + 1
                    })
            # Save the model
            model_path = f"./resnet_50_SE_epoch{config.epochs}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)  # Save model to wandb
            print("Model saved.")
            writer.close()
        except Exception as e:
            logging.error(f'Program encountered an error: {e}', exc_info=True)
        logging.info('Program finished.')

if __name__ == '__main__':
    # Initialize the sweep
    sweep_id = wandb.sweep(sweep_config, project="garbage-classification")
    # Start the sweep agent
    wandb.agent(sweep_id, function=train,count=5)
