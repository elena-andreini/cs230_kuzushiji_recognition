from torch.utils.tensorboard import SummaryWriter
import numpy as np
# Function to calculate accuracy
def calculate_accuracy(output, labels):
    _, preds = torch.max(output, dim=1)
    correct = torch.sum(preds == labels)
    return correct.item() / len(labels)

# Function to calculate per-class accuracy
def per_class_accuracy(y_true, output, num_classes):
    accuracies = {}
    #y_true = y_true.detach().cpu().numpy()
    #y_pred = y_pred.detach().cpu().numpy()
    #print(output.shape)
    for cls in range(num_classes):
        cls_mask = (y_true == cls)
        if torch.sum(cls_mask) == 0:
            cls_accuracy = float('nan')  # Handle case with no samples of this class
        else:
            #print(output[cls_mask].shape)
            cls_accuracy = calculate_accuracy(output[cls_mask], y_true[cls_mask])
        accuracies[cls] = cls_accuracy

    return accuracies
# Example training loop
def train_dual_branch(model, train_dataloader, valid_dataloader, criterion, optimizer, labels_to_int,  writer, epochs=10):
    model = model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        class_accuracies = {cls: [] for cls in range(num_classes)}
        for char_input, context_input, label in train_dataloader:
            optimizer.zero_grad()
            integer_labels = [labels_to_int[l] for l in label]
            #print(f'integer labels {integer_labels}')
            integer_labels = torch.tensor(integer_labels).to(device)
            char_input = char_input.to(device)
            context_input = context_input.to(device)
            output = model(char_input, context_input)
            #print(f'output shape {output.shape}')
            loss = criterion(output, integer_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Calculate per-class accuracy
            batch_class_accuracies = per_class_accuracy(integer_labels, output, num_classes)
            for cls in batch_class_accuracies:
                if not np.isnan(batch_class_accuracies[cls]):
                    class_accuracies[cls].append(batch_class_accuracies[cls])
        avg_train_loss = total_loss / len(train_dataloader)
        writer.add_scalar('Train Loss', avg_train_loss, epoch)
         # Calculate average per-class accuracy for the epoch
        epoch_class_accuracies = {cls: np.mean(class_accuracies[cls]) if class_accuracies[cls]
                                  else None for cls in class_accuracies}
        # Log per-class accuracies to TensorBoard, handle None values
        for cls, acc in epoch_class_accuracies.items():
          if acc is not None:
            writer.add_scalar(f'Class_{cls}_Accuracy', acc, epoch)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        with torch.no_grad():
            for char_input, context_input, label in valid_dataloader:
                integer_labels = [labels_to_int[l] for l in label]
                #print('integer_labels')
                integer_labels = torch.tensor(integer_labels).to(device)
                char_input = char_input.to(device)
                context_input = context_input.to(device)
                output = model(char_input, context_input)
                loss = criterion(output, integer_labels)
                val_loss += loss.item()
                val_correct += calculate_accuracy(output, integer_labels) * char_input.size(0)
                val_samples += char_input.size(0)
        val_accuracy = val_correct / val_samples
        avg_val_loss = val_loss / len(valid_dataloader)
        model.train()  # Set model back to training mode
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} Validation Accuracy: {val_accuracy:.4f}')
    print("Training completed!")
