import torch


@torch.no_grad()
def compute_metrics(model, criterion, dataloader, args):
    model.eval()

    correct = 0
    loss = 0
    for data in dataloader:
        inputs, labels = data
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        labels_loss = torch.nn.functional.one_hot(labels, num_classes=args.num_classes).to(torch.float32).to(
            args.device)
        preds = model(inputs)
        pred_labels = preds.argmax(dim=1)
        correct += pred_labels.eq(labels).sum().item()
        loss += criterion(preds, labels_loss) / inputs.shape[0]

    return correct / len(dataloader.dataset), loss.item()
