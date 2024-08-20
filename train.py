from densenet.mlx import DenseNet, BasicBlock
import mlx.core as mx
from mlx.data import datasets
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import trainer
import time
from sklearn.metrics import f1_score, precision_score, recall_score

model = DenseNet(
    BasicBlock,
    nblocks=[6, 12, 24, 16],
    growth_rate=6,
    num_classes=100,
    reduction=0.5,
)
mx.eval(model)

model_parameters = sum(x.size for _, x in tree_flatten(model.parameters()))
print(f"Model has {model_parameters/(1024**2):.2f}M parameters")

layers = [name for name, _ in tree_flatten(model.parameters())]
print(f"Model has {len(layers)} layers")

train_set = datasets.load_cifar100(train=True)
test_set = datasets.load_cifar100(train=False)


def get_streamed_data(data, batch_size=0, shuffled=True):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    buffer = data.shuffle() if shuffled else data
    stream = buffer.to_stream()
    stream = stream.image_random_h_flip("image", prob=0.5)
    stream = stream.pad("image", 0, 4, 4, 0.0)
    stream = stream.pad("image", 1, 4, 4, 0.0)
    stream = stream.image_random_crop("image", 32, 32)
    stream = stream.key_transform("image", normalize)
    stream = stream.batch(batch_size) if batch_size > 0 else stream
    return stream.prefetch(8, 8)


epochs = 300
optimizer = optim.SGD(
    learning_rate=0.1,
    weight_decay=1e-4,
    momentum=0.9,
    dampening=False,
)

batch_size = 64
train_data = get_streamed_data(batch_size=batch_size, data=train_set, shuffled=True)
test_data = get_streamed_data(batch_size=batch_size, data=test_set, shuffled=False)

train_accuracies = []
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    if epoch % 50 == 0 and epoch != 0:
        print(f"Decaying learning rate by 10x at epoch {epoch}")
        optimizer.learning_rate /= 10
        print(f"New learning rate: {optimizer.learning_rate}")

    tic = time.perf_counter()
    train_loss, train_acc, throughput = trainer.train_epoch(
        model,
        train_data,
        optimizer,
        epoch,
        verbose=True,
    )
    toc = time.perf_counter()
    print(
        " | ".join(
            (
                f"Epoch: {epoch+1}",
                f"avg. Train loss {train_loss.item():.3f}",
                f"avg. Train acc {train_acc.item():.3f}",
                f"Throughput: {throughput.item():.2f} images/sec",
                f"Time: {toc-tic:.2f} sec",
            )
        )
    )
    tic = time.perf_counter()
    test_acc = trainer.test_epoch(model, test_data, epoch)
    toc = time.perf_counter()
    print(f"Epoch: {epoch+1} | Test acc {test_acc.item():.3f}, Time: {toc-tic:.2f} sec")

    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    test_accuracies.append(test_acc)

    train_data.reset()
    test_data.reset()


y_true = []
y_pred = []
model.eval()
for batch in test_data:
    X, y = batch["image"], batch["label"]
    X, y = mx.array(X), mx.array(y)
    logits = model(X)
    prediction = mx.argmax(mx.softmax(logits), axis=1)
    y_true = y_true + y.tolist()
    y_pred = y_pred + prediction.tolist()

y_true = np.array(y_true)
y_pred = np.array(y_pred)

precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")
print(f"Precision: {precision}\nRecall: {recall}\nF1 Score: {f1}")

test_data.reset()

model.save_weights("mlx_densenet_dev_cifar100.safetensors")
