from densenet.mlx import DenseNet, BasicBlock
import mlx.core as mx
from mlx.data import datasets
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import trainer
import time
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorboard as tb
import datetime

growth_rate = 12
model_nblocks = [6, 12, 24, 16]
reduction_rate = 0.5
num_classes = 100

model = DenseNet(
    BasicBlock,
    nblocks=model_nblocks,
    growth_rate=growth_rate,
    num_classes=num_classes,
    reduction=reduction_rate,
)
mx.eval(model)

model_parameters = sum(x.size for _, x in tree_flatten(model.parameters()))
print(f"Model has {model_parameters/(1024**2):.2f}M parameters")

layers = [name for name, _ in tree_flatten(model.parameters())]
print(f"Model has {len(layers)} layers")

train_set = datasets.load_cifar100(train=True)
test_set = datasets.load_cifar100(train=False)


def get_streamed_data(data, batch_size=0, train=True):
    mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))

    def normalize(x):
        x = x.astype("float32") / 255.0
        return (x - mean) / std

    if not train:
        stream = data.to_stream()
        return stream.key_transform("image", normalize).batch(batch_size).prefetch(8, 8)

    stream = data.shuffle().to_stream()
    stream = stream.image_random_area_crop("image", (0.08, 1.0), (0.75, 1.333))
    stream = stream.image_resize("image", 32, 32)
    stream = stream.image_random_h_flip("image", prob=0.5)
    stream = stream.pad("image", 0, 4, 4, 0.0)
    stream = stream.pad("image", 1, 4, 4, 0.0)
    stream = stream.image_random_crop("image", 32, 32)
    stream = stream.key_transform("image", normalize)
    stream = stream.batch(batch_size) if batch_size > 0 else stream
    return stream.prefetch(8, 8)


epochs = 120
optimizer = optim.SGD(
    learning_rate=0.1,
    weight_decay=1e-4,
    momentum=0.9,
    dampening=False,
)

batch_size = 128
train_data = get_streamed_data(batch_size=batch_size, data=train_set, train=True)
test_data = get_streamed_data(batch_size=batch_size, data=test_set, train=False)

test_accuracies = []

now_formatted = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_output_dir = f"logs/densenet_cifar100/{now_formatted}/train"
train_writer = tb.summary.Writer(train_output_dir)
test_output_dir = f"logs/densenet_cifar100/{now_formatted}/test"
test_writer = tb.summary.Writer(test_output_dir)

for epoch in range(epochs):
    if epoch in [30, 60, 90]:
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

    train_writer.add_scalar("loss", train_loss.item(), step=epoch)
    train_writer.add_scalar("accuracy", train_acc.item(), step=epoch)
    train_writer.add_scalar("learning_rate", optimizer.learning_rate, step=epoch)
    test_writer.add_scalar("accuracy", test_acc.item(), step=epoch)

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
