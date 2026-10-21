"""Datasets, models and training loops, written separately from reference scripts."""
from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
from .optimizers import GradientCALR, HimmelblauCALR, DiagonalHessianCALR

ROOT = Path(__file__).resolve().parents[1]


def network(widths, flatten=False):
    layers = [nn.Flatten()] if flatten else []
    for index, (source, target) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(source, target))
        if index < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def digits(training=True):
    transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])
    return datasets.MNIST(str(Path(os.environ.get('CALR_DATA_DIR', ROOT / 'data'))), train=training,
                          download=True, transform=transform)


def himmelblau(steps=350):
    settings = [
        ('Adam', torch.optim.Adam, {'lr': 0.01}),
        ('RMSProp', torch.optim.RMSprop, {'lr': 0.01}),
        ('Nadam', torch.optim.NAdam, {'lr': 0.01}),
        ('CALR Default', HimmelblauCALR, {'lr': 0.1}),
        ('CALR Grad Clip', HimmelblauCALR, {'lr': 0.01, 'clip': 0.1}),
    ]
    result = {}
    for name, constructor, options in settings:
        position = torch.zeros(2, requires_grad=True)
        optimizer = constructor([position], **options)
        losses, norms, positions = [], [], []
        for _ in range(steps):
            optimizer.zero_grad()
            x, y = position.detach().numpy()
            # Keep float32 NumPy arithmetic to match the reference benchmark.
            value = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
            derivative = np.array([4*x*(x**2+y-11)+2*(x+y**2-7),
                                   2*(x**2+y-11)+4*y*(x+y**2-7)])
            position.grad = torch.tensor(derivative, dtype=torch.float32)
            losses.append(float(value))
            norms.append(float(np.linalg.norm(position.grad.numpy())))
            positions.append(position.detach().numpy().copy().tolist())
            optimizer.step()
        result[name] = dict(loss=losses, gradient_norm=norms, positions=positions,
                            final_position=position.detach().tolist())
    return result


def small_comparison(kind, epochs=None):
    torch.manual_seed(42 if kind == 'toy' else 0)
    if kind == 'toy':
        inputs = torch.rand(100, 2) * 2 - 1
        targets = (inputs[:, 0]**2 + inputs[:, 1]**2 > 0.5).float().unsqueeze(1)
        widths, flatten, default_epochs = [2, 16, 16, 1], False, 500
    else:
        dataset = digits()
        subset, _ = torch.utils.data.random_split(dataset, [16, len(dataset)-16])
        loader = torch.utils.data.DataLoader(subset, batch_size=16, shuffle=True)
        inputs, labels = next(iter(loader))
        inputs = inputs.view(16, -1)
        targets = nn.functional.one_hot(labels, num_classes=10).float()
        widths, flatten, default_epochs = [784, 32, 10], True, 10
    models = [network(widths, flatten), network(widths, flatten)]
    objective = nn.BCEWithLogitsLoss()
    optimizers = [GradientCALR(models[0].parameters(), exploratory=True),
                  DiagonalHessianCALR(models[1], objective, inputs, targets)]
    result = {}
    for label, model, optimizer in zip(['Gradient CALR', 'Hessian CALR'], models, optimizers):
        history = []
        for _ in range(default_epochs if epochs is None else epochs):
            model.zero_grad(set_to_none=True)
            loss = objective(model(inputs), targets)
            loss.backward()
            optimizer.step()
            history.append(loss.item())
        result[label] = {'loss': history}
    return result


def mnist(epochs=50, progress=print):
    torch.manual_seed(42)
    np.random.seed(42)
    training = torch.utils.data.DataLoader(digits(), batch_size=64, shuffle=True)
    testing = torch.utils.data.DataLoader(digits(False), batch_size=1000, shuffle=False)
    settings = [('Adam', torch.optim.Adam, {}), ('Nadam', torch.optim.NAdam, {}),
                ('RMSProp', torch.optim.RMSprop, {}),
                ('SGD', torch.optim.SGD, {'momentum': 0.9}),
                ('CALR Default', GradientCALR, {})]
    objective, result = nn.CrossEntropyLoss(), {}
    for name, constructor, options in settings:
        model = network([784, 256, 128, 10], flatten=True)
        optimizer = constructor(model.parameters(), lr=0.01, **options)
        history = {key: [] for key in ['train_loss', 'train_accuracy', 'test_loss',
                                       'test_accuracy', 'batch_loss']}
        for epoch in range(epochs):
            for phase, loader in [('train', training), ('test', testing)]:
                model.train(phase == 'train')
                loss_sum, correct, count = 0.0, 0, 0
                with torch.set_grad_enabled(phase == 'train'):
                    for inputs, labels in loader:
                        if phase == 'train':
                            optimizer.zero_grad()
                        predictions = model(inputs)
                        loss = objective(predictions, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            history['batch_loss'].append(loss.item())
                        loss_sum += loss.item()
                        correct += predictions.argmax(1).eq(labels).sum().item()
                        count += len(labels)
                # Reference reports mean of batch means, not sample-weighted loss.
                history[phase + '_loss'].append(loss_sum / len(loader))
                history[phase + '_accuracy'].append(100.0 * correct / count)
            progress(f'{name}: epoch {epoch+1}/{epochs}, test accuracy '
                     f'{history["test_accuracy"][-1]:.2f}%', flush=True)
        result[name] = history
    return result
