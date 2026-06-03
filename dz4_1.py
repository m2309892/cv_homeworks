import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    # pin_memory=True ускоряет передачу CPU->GPU при использовании non_blocking=True
    # num_workers>0 параллелит загрузку данных
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )

    device = torch.device('cuda')

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    for batch_idx, (data, target) in enumerate(dataloader):
        # non_blocking=True позволяет асинхронно копировать данные на GPU, пока выполняются другие операции (работает совместно с pin_memory)
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # генерируем шум сразу на GPU, чтобы не было лишнего CPU->GPU копирования
        noise = torch.randn_like(data)
        data = data + noise

        optimizer.zero_grad(set_to_none=True)  # быстрее и экономит память

        # time.time() без torch.cuda.synchronize() даёт нечестные метрики,т.к. CUDA-операции асинхронные
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)

        fwd_start.record()
        output = model(data)
        loss = criterion(output, target)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        optimizer.step()

        # синхронизация нужна, чтобы получить корректные тайминги.
        torch.cuda.synchronize()
        forward_times.append(fwd_start.elapsed_time(fwd_end) / 1000.0)  # ms -> s
        backward_times.append(bwd_start.elapsed_time(bwd_end) / 1000.0)

        # храним loss.item(), а не тензор с графом вычислений, иначе удерживается весь autograd-граф, что может вызвать утечку памяти и OOM.
        loss_value = loss.item()
        losses_history.append(loss_value)
        print(f"Batch {batch_idx} loss: {loss_value:.4f}")

        # torch.cuda.empty_cache() в цикле не освобождает память по-настоящему, но синхронизирует поток и ломает асинхронный конвейер и замедляет работу.

    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times)}, "
          f"avg backward time is {statistics.mean(backward_times)}")


train()
