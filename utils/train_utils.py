import torch
import math

from torch.optim import AdamW
from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast
from torch.amp import GradScaler, autocast

class Trainer:
    def __init__(self, args, dataset, client):
        self.args = args
        self.client = client
        self.train_loader = DataLoader(
            dataset['train'],
            batch_size=self.args.bs,
            shuffle=True,
            drop_last=True
        )
        self.eval_loader = DataLoader(
            dataset['test'],
            batch_size=1,
            shuffle=False
        )

    def train(self, model):
        model.train()
        train_loader = self.train_loader

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )

        scaler = GradScaler()
        accumulation_steps = self.args.grad_accum

        global_step = 0
        optimizer.zero_grad()

        for epoch in range(self.args.epoch):
            for step, batch in enumerate(train_loader):
                input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
                labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)

                with autocast('cuda'):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if step % (5 * accumulation_steps) == 0:
                    print(
                        f"Round {self.client.server.round} | Client {self.client.id} | Epoch {epoch + 1} | Step {step} | Loss: {loss.item() * accumulation_steps:.4f}")

        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


    def eval(self, model):
        model.eval()
        total_loss = 0
        total_steps = 0

        eval_loader = self.eval_loader

        for batch in eval_loader:
            input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
            labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)

            with torch.no_grad(), autocast('cuda'):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    total_steps += 1

        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

        metrics = {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }

        print(f"Round {self.client.server.round} | Client {self.client.id} | Loss: {metrics['eval_loss']:.4f} | Perplexity: {metrics['perplexity']:.4f}")
        return metrics