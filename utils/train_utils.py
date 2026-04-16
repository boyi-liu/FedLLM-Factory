import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
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
        done = False

        for epoch in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(train_loader):
                if self.args.task_type == 'SEQ_CLS':
                    input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
                    attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
                    labels = torch.tensor(batch['labels']).to(model.device)

                    with autocast('cuda'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        loss = loss / accumulation_steps
                    scaler.scale(loss).backward()

                elif self.args.task_type == 'CAUSAL_LM':
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

                if global_step >= self.args.step:
                    done = True
                    break

            print(
                f"Round {self.client.server.round} | Client {self.client.id} | Epoch {epoch + 1} | Loss: {loss.item() * accumulation_steps:.4f}")

        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


