import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast


class Trainer:
    """
    Stateless trainer decoupled from model architecture.
    Instantiated once per client and reused across rounds.
    """

    def __init__(self, args, dataset, client):
        self.args = args
        self.client = client
        self.task_type = args.task_type
        self.train_loader = DataLoader(
            dataset['train'],
            batch_size=self.args.bs,
            shuffle=True,
            drop_last=True
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self, model):
        """Run one round of local training and return the final loss."""
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        loss = self._train_loop(model, optimizer, scaler)
        self._log(loss)
        return loss

    # ------------------------------------------------------------------
    # Per-task implementations
    # ------------------------------------------------------------------

    def _forward(self, model, batch):
        if self.task_type == 'SEQ_CLS':
            return self._forward_seq_cls(model, batch)
        elif self.task_type == 'CAUSAL_LM':
            return self._forward_causal_lm(model, batch)
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type}")

    def _forward_seq_cls(self, model, batch):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.tensor(batch['labels']).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    def _forward_causal_lm(self, model, batch):
        input_ids = torch.stack(batch['input_ids']).transpose(0, 1).to(model.device)
        labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, labels=labels)
        return outputs.loss

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(self, model, optimizer, scaler):
        accumulation_steps = self.args.grad_accum
        global_step = 0
        loss = None
        done = False

        optimizer.zero_grad()
        for _ in range(self.args.epoch):
            if done:
                break
            for step, batch in enumerate(self.train_loader):
                loss = self._forward(model, batch) / accumulation_steps
                scaler.scale(loss).backward()

                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    global_step += 1

                if global_step >= self.args.step:
                    done = True
                    break

        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, loss):
        print(
            f"Round {self.client.server.round} | "
            f"Client {self.client.id} | "
            f"Loss: {loss.item() * self.args.grad_accum:.4f}"
        )
