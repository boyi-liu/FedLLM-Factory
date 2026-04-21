import time
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

    def train_windowed(self, model, start_time, window_end_time):
        """Run local training but stop early when the simulated window closes.

        Args:
            model: the model to train.
            start_time: simulated wall-clock datetime when training is assigned.
            window_end_time: simulated datetime at which the device's window
                closes; training stops after the first gradient update whose
                real elapsed time exceeds this simulated budget.

        Returns:
            (loss, truncated): the last computed loss and a bool indicating
            whether the run was cut short by the window deadline.
        """
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(model, optimizer, scaler, start_time, window_end_time, delay=delay)
        self._log(loss)
        return loss, truncated, global_step

    def train_resumed(self, model, start_step: int, start_time, window_end_time):
        """Resume windowed training from *start_step* completed gradient steps.

        Loads model weights from the caller (already restored externally) and
        continues for the remaining ``args.step - start_step`` optimizer steps,
        still subject to the window deadline.

        Returns:
            (loss, truncated, completed_steps)
        """
        model.train()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr * (0.99 ** self.client.server.round)
        )
        scaler = GradScaler()

        delay = getattr(self.client, 'delay', 1.0)
        loss, truncated, global_step = self._train_loop_windowed(
            model, optimizer, scaler, start_time, window_end_time, resume_step=start_step, delay=delay
        )
        self._log(loss)
        return loss, truncated, global_step

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
        attention_mask = torch.stack(batch['attention_mask']).transpose(0, 1).to(model.device)
        labels = torch.stack(batch['labels']).transpose(0, 1).to(model.device)
        with autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

    def _train_loop_windowed(self, model, optimizer, scaler, start_time, window_end_time, resume_step: int = 0, delay: float = 1.0):
        accumulation_steps = self.args.grad_accum
        global_step = resume_step
        loss = None
        done = False
        truncated = False

        # real_elapsed = sim_budget / delay  (because training_time = real_elapsed * delay)
        sim_budget = (window_end_time - start_time).total_seconds()
        real_deadline = time.time() + sim_budget / delay

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
                    
                    if time.time() > real_deadline:
                        
                        truncated = True
                        done = True
                        break

                if global_step >= self.args.step:
                    done = True
                    break

        if not truncated and (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        return loss, truncated, global_step

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, loss):
        print(
            f"Round {self.client.server.round} | "
            f"Client {self.client.id} | "
            f"Loss: {loss.item() * self.args.grad_accum:.4f}"
        )
