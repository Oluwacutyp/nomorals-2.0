"""VITS fine-tune trainer (``VITSTrainer``).

Port of upstream ``train.py`` (single-speaker) and ``train_ms.py``
(multi-speaker), unified into one trainer:

* Single-process CPU or single-GPU by default — upstream asserts CUDA
  and spawns one process per GPU. DDP multi-GPU is kept for CUDA runs
  (``use_ddp=True``); the CPU path runs the same loop in-process with a
  warning that it is slow, instead of refusing.
* The dataset logic is the numpy loader in ``data.py``; a thin torch
  ``Dataset`` adapter converts each numpy batch to tensors, so one
  loader serves both halves.
* ``train_ms.py`` upstream cannot run as shipped (its loop calls the
  model with the wrong signature and its ``evaluate`` passes speaker
  ids as the TorchMoji vector). The multi-speaker path here calls
  forward correctly (``sid=speakers``) and the speaker loader carries
  the TorchMoji + TinyBERT sidecars.
* Runs are recorded in Devon's ``TrainingRegistry`` when available
  (start / complete / fail); without it the trainer still runs and
  logs locally.
* 8-bit AdamW (bitsandbytes) is used only when the config asks for it
  and the package is importable; otherwise AdamW with a named warning.
"""

import itertools
import os

import numpy as np

from . import train_utils
from ._torch import require_torch
from .configs import load_config
from .text.symbols import symbols

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAVE_TENSORBOARD = True
except Exception:  # pragma: no cover - tensorboard is optional here
    SummaryWriter = None
    _HAVE_TENSORBOARD = False


class _NumpyDataset:
    """Torch Dataset adapter around the numpy loader."""

    def __init__(self, loader):
        self.loader = loader

    def __getitem__(self, index):
        return self.loader[index]

    def __len__(self):
        return len(self.loader)


def _to_torch_batch(batch, device, multi_speaker=False):
    torch = require_torch("VITS training")
    if multi_speaker:
        (x, x_lengths, spec, spec_lengths, y, y_lengths, tm, bert,
         bert_lens, sid) = batch
    else:
        (x, x_lengths, spec, spec_lengths, y, y_lengths, tm, bert,
         bert_lens) = batch
        sid = None
    out = (
        torch.from_numpy(np.ascontiguousarray(x)).to(device),
        torch.from_numpy(np.ascontiguousarray(x_lengths)).to(device),
        torch.from_numpy(np.ascontiguousarray(spec)).to(device),
        torch.from_numpy(np.ascontiguousarray(spec_lengths)).to(device),
        torch.from_numpy(np.ascontiguousarray(y)).to(device),
        torch.from_numpy(np.ascontiguousarray(y_lengths)).to(device),
        torch.from_numpy(np.ascontiguousarray(tm)).to(device),
        torch.from_numpy(np.ascontiguousarray(bert)).to(device),
        torch.from_numpy(np.ascontiguousarray(bert_lens)).to(device),
    )
    if sid is not None:
        out = out + (torch.from_numpy(np.ascontiguousarray(sid)).to(device),)
    return out


class VITSTrainer:
    def __init__(self, hps, model_dir, device="cpu", multi_speaker=False,
                 use_ddp=False, rank=0, world_size=1, workers=0):
        torch = require_torch("VITS training")
        from . import commons
        from .data import (
            DistributedBucketSampler,
            TextAudioCollate,
            TextAudioLoader,
            TextAudioSpeakerCollate,
            TextAudioSpeakerLoader,
        )
        from .losses import BinLoss, ForwardSumLoss
        from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
        from .models import MultiCoMBDiscriminator, MultiSubBandDiscriminator, SynthesizerTrn

        self.torch = torch
        self.hps = hps
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.multi_speaker = multi_speaker
        self.use_ddp = use_ddp and torch.cuda.is_available() and world_size > 1
        self.rank = rank
        self.world_size = world_size if self.use_ddp else 1
        self.workers = workers
        self.global_step = 0
        self.epoch = 1
        self._commons = commons
        self._mel = (mel_spectrogram_torch, spec_to_mel_torch)
        self._loss_fns = (BinLoss(), ForwardSumLoss())
        self._collate_cls = (TextAudioSpeakerCollate if multi_speaker
                             else TextAudioCollate)
        self._loader_cls = (TextAudioSpeakerLoader if multi_speaker
                            else TextAudioLoader)
        self._sampler_cls = DistributedBucketSampler

        # The plain VITS configs (ljs_li22, ljs_li44, ...) carry no
        # bert/moji sizes: upstream's own model constructor requires them,
        # so a plain config cannot train here. Fail with the reason named.
        for _key in ("moji_start_size", "moji_enc_sizes", "bert_size",
                     "bert_final"):
            if _key not in hps.model:
                raise ValueError(
                    f"config has no model.{_key}: this is a plain VITS config, "
                    "and the fine-tune trainer needs a tmbert config "
                    "(ljs_li44_tmbert_ft_s1_arpa.json and kin)")
        os.makedirs(model_dir, exist_ok=True)
        self.logger = train_utils.get_logger(model_dir)
        self.logger.info(hps)
        train_utils.check_git_hash(model_dir)

        n_speakers = int(getattr(hps.data, "n_speakers", 0)) if multi_speaker else 0
        self.net_g = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=n_speakers,
            **hps.model).to(self.device)
        self.mcmbd = MultiCoMBDiscriminator(
            hps.disc.combd_kernels, hps.disc.combd_channels,
            hps.disc.combd_groups, hps.disc.combd_strides).to(self.device)
        self.msbd = MultiSubBandDiscriminator(
            hps.disc.tkernels, hps.disc.fkernel, hps.disc.tchannels,
            hps.disc.fchannels, hps.disc.tstrides, hps.disc.fstride,
            hps.disc.tdilations, hps.disc.fdilations, hps.disc.tsubband,
            hps.disc.pqmf_n, hps.disc.pqmf_m,
            hps.disc.freq_init_ch).to(self.device)

        if getattr(hps.train, "use_8bit", False):
            try:
                import bitsandbytes as bnb
                optim_cls = bnb.optim.AdamW
            except ImportError:
                self.logger.warning(
                    "config asks for 8-bit AdamW but bitsandbytes is not "
                    "installed; falling back to torch AdamW")
                optim_cls = torch.optim.AdamW
        else:
            optim_cls = torch.optim.AdamW
        self.optim_g = optim_cls(
            self.net_g.parameters(), hps.train.learning_rate,
            betas=hps.train.betas, eps=hps.train.eps)
        self.optim_d = optim_cls(
            itertools.chain(self.msbd.parameters(), self.mcmbd.parameters()),
            hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=hps.train.lr_decay, last_epoch=-1)
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=hps.train.lr_decay, last_epoch=-1)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=getattr(hps.train, "fp16_run", False))

        if self.use_ddp:
            import torch.distributed as dist
            from torch.nn.parallel import DistributedDataParallel as DDP
            dist.init_process_group(backend="nccl", init_method="env://",
                                    world_size=self.world_size, rank=rank)
            torch.cuda.set_device(rank)
            self.net_g = DDP(self.net_g, device_ids=[rank],
                             find_unused_parameters=not multi_speaker)
            self.mcmbd = DDP(self.mcmbd, device_ids=[rank],
                             find_unused_parameters=not multi_speaker)
            self.msbd = DDP(self.msbd, device_ids=[rank])
        self._net_g = self.net_g.module if hasattr(self.net_g, "module") else self.net_g

        self.writer = None
        self.writer_eval = None
        if rank == 0 and _HAVE_TENSORBOARD:
            self.writer = SummaryWriter(log_dir=model_dir)
            self.writer_eval = SummaryWriter(log_dir=os.path.join(model_dir, "eval"))
        self._registry_run = None

    # ------------------------------------------------------------------ data
    def build_loaders(self):
        from torch.utils.data import DataLoader
        hps = self.hps
        train_dataset = self._loader_cls(hps.data.training_files, hps.data)
        train_sampler = self._sampler_cls(
            train_dataset, hps.train.batch_size,
            [32, 300, 400, 500, 600, 700, 800, 900, 1000],
            num_replicas=self.world_size, rank=self.rank, shuffle=True)
        collate = self._collate_cls()
        train_loader = DataLoader(
            _NumpyDataset(train_dataset), num_workers=self.workers,
            shuffle=False, pin_memory=self.device.type == "cuda",
            collate_fn=collate, batch_sampler=train_sampler)
        eval_loader = None
        if self.rank == 0:
            eval_dataset = self._loader_cls(hps.data.validation_files, hps.data)
            eval_loader = DataLoader(
                _NumpyDataset(eval_dataset), num_workers=self.workers,
                shuffle=False, batch_size=hps.train.batch_size,
                pin_memory=self.device.type == "cuda",
                drop_last=False, collate_fn=collate)
        return train_loader, eval_loader

    # -------------------------------------------------------------- checkpoint
    def resume_or_warmstart(self, pretrained_path="", train_batches=0):
        hps = self.hps
        try:
            _, _, _, epoch_str = train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(self.model_dir, "G_*.pth"),
                self.net_g, self.optim_g)
            _, _, _, epoch_str = train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(self.model_dir, "D0_*.pth"),
                self.mcmbd, self.optim_d)
            _, _, _, epoch_str = train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(self.model_dir, "D1_*.pth"),
                self.msbd, self.optim_d)
            self.epoch = epoch_str
            self.global_step = (epoch_str - 1) * train_batches
        except Exception:
            self.epoch = 1
            self.global_step = 0
        if pretrained_path and self.global_step == 0:
            self.logger.info(f"Trying to load pretrained weights from {pretrained_path}")
            train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(pretrained_path, "G_*.pth"),
                self.net_g, None)
            train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(pretrained_path, "D0_*.pth"),
                self.mcmbd, None)
            train_utils.load_checkpoint(
                train_utils.latest_checkpoint_path(pretrained_path, "D1_*.pth"),
                self.msbd, None)
        self.scheduler_g = self.torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=hps.train.lr_decay, last_epoch=self.epoch - 2)
        self.scheduler_d = self.torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=hps.train.lr_decay, last_epoch=self.epoch - 2)

    # ------------------------------------------------------------------ train
    def train(self, epochs=None, max_steps=None, pretrained_path="",
              test_data_path="test_preproc"):
        hps = self.hps
        if self.device.type == "cpu":
            self.logger.warning(
                "CPU training: this is the honest slow path upstream "
                "refused (train.py asserts CUDA). Expect ~minutes per step "
                "on a full-size model; use --max-steps to smoke-test.")
        if self.world_size > 1 and getattr(hps.train, "grad_acc_steps", 1) > 1:
            hps.train.grad_acc_steps = 1
            self.logger.info("Gradient accumulation not available for Multi-GPU")
        train_loader, eval_loader = self.build_loaders()
        self.resume_or_warmstart(pretrained_path, len(train_loader))
        self._registry_start()
        epochs = hps.train.epochs if epochs is None else epochs
        try:
            for epoch in range(self.epoch, epochs + 1):
                self._train_epoch(epoch, train_loader, eval_loader, test_data_path,
                                  max_steps)
                self.scheduler_g.step()
                self.scheduler_d.step()
                if max_steps is not None and self.global_step >= max_steps:
                    break
        except Exception as exc:
            self._registry_fail(str(exc))
            raise
        self._registry_complete()
        return self.global_step

    def _train_epoch(self, epoch, train_loader, eval_loader, test_data_path,
                     max_steps):
        torch = self.torch
        F = torch.nn.functional
        hps = self.hps
        net_g, mcmbd, msbd = self.net_g, self.mcmbd, self.msbd
        optim_g, optim_d = self.optim_g, self.optim_d
        mel_spectrogram_torch, spec_to_mel_torch = self._mel
        bin_loss, forward_sum = self._loss_fns
        from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
        train_loader.batch_sampler.set_epoch(epoch)
        n_acc_steps = getattr(hps.train, "grad_acc_steps", 1)
        net_g.train()
        mcmbd.train()
        msbd.train()
        for batch_idx, raw_batch in enumerate(train_loader):
            batch = _to_torch_batch(raw_batch, self.device, self.multi_speaker)
            if self.multi_speaker:
                (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden,
                 bert, bert_lens, sid) = batch
            else:
                (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden,
                 bert, bert_lens) = batch
                sid = None
            with torch.cuda.amp.autocast(enabled=getattr(hps.train, "fp16_run", False)):
                mel = spec_to_mel_torch(
                    spec, hps.data.filter_length, hps.data.n_mel_channels,
                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                (y_hat, x1, l_length, attn, attn_logprob, attn_soft,
                 ids_slice, x_mask, z_mask,
                 (z, z_p, m_p, logs_p, m_q, logs_q)) = net_g(
                     x, x_lengths, spec, spec_lengths, mel, tm_hidden, bert,
                     bert_lens, sid=sid)
                y_mel = self._commons.slice_segments(
                    mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.squeeze(1), hps.data.filter_length,
                    hps.data.n_mel_channels, hps.data.sampling_rate,
                    hps.data.hop_length, hps.data.win_length,
                    hps.data.mel_fmin, hps.data.mel_fmax)
                y = self._commons.slice_segments(
                    y, ids_slice * hps.data.hop_length, hps.train.segment_size)
                y_d_hat_r, y_d_hat_g, _, _ = msbd(y, y_hat.detach())
                with torch.cuda.amp.autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g)
                y_d1_hat_r, y_d1_hat_g, _, _ = mcmbd(y, y_hat.detach(), x1.detach())
                with torch.cuda.amp.autocast(enabled=False):
                    loss_disc_d1, losses_disc_d1_r, losses_disc_d1_g = discriminator_loss(
                        y_d1_hat_r, y_d1_hat_g)
                    loss_disc_all = (loss_disc + loss_disc_d1) / n_acc_steps
            if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optim_d.zero_grad()
            self.scaler.scale(loss_disc_all).backward()
            self.scaler.unscale_(optim_d)
            grad_norm_d = (self._commons.clip_grad_value_(msbd.parameters(), None)
                           + self._commons.clip_grad_value_(mcmbd.parameters(), None))
            if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.scaler.step(optim_d)

            with torch.cuda.amp.autocast(enabled=getattr(hps.train, "fp16_run", False)):
                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mcmbd(y, y_hat, x1)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msbd(y, y_hat)
                with torch.cuda.amp.autocast(enabled=False):
                    loss_dur = torch.sum(l_length.float())
                    ctc_loss = forward_sum(attn_logprob=attn_logprob,
                                           in_lens=x_lengths, out_lens=spec_lengths)
                    loss_align = ctc_loss
                    if epoch > getattr(hps.train, "bin_loss_start_epoch", 10 ** 9):
                        bin_loss_scale = min(
                            (epoch - hps.train.bin_loss_start_epoch)
                            / hps.train.bin_loss_warmup_epochs, 1.0)
                        al_match_loss = bin_loss(
                            hard_attention=attn, soft_attention=attn_soft) * bin_loss_scale
                        loss_align += al_match_loss
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = (2 * feature_loss(fmap_f_r, fmap_f_g)
                               + 2 * feature_loss(fmap_s_r, fmap_s_g))
                    loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                    loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                    loss_gen = loss_gen_f + loss_gen_s
                    losses_gen = losses_gen_f + losses_gen_s
                    loss_gen_all = (loss_gen + loss_fm + loss_mel + loss_dur
                                    + loss_kl + loss_align) / n_acc_steps
            if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
                optim_g.zero_grad()
            self.scaler.scale(loss_gen_all).backward()
            self.scaler.unscale_(optim_g)
            grad_norm_g = self._commons.clip_grad_value_(net_g.parameters(), None)
            if ((batch_idx + 1) % n_acc_steps == 0) or (batch_idx + 1 == len(train_loader)):
                self.scaler.step(optim_g)
            self.scaler.update()

            if self.rank == 0:
                if self.global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur,
                              loss_kl, ctc_loss]
                    self.logger.info(f"Train Epoch: {epoch} [{100.0 * batch_idx / len(train_loader):.0f}%]")
                    self.logger.info([x.item() for x in losses]
                                     + [self.global_step, lr])
                    if self.writer is not None:
                        scalar_dict = {
                            "loss/g/total": loss_gen_all,
                            "loss/d/total": loss_disc_all,
                            "learning_rate": lr,
                            "grad_norm_d": grad_norm_d,
                            "grad_norm_g": grad_norm_g,
                            "loss/g/fm": loss_fm, "loss/g/mel": loss_mel,
                            "loss/g/dur": loss_dur, "loss/g/kl": loss_kl,
                            "loss/g/align": loss_align,
                        }
                        scalar_dict.update({f"loss/g/{i}": v
                                            for i, v in enumerate(losses_gen)})
                        scalar_dict.update({f"loss/d_r/{i}": v
                                            for i, v in enumerate(losses_disc_r)})
                        scalar_dict.update({f"loss/d_g/{i}": v
                                            for i, v in enumerate(losses_disc_g)})
                        image_dict = {
                            "slice/mel_org": train_utils.plot_spectrogram_to_numpy(
                                y_mel[0].data.cpu().numpy()),
                            "slice/mel_gen": train_utils.plot_spectrogram_to_numpy(
                                y_hat_mel[0].data.cpu().numpy()),
                            "all/mel": train_utils.plot_spectrogram_to_numpy(
                                mel[0].data.cpu().numpy()),
                            "all/attn": train_utils.plot_alignment_to_numpy(
                                attn[0, 0].data.cpu().numpy()),
                        }
                        train_utils.summarize(
                            writer=self.writer, global_step=self.global_step,
                            images=image_dict, scalars=scalar_dict)
                if self.global_step % hps.train.eval_interval == 0:
                    self.evaluate(eval_loader, test_data_path)
                if self.global_step % getattr(hps.train, "save_interval",
                                               10000) == 0:
                    self.evaluate(eval_loader, test_data_path)
                    g_name = f"G_{self.global_step}.pth"
                    d0_name = f"D0_{self.global_step}.pth"
                    d1_name = f"D1_{self.global_step}.pth"
                    if getattr(hps, "use_latest", False):
                        g_name, d0_name, d1_name = "G_latest.pth", "D0_latest.pth", "D1_latest.pth"
                    train_utils.save_checkpoint(
                        net_g, optim_g, hps.train.learning_rate, epoch,
                        os.path.join(self.model_dir, g_name))
                    train_utils.save_checkpoint(
                        mcmbd, optim_d, hps.train.learning_rate, epoch,
                        os.path.join(self.model_dir, d0_name))
                    train_utils.save_checkpoint(
                        msbd, optim_d, hps.train.learning_rate, epoch,
                        os.path.join(self.model_dir, d1_name))
            self.global_step += 1
            if max_steps is not None and self.global_step >= max_steps:
                break
        if self.rank == 0:
            self.logger.info(f"====> Epoch: {epoch}")

    def evaluate(self, eval_loader, test_data_path="test_preproc"):
        torch = self.torch
        hps = self.hps
        generator = self._net_g
        generator.eval()
        if eval_loader is None:
            generator.train()
            return
        mel_spectrogram_torch, spec_to_mel_torch = self._mel
        with torch.no_grad():
            for _batch_idx, raw_batch in enumerate(eval_loader):
                batch = _to_torch_batch(raw_batch, self.device, self.multi_speaker)
                if self.multi_speaker:
                    (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden,
                     bert, bert_lens, sid) = batch
                else:
                    (x, x_lengths, spec, spec_lengths, y, y_lengths, tm_hidden,
                     bert, bert_lens) = batch
                    sid = None
                x = x[:1]
                x_lengths = x_lengths[:1]
                spec = spec[:1]
                spec_lengths = spec_lengths[:1]
                y = y[:1]
                y_lengths = y_lengths[:1]
                tm_hidden = tm_hidden[:1]
                bert = bert[:1]
                bert_lens = bert_lens[:1]
                if sid is not None:
                    sid = sid[:1]
                break
            image_dict = {}
            audio_dict = {}
            y_hat, attn, mask, *_ = generator.infer(
                x, x_lengths, tm_hidden, bert, bert_lens, sid=sid, max_len=1000)
            y_hat_lengths = mask.sum([1, 2]).long() * hps.data.hop_length
            mel = spec_to_mel_torch(
                spec, hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(), hps.data.filter_length,
                hps.data.n_mel_channels, hps.data.sampling_rate,
                hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
            image_dict["gen/mel"] = train_utils.plot_spectrogram_to_numpy(
                y_hat_mel[0].cpu().numpy())
            audio_dict["gen/audio"] = y_hat[0, :, :y_hat_lengths[0]]
            if os.path.isdir(test_data_path):
                for t_idx, test_file in enumerate(sorted(os.listdir(test_data_path))):
                    if ".pt" not in test_file:
                        continue
                    t_text_norm, t_moji, t_bert = torch.load(
                        os.path.join(test_data_path, test_file),
                        weights_only=False)
                    t_text_norm = t_text_norm.unsqueeze(0).to(self.device)
                    t_text_lengths = torch.LongTensor(
                        [t_text_norm.size(1)]).to(self.device)
                    t_moji = t_moji.squeeze().unsqueeze(0).to(self.device)
                    t_bert_lens = torch.LongTensor([t_bert.size(1)]).to(self.device)
                    t_bert = t_bert.to(self.device)
                    with torch.no_grad():
                        audio, attn, t_mask, _ = generator.infer(
                            t_text_norm, t_text_lengths, t_moji, t_bert,
                            t_bert_lens, noise_scale=.667, noise_scale_w=0.8,
                            length_scale=1.0)
                    test_audio_lengths = t_mask.sum([1, 2]).long() * hps.data.hop_length
                    y_test_mel = mel_spectrogram_torch(
                        audio.squeeze(1).float(), hps.data.filter_length,
                        hps.data.n_mel_channels, hps.data.sampling_rate,
                        hps.data.hop_length, hps.data.win_length,
                        hps.data.mel_fmin, hps.data.mel_fmax)
                    image_dict[f"gen/mel_test{t_idx}"] = (
                        train_utils.plot_spectrogram_to_numpy(
                            y_test_mel[0].cpu().numpy()))
                    audio_dict[f"gen/audio_test{t_idx}"] = (
                        audio[0, :, :test_audio_lengths[0]])
        if self.global_step == 0:
            image_dict.update({"gt/mel": train_utils.plot_spectrogram_to_numpy(
                mel[0].cpu().numpy())})
            audio_dict.update({"gt/audio": y[0, :, :y_lengths[0]]})
        if self.writer_eval is not None:
            train_utils.summarize(
                writer=self.writer_eval, global_step=self.global_step,
                images=image_dict, audios=audio_dict,
                audio_sampling_rate=hps.data.sampling_rate)
        generator.train()

    # ------------------------------------------------------- registry hooks
    def _registry_start(self):
        try:
            from nomorals.training import TrainingRegistry
            registry = TrainingRegistry()
            self._registry_run = registry.start(
                name="vits_finetune",
                base_model="vits-finetuning/{}".format(getattr(
                    self.hps, "model", "base")),
                output_model=self.model_dir,
                dataset_id=getattr(self.hps.data, "training_files", ""),
                backend="vits",
                config={"multi_speaker": self.multi_speaker,
                        "device": str(self.device)},
                metadata={"upstream": "storytold/vits-finetuning"},
            )
        except Exception as exc:
            self.logger.warning(f"TrainingRegistry unavailable: {exc}")
            self._registry_run = None

    def _registry_complete(self):
        if self._registry_run is None:
            return
        try:
            from nomorals.training import TrainingRegistry
            TrainingRegistry().complete(
                self._registry_run, metrics={"global_step": self.global_step},
                output_path=self.model_dir)
        except Exception:
            pass

    def _registry_fail(self, error):
        if self._registry_run is None:
            return
        try:
            from nomorals.training import TrainingRegistry
            TrainingRegistry().fail(self._registry_run, error)
        except Exception:
            pass


def train_from_config(config=None, config_name=None, model_name="vits_finetune",
                      device="cpu", multi_speaker=False, epochs=None,
                      max_steps=None, pretrained_path="", workers=0,
                      test_data_path="test_preproc"):
    """Build a trainer from a config and run it (single-process entry)."""
    hps = load_config(config, config_name)
    model_dir = os.path.join("logs", model_name)
    hps.model_dir = model_dir
    hps.pt_path = pretrained_path
    hps.n_workers = workers
    hps.test_dp = test_data_path
    trainer = VITSTrainer(hps, model_dir, device=device,
                          multi_speaker=multi_speaker, workers=workers)
    return trainer.train(epochs=epochs, max_steps=max_steps,
                         pretrained_path=pretrained_path,
                         test_data_path=test_data_path)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="VITS fine-tuning (iSTFTNet + Avocodo)")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--config-name", type=str,
                        default="ljs_li44_tmbert_ft_small_s1_arpa.json")
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-p", "--pretrained", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--multi-speaker", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--test-dp", type=str, default="test_preproc")
    args = parser.parse_args(argv)
    require_torch("VITS training")
    train_from_config(config=args.config, config_name=args.config_name,
                      model_name=args.model, device=args.device,
                      multi_speaker=args.multi_speaker, epochs=args.epochs,
                      max_steps=args.max_steps, pretrained_path=args.pretrained,
                      workers=args.workers, test_data_path=args.test_dp)


if __name__ == "__main__":
    main()
