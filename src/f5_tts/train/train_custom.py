# training script.

import os
from importlib.resources import files
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from f5_tts.model import CFM, TrainerCustom
from f5_tts.model.dataset import load_dataset, load_multiple_phonemize_datasets
from f5_tts.model.utils import get_tokenizer
from f5_tts.logger import loggerConfig
from torch.utils.tensorboard import SummaryWriter

project_root = str(files("f5_tts").joinpath("../.."))
os.chdir(project_root)  # change working directory to root of project (local editable)
print(f"[INFO] Changed working directory to {project_root}")

@hydra.main(version_base="1.3", config_path=str(files("f5_tts").joinpath("configs")), config_name=None)
def main(model_cfg):
    logger = loggerConfig(logger_name="f5tts_train", log_dir=model_cfg.ckpts.save_dir, log_type='file', is_print=True)
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.get("mel_spec_type", "vocos")

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{model_cfg.datasets.name}"
    wandb_resume_id = None
    logger.add_info('Train setting', f'model name: {exp_name}')

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path
    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    logger.add_info('Train setting', f'Tokenizer set.')
    logger.add_info('Vocab size', f'{vocab_size}') 

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )
    logger.add_info('Train setting', f'Model load.')

    # init trainer
    trainer = TrainerCustom(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("f5_tts").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="CFM-TTS",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )


    train_dataset = load_multiple_phonemize_datasets(model_cfg.datasets.train_paths, model_cfg.datasets.train_metadatas,
                                                     mel_spec_kwargs=model_cfg.model.mel_spec)
    
    # valid_dataset = load_multiple_phonemize_datasets(model_cfg.datasets.valid_paths, model_cfg.datasets.valid_metadatas,
    #                                                  tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()

# accelerate config
# accelerate launch src/f5_tts/train/train_custom.py --config-name F5TTS_zzal_v1.yaml
# accelerate launch --mixed_precision=fp16 src/f5_tts/train/train_custom.py --config-name F5TTS_zzal_v1.yaml