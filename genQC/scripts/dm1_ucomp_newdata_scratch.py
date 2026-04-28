# genQC/scripts/train_dm1_discrete_unitary_compilation.py
#
# Single-file end-to-end script for:
#   raw dataset generation  -> library dataset object  -> dataloaders  -> DM1 training from scratch
#
# IMPORTANT:
# 1) This script is for the DM1 / discrete-gate-set-only compilation path.
# 2) It uses the Qiskit backend because that backend implements the full
#    random-circuit -> optimize -> randomize -> unitary -> backend_to_genqc path
#    needed by circuits_generation.py.
# 3) This is set up for a SMOKE TEST:
#    - small dataset
#    - few epochs
#    - explicit loss plot saved at the end
# 4) Run from the repository root, e.g.
#       python genQC/scripts/train_dm1_discrete_unitary_compilation.py
#    or if needed:
#       PYTHONPATH=. python genQC/scripts/train_dm1_discrete_unitary_compilation.py

from pathlib import Path
import torch
import torch.nn as nn

from genQC.platform.backends.circuits_qiskit import CircuitsQiskitBackend
from genQC.platform.tokenizer.circuits_tokenizer import CircuitTokenizer
from genQC.platform.circuits_generation import (
    CircuitConditionType,
    generate_circuit_dataset,
)
from genQC.dataset.circuits_dataset import CircuitsConfigDataset

from genQC.pipeline.compilation_diffusion_pipeline import DiffusionPipeline_Compilation
from genQC.pipeline.pipeline import CheckpointCB
from genQC.scheduler.scheduler_ddpm import DDPMScheduler

from genQC.models.config_model import ConfigModel
from genQC.models.frozen_open_clip import CachedFrozenOpenCLIPEmbedder
from genQC.models.unet_qc import QC_Compilation_UNet, QC_Cond_UNet
from genQC.models.unitary_encoder import Unitary_encoder_config




def build_pipeline_from_scratch(
    device: torch.device,
    num_clrs: int,
) -> DiffusionPipeline_Compilation:
    """
    Build a DM1 compilation pipeline from scratch.

    Components:
      - scheduler: DDPM scheduler for training-time noising
      - text_encoder: cached frozen OpenCLIP text encoder for prompt conditioning
      - model: QC_Compilation_UNet (DM1 compilation denoiser)
      - embedder: frozen QC_Cond_UNet used only for .embed(...) on discrete circuit tensors

    Notes:
      - The compilation training step calls:
            latents = self.embedder.embed(latents)
            y_emb   = self.text_encoder(y, pool=False)
            eps     = self.model(noisy_latents, timesteps, y_emb, U=U)
      - So the embedder must expose .embed(...), the text encoder must support
        caching + empty_token, and the model must accept U in forward(...).
    """

    # ------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------
    scheduler = DDPMScheduler(
        device=device,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        input_perturbation=0.1,
        prediction_type="epsilon",
        enable_zero_terminal_snr=True,
    )

    # ------------------------------------------------------------
    # Text encoder
    #
    # This is the same kind of cached OpenCLIP text encoder expected by the
    # cached dataset stack. It provides:
    #   - empty_token
    #   - tokenize_and_push_to_device(...)
    #   - generate_cache(...)
    #   - forward(...) on cached prompt indices
    # ------------------------------------------------------------
    text_encoder = CachedFrozenOpenCLIPEmbedder(
        arch="ViT-B-32",
        version="datacomp_xl_s13b_b90k",
        max_length=77,
        freeze=True,
        layer="penultimate",
        enable_cache_token_limit=True,
    ).to(device)

    # ------------------------------------------------------------
    # Shared model hyperparameters
    #
    # num_clrs = number of discrete token classes including the zero/background
    # token. With a tokenizer mapping i+1 for each gate, this is len(gate_pool)+1.
    # ------------------------------------------------------------
    clr_dim = 8
    cond_emb_size = 512
    t_emb_size = 128

    model_features = [32, 32, 64]
    num_heads = [8, 8, 2]
    num_res_blocks = [2, 2, 4]
    transformer_depths = [1, 2, 1]

    # ------------------------------------------------------------
    # Unitary encoder config used inside QC_Compilation_UNet
    # ------------------------------------------------------------
    unitary_encoder_config = Unitary_encoder_config(
        cond_emb_size=cond_emb_size,
        model_features=None,
        num_heads=8,
        transformer_depths=[4, 4],
        dropout=0.1,
    )

    # ------------------------------------------------------------
    # DM1 compilation denoiser model
    #
    # This is the model that predicts noise eps from:
    #   noisy_latents, diffusion timestep, text embedding, and U
    # ------------------------------------------------------------
    model = QC_Compilation_UNet(
        model_features=model_features,
        clr_dim=clr_dim,
        num_clrs=num_clrs,
        t_emb_size=t_emb_size,
        cond_emb_size=cond_emb_size,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        transformer_depths=transformer_depths,
        unitary_encoder_config=unitary_encoder_config,
    ).to(device)

    # ------------------------------------------------------------
    # Embedder
    #
    # The training step needs an object with:
    #   embedder.embed(latents)
    #
    # QC_Cond_UNet already provides .embed(...) for discrete circuit tensors.
    # For this smoke test, we use a separate frozen instance purely as embedder.
    # ------------------------------------------------------------
    embedder = QC_Cond_UNet(
        model_features=model_features,
        clr_dim=clr_dim,
        num_clrs=num_clrs,
        t_emb_size=t_emb_size,
        cond_emb_size=cond_emb_size,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        transformer_depths=transformer_depths,
    ).to(device)

    # Freeze because we only want its discrete token embedding behavior here.
    embedder.freeze(True)

    pipeline = DiffusionPipeline_Compilation(
        scheduler=scheduler,
        model=model,
        text_encoder=text_encoder,
        embedder=embedder,
        device=device,
        enable_guidance_train=True,
        guidance_train_p=0.1,
        cached_text_enc=True,
    )

    return pipeline


def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------

    # Directory where the generated dataset will be saved
    DATASET_DIR = Path("runs/dm1_discrete_compile_q4_smoketest")

    # Directory where the trained pipeline and monitoring artifacts will be saved
    TRAIN_OUT_DIR = Path("runs/dm1_discrete_compile_q4_smoketest_trained")

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # SMOKE TEST SETTINGS
    #
    # Keep this small so you can quickly verify:
    #   - the pipeline runs
    #   - gradients flow
    #   - the loss starts decreasing
    # ------------------------------------------------------------
    num_qubits = 4
    min_gates = 4
    max_gates = 24

    # Small dataset so that a quick overfitting-style smoke test is feasible
    total_samples =  128
    batch_samples =  64
    n_jobs = 1

    # DM1 / discrete-only gate pool
    # gate_pool = ["h","x","y","z","s","sdg","t","tdg","cx","cz","swap","ccx",] full discrete gate set for 4 qubits
    gate_pool = ["h", "x", "z", "cx", "ccx"]  


    # Small training setup for checking that loss decreases
    batch_size = 8
    p_valid = 0.10
    num_epochs = 5
    learning_rate = 1e-4

    # Save a checkpoint every epoch during the smoke test
    checkpoint_every = 1

    # ------------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # BUILD BACKEND + TOKENIZER
    # ------------------------------------------------------------

    # Use Qiskit backend because it supports:
    #   rnd_circuit, optimize_circuit, randomize_params,
    #   get_unitary, backend_to_genqc
    backend = CircuitsQiskitBackend()

    # The tokenizer maps each gate name to an integer token.
    # We start at 1 because 0 is reserved as the background / no-gate token.
    tokenizer = CircuitTokenizer({gate: i + 1 for i, gate in enumerate(gate_pool)})

    # num_clrs for the embedding models:
    # token ids are 0..len(gate_pool), where 0 is background/no-gate
    num_clrs = len(gate_pool) + 1

    # ------------------------------------------------------------
    # GENERATE RAW UNITARY-COMPILATION DATA
    # ------------------------------------------------------------

    # Even though we want the discrete-only DM1 path, generate_circuit_dataset(...)
    # currently requires return_params=True for UNITARY generation.
    #
    # That means it will still return:
    #   x ... encoded circuit tensors
    #   y ... prompt strings
    #   U ... target unitaries
    #   p ... parameter tensors
    #
    # For a discrete-only gate pool, p is not needed for DM1 training,
    # so we ignore it afterward.
    print("\n[STEP 1] Generating raw dataset ...")
    x, y, U, p_unused = generate_circuit_dataset(
        backend=backend,
        tokenizer=tokenizer,
        condition=CircuitConditionType.UNITARY,
        total_samples=total_samples,
        num_of_qubits=num_qubits,
        min_gates=min_gates,
        max_gates=max_gates,
        batch_samples=batch_samples,
        n_jobs=n_jobs,
        # float32 is safer than float16 for training/debugging;
        # can switch to float16 later if memory becomes an issue.
        unitary_dtype=torch.float32,
        # Fix the sub-gate pool to the full discrete gate set.
        min_sub_gate_pool_cnt=len(gate_pool),
        max_sub_gate_pool_cnt=len(gate_pool),
        fixed_sub_gate_pool=gate_pool,
        max_num_params=None,
        filter_unique=True,
        optimized=True,
        post_randomize_params=True,
        return_params=True,
    )

    print("Raw generation finished.")
    print("x shape       :", tuple(x.shape))
    print("y shape       :", tuple(y.shape))
    print("U shape       :", tuple(U.shape))
    print("p_unused shape:", tuple(p_unused.shape))

    # ------------------------------------------------------------
    # WRAP RAW ARRAYS INTO A LIBRARY DATASET OBJECT
    # ------------------------------------------------------------

    # For the DM1 discrete compilation pipeline, the training batch must be:
    #   (x, y, U)
    #
    # Therefore the dataset store_dict must contain exactly these fields.
    #
    # We do NOT include "params" here, because DiffusionPipeline_Compilation
    # expects train_step(data) to unpack:
    #   latents, y, U = data
    print("\n[STEP 2] Wrapping raw arrays in CircuitsConfigDataset ...")
    dataset = CircuitsConfigDataset(
        device=torch.device("cpu"),
        store_dict={
            "x": "tensor",
            "y": "numpy",
            "U": "tensor",
        },
        dataset_to_gpu=False,
        optimized=True,
        random_samples=int(x.shape[0]),
        num_of_qubits=num_qubits,
        min_gates=min_gates,
        max_gates=max_gates,
        max_params=0,
        gate_pool=gate_pool,
    )

    dataset.x = x
    dataset.y = y
    dataset.U = U

    dataset.memory_summary()

    # ------------------------------------------------------------
    # SAVE DATASET IN LIBRARY FORMAT
    # ------------------------------------------------------------

    print("\n[STEP 3] Saving dataset ...")
    dataset_config_path = str(DATASET_DIR / "config.yaml")
    dataset_save_path = str(DATASET_DIR / "dataset")

    dataset.save_dataset(
        config_path=dataset_config_path,
        save_path=dataset_save_path,
    )

    print(f"Saved dataset config to : {dataset_config_path}")
    print(f"Saved dataset tensors to: {dataset_save_path}")

    # ------------------------------------------------------------
    # RELOAD DATASET ONCE TO VERIFY THE SAVE/LOAD PATH
    # ------------------------------------------------------------

    print("\n[STEP 4] Reloading dataset from disk ...")
    dataset = CircuitsConfigDataset.from_config_file(
        config_path=dataset_config_path,
        device=torch.device("cpu"),
        save_path=dataset_save_path,
    )

    dataset.memory_summary()

    # ------------------------------------------------------------
    # BUILD FRESH DM1 COMPILATION PIPELINE FROM SCRATCH
    # ------------------------------------------------------------

    print("\n[STEP 5] Building DM1 compilation pipeline from scratch ...")
    pipeline = build_pipeline_from_scratch(
        device=device,
        num_clrs=num_clrs,
    )

    # ------------------------------------------------------------
    # BUILD DATALOADERS
    # ------------------------------------------------------------

    # CircuitsConfigDataset inherits CachedOpenCLIPDataset.
    # That means the string prompts in y will be tokenized and cached
    # through the pipeline's text encoder when we build dataloaders.
    print("\n[STEP 6] Building dataloaders with prompt caching ...")
    dls = dataset.get_dataloaders(
        batch_size=batch_size,
        text_encoder=pipeline.text_encoder,
        p_valid=p_valid,
        balance_max=None,
        max_samples=None,
    )

    # ------------------------------------------------------------
    # PREPARE TRAINING
    # ------------------------------------------------------------

    # compile() expects a loss class / constructor, not an already-created object,
    # because internally it calls loss_fn().
    #
    # We also attach a checkpoint callback so that even the smoke test
    # saves intermediate pipeline states every epoch.
    print("\n[STEP 7] Compiling training setup ...")
    pipeline.compile(
        optim_fn=torch.optim.AdamW,
        loss_fn=nn.MSELoss,
        metrics=[],
        cbs=[
            CheckpointCB(
                ck_interval=checkpoint_every,
                ck_path=str(TRAIN_OUT_DIR) + "/",
            )
        ],
        compile_model=False,
    )

    # Store dataset metadata inside the pipeline config.
    pipeline.add_config = {
        "dataset": dataset.get_config(
            save_path=dataset_save_path,
            without_metadata=False,
        )
    }

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\n[STEP 8] Starting smoke-test training ...")
    pipeline.fit(
        num_epochs=num_epochs,
        data_loaders=dls,
        lr=learning_rate,
        lr_sched=None,
        log_summary=True,
    )

    # ------------------------------------------------------------
    # EXPLICITLY SAVE LOSS PLOT
    # ------------------------------------------------------------

    print("\n[STEP 9] Saving loss plot ...")
    fig = pipeline.fit_summary(return_fig=True)
    fig.savefig(TRAIN_OUT_DIR / "loss_curve.png", dpi=150)

    # Also print the last few losses for quick terminal inspection
    if hasattr(pipeline, "fit_losses") and len(pipeline.fit_losses) > 0:
        print("Last 10 training losses:")
        print(pipeline.fit_losses[-10:])

    if hasattr(pipeline, "fit_valid_losses") and len(pipeline.fit_valid_losses) > 0:
        print("Validation loss points:")
        print(pipeline.fit_valid_losses)

    # ------------------------------------------------------------
    # SAVE TRAINED PIPELINE
    # ------------------------------------------------------------

    print("\n[STEP 10] Saving trained pipeline ...")
    train_out_dir = str(TRAIN_OUT_DIR) + "/"

    pipeline.store_pipeline(
        config_path=train_out_dir,
        save_path=train_out_dir,
    )

    # ------------------------------------------------------------
    # EXTRA SAVE: force creation of text_encoder.safetensors
    #
    # CachedFrozenOpenCLIPEmbedder.store_model() intentionally does not
    # write local weights, so we call the base ConfigModel.store_model(...)
    # directly to force saving the state dict.
    # ------------------------------------------------------------
    print("[STEP 10b] Forcing save of text_encoder.safetensors ...")
    ConfigModel.store_model(
        pipeline.text_encoder,
        config_path=None,
        save_path=train_out_dir + "text_encoder",
    )

    print("Done.")
    print(f"Trained pipeline saved to: {train_out_dir}")
    print(f"Loss plot saved to: {TRAIN_OUT_DIR / 'loss_curve.png'}")


if __name__ == "__main__":
    main()