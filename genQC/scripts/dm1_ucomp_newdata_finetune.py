# train_dm1_discrete_unitary_compilation.py
#
# Single-file end-to-end script for:
#   raw dataset generation  -> library dataset object  -> dataloaders  -> DM1 training
#
# IMPORTANT:
# 1) This script is for the DM1 / discrete-gate-set-only compilation path.
# 2) It uses the Qiskit backend because that backend implements the full
#    random-circuit -> optimize -> randomize -> unitary -> backend_to_genqc path
#    needed by circuits_generation.py.
# 3) You must point CHECKPOINT_DIR to an EXISTING compatible DM1 compilation pipeline.
#    The tokenizer gate order must match what that checkpoint expects.

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


def ensure_dir_suffix(path_str: str) -> str:
    """
    DiffusionPipeline.from_config_file(...) expects a directory-like path and
    internally appends 'config.yaml', so we make sure the string ends with '/'.
    """
    return path_str if path_str.endswith("/") else path_str + "/"


def main():
    # ------------------------------------------------------------
    # USER SETTINGS
    # ------------------------------------------------------------

    # Directory where the generated dataset will be saved
    DATASET_DIR = Path("runs/dm1_discrete_compile_q4")

    # Directory where the fine-tuned pipeline will be saved
    TRAIN_OUT_DIR = Path("runs/dm1_discrete_compile_q4_trained")

    # MUST point to an existing DM1 compilation pipeline directory.
    # That directory should contain the saved pipeline config/weights.
    CHECKPOINT_DIR = "PATH_TO_EXISTING_DM1_COMPILATION_PIPELINE/"

    # Training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Raw dataset generation settings
    num_qubits = 4
    min_gates = 4
    max_gates = 24

    # total_samples is the target budget for raw generation;
    # after uniqueness filtering the final number may be smaller.
    total_samples = 5000
    batch_samples = 256
    n_jobs = 4

    # DM1 / discrete-only gate pool
    # Keep this compatible with the checkpoint's expected tokenization/order.
    gate_pool = [
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "cz",
        "swap",
        "ccx",
    ]

    # Training settings
    batch_size = 32
    p_valid = 0.05
    num_epochs = 10
    learning_rate = 1e-4

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
    # so we will simply ignore it afterward.
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
        # you can switch to float16 later if memory becomes an issue.
        unitary_dtype=torch.float32,
        # We fix the sub-gate pool to the full discrete gate set.
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
        # No meaningful params tensor is stored in this discrete-only dataset object.
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
    # LOAD EXISTING DM1 COMPILATION PIPELINE
    # ------------------------------------------------------------

    print("\n[STEP 5] Loading DM1 compilation pipeline ...")
    checkpoint_dir = ensure_dir_suffix(CHECKPOINT_DIR)

    pipeline = DiffusionPipeline_Compilation.from_config_file(
        config_path=checkpoint_dir,
        device=device,
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
    print("\n[STEP 7] Compiling training setup ...")
    pipeline.compile(
        optim_fn=torch.optim.AdamW,
        loss_fn=nn.MSELoss,
        metrics=[],
        compile_model=False,
    )

    # Optionally store dataset metadata inside the pipeline config.
    pipeline.add_config = {
        "dataset": dataset.get_config(
            save_path=dataset_save_path,
            without_metadata=False,
        )
    }

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    print("\n[STEP 8] Starting training ...")
    pipeline.fit(
        num_epochs=num_epochs,
        data_loaders=dls,
        lr=learning_rate,
        lr_sched=None,
        log_summary=True,
    )

    # ------------------------------------------------------------
    # SAVE TRAINED PIPELINE
    # ------------------------------------------------------------

    print("\n[STEP 9] Saving trained pipeline ...")
    train_out_dir = ensure_dir_suffix(str(TRAIN_OUT_DIR))

    pipeline.store_pipeline(
        config_path=train_out_dir,
        save_path=train_out_dir,
    )

    print("Done.")
    print(f"Trained pipeline saved to: {train_out_dir}")


if __name__ == "__main__":
    main()