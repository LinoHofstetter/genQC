"""Special extension to `DiffusionPipeline` for source-circuit equivalence conditioning."""


# %% auto #0
__all__ = ["DiffusionPipeline_Equivalence"]

# %% ../../src/pipeline/equivalence_diffusion_pipeline.ipynb
from ..imports import *
from .diffusion_pipeline import DiffusionPipeline
from ..scheduler.scheduler import Scheduler
from ..models.config_model import ConfigModel
from ..utils.config_loader import load_config


class DiffusionPipeline_Equivalence(DiffusionPipeline):
    """
    Diffusion pipeline for learning p(target_circuit | source_circuit, optional_text).

    Target path:
        target_tokens -> embedder.embed(...) -> noisy DDPM target

    Source path:
        source_tokens -> (optional cfg-drop in token space) -> embedder.embed(...) -> source_encoder(...) -> s_emb

    Final condition:
        c_total = [text_embedding, source_embedding]
    """

    def __init__(
        self,
        scheduler, # handles noising and denoising steps
        model, # actual denoising model (e.g. U-Net)
        text_encoder, # encodes text conditions (e.g. OpenCLIP)
        embedder, # maps circuit token tensors into embedding space 
        source_encoder, # this is the new component compared to DiffusionPipeline_Compilation
        device,
        enable_guidance_train: bool = True,
        guidance_train_p: float = 0.1,
        cached_text_enc: bool = True,
        non_blocking: bool = False,
    ):
        super().__init__(
            scheduler=scheduler,
            model=model,
            text_encoder=text_encoder,
            embedder=embedder,
            device=device,
            enable_guidance_train=enable_guidance_train,
            guidance_train_p=guidance_train_p,
            cached_text_enc=cached_text_enc,
            #non_blocking=non_blocking,
        )
        self.non_blocking = non_blocking
        self.source_encoder = source_encoder.to(device)
        self.trainables.append(self.source_encoder)

    # ------------------------------------------------------------------
    # IO / config
    # ------------------------------------------------------------------

    def params_config(self, save_path: str):
        config = super().params_config(save_path)
        config["source_encoder"] = self.source_encoder.get_config(
            save_path + "source_encoder/",
            without_metadata=True,
        )
        return config

    def store_pipeline(self, config_path: str, save_path: str):
        super().store_pipeline(config_path=config_path, save_path=save_path) # lets parent save all the standard components

        # additionally save the source encoder config and weights
        if exists(save_path):
            os.makedirs(save_path + "source_encoder/", exist_ok=True)
            self.source_encoder.store_model(
                config_path=None,
                save_path=save_path + "source_encoder/",
            )

    @staticmethod
    def from_config_file(config_path, device: torch.device, save_path: str = None):
        """
        Mirrors DiffusionPipeline.from_config_file(...) but also loads source_encoder.
        """
        config = load_config(config_path + "config.yaml")
        if "params" in config:
            config = config["params"]

        scheduler = Scheduler.from_config(config["scheduler"], device=device, save_path=save_path)
        model = ConfigModel.from_config(config["model"], device=device, save_path=save_path)

        # text encoder is usually frozen in the current stack
        text_encoder = ConfigModel.from_config(
            config["text_encoder"],
            device=device,
            save_path=save_path,
        )

        if "embedder" in config and exists(config["embedder"]):
            embedder = ConfigModel.from_config(
                config["embedder"],
                device=device,
                save_path=save_path,
            )
        else:
            # legacy behavior already used elsewhere in the repo
            embedder = model

        source_encoder = ConfigModel.from_config(
            config["source_encoder"],
            device=device,
            save_path=save_path,
            is_frozen=False,
        )

        return DiffusionPipeline_Equivalence(
            scheduler=scheduler,
            model=model,
            text_encoder=text_encoder,
            embedder=embedder,
            source_encoder=source_encoder,
            device=device,
            enable_guidance_train=config.get("enable_guidance_train", True),
            guidance_train_p=config.get("guidance_train_p", 0.1),
            cached_text_enc=config.get("cached_text_enc", True),
            non_blocking=config.get("non_blocking", False),
        )

    # ------------------------------------------------------------------
    # source-circuit conditioning helpers
    # ------------------------------------------------------------------

    def empty_source_fn(self, source_tokens: torch.Tensor):
        """
        Empty unconditional source condition in raw token space.
        Uses all-zero tokens, which correspond to background/no-gate in CircuitTokenizer.
        """
        return torch.zeros_like(source_tokens) # tensor with same shape, all zeros

    def _source_tokens_to_latents(self, source_tokens: torch.Tensor):
        """
        Converts raw source tokens [b,s,t] to embedded source latents [b,c,s,t].
        """
        source_tokens = source_tokens.to(self.device, non_blocking=self.non_blocking)
        return self.embedder.embed(source_tokens)

    def prepare_source_emb(
        self,
        source_tokens: torch.Tensor,
        enable_guidance: bool = True,
        negative_source: Optional[torch.Tensor] = None,
    ):
        """
        Builds source-circuit conditioning sequence.

        Input:
            source_tokens: [b,s,t]

        Returns:
            s_emb: [b or 2b, seq_source, cond_emb_size]
        """
        if not exists(source_tokens):
            return None

        source_tokens = source_tokens.to(self.device, non_blocking=self.non_blocking)

        if enable_guidance: # in case of classifier-free guidance, we need to prepare negative samples for the source circuit conditions as well, by applying cfg dropout in token space and encoding the resulting "empty" source tokens through the same embedding + source encoder path. The resulting negative source embeddings are then concatenated to the real source embeddings along the batch dimension, effectively doubling the batch size for the source conditions and allowing us to apply classifier-free guidance in the same way as for text conditions.
            if exists(negative_source):
                negative_source = negative_source.to(self.device, non_blocking=self.non_blocking)
            else:
                negative_source = self.empty_source_fn(source_tokens)

            # check if this is mathematically what we want
            source_tokens = torch.cat([negative_source, source_tokens], dim=0)

        source_latents = self._source_tokens_to_latents(source_tokens)
        s_emb = self.source_encoder(source_latents)
        return s_emb

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        latents,
        c,
        source_tokens,
        g,
        negative_c=None,
        negative_source=None,
        no_bar=False,
    ):
        latents = latents.to(self.device)
        c = c.to(self.device)
        source_tokens = source_tokens.to(self.device)

        return self.denoising(
            latents,
            c=c,
            source_tokens=source_tokens,
            negative_c=negative_c,
            negative_source=negative_source,
            enable_guidance=True,
            g=g,
            no_bar=no_bar,
        )

    @torch.no_grad()
    def denoising(
        self,
        latents,
        c,
        source_tokens,
        negative_c=None,
        negative_source=None,
        enable_guidance=True,
        g=1.0,
        t_start_index=0,
        no_bar=False,
        return_predicted_x0=False,
    ):
        """
        Overrides the parent denoising path so we can inject source-circuit conditioning.
        """
        self.model.eval()
        self.text_encoder.eval()
        self.source_encoder.eval()
        self.scheduler.to(self.device)

        latents = latents.to(self.device)

        # parent helper for text guidance
        c_emb = self.prepare_c_emb(c, negative_c, enable_guidance)

        # new helper for source-circuit conditioning
        s_emb = self.prepare_source_emb(source_tokens, enable_guidance, negative_source)

        if exists(s_emb):
            c_emb = torch.cat([c_emb, s_emb], dim=1)

        timesteps = self.scheduler.timesteps[t_start_index:]

        pred_x0_all = [] if return_predicted_x0 else None
        iterator = timesteps if no_bar else self.progress_bar(iterable=timesteps, desc="Denoising", unit=" step")

        for ts in iterator:
            latents, pred_x0 = super().denoising_step(
                latents,
                ts,
                c_emb=c_emb,
                enable_guidance=enable_guidance,
                g=g,
            )
            if return_predicted_x0:
                pred_x0_all.append(pred_x0)

        if return_predicted_x0:
            return latents, torch.stack(pred_x0_all, dim=1)

        return latents

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train_step(self, data, train, **kwargs):
        """
        Expected batch:
            (target_tokens, y, source_tokens)

        target_tokens: [b, s, t]
        y:             tokenized prompts (text = gate set)
        source_tokens: [b, s, t]
        """
        target_tokens, y, source_tokens = data
        b, s, t = target_tokens.shape

        # --------------------------------------------------------------
        # target path: clean target circuit -> embedded target latents
        # --------------------------------------------------------------
        target_tokens = target_tokens.to(self.device, non_blocking=self.non_blocking)
        target_latents = self.embedder.embed(target_tokens)

        # --------------------------------------------------------------
        # text + source token path
        # --------------------------------------------------------------
        y = y.to(self.device, non_blocking=self.non_blocking)
        source_tokens = source_tokens.to(self.device, non_blocking=self.non_blocking)

        # classifier-free guidance dropout on CONDITIONS in token space
        if self.enable_guidance_train and train:
            rnd_y, rnd_s = (
                torch.empty((2 * b,), device=self.device)
                .bernoulli_(p=1.0 - self.guidance_train_p)
                .type(torch.int64)
                .chunk(2, dim=0) # ?? this chunk is not in parent -> check
            )

            y = self.cfg_drop(y, self.empty_token_fn(y), rnd_y)
            source_tokens = self.cfg_drop(source_tokens, self.empty_source_fn(source_tokens), rnd_s)

        # --------------------------------------------------------------
        # encode conditions
        # --------------------------------------------------------------
        y_emb = self.text_encoder(y, pool=False)

        source_latents = self._source_tokens_to_latents(source_tokens)
        s_emb = self.source_encoder(source_latents)

        c_total = torch.cat([y_emb, s_emb], dim=1)

        # --------------------------------------------------------------
        # standard DDPM target corruption
        # --------------------------------------------------------------
        timesteps = torch.randint(
            low=0,
            high=self.scheduler.num_train_timesteps,
            size=(b,),
            device=self.device,
            dtype=torch.int64,
        )

        noise = torch.randn(target_latents.shape, device=self.device)
        noisy_target = self.scheduler.add_noise(target_latents, noise, timesteps, train=train)

        # predict epsilon conditioned on text + source circuit
        eps = self.model(noisy_target, timesteps, c_total)

        # standard denoising loss
        loss = self.loss_fn(eps, noise)
        return loss