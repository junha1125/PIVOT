import os, math
import torch
import matplotlib.pyplot as plt
from transformers import TrainerCallback
import numpy as np
from PIL import Image
from PIL import ImageEnhance


class _FeatureTap:
    def __init__(self):
        self.tensor = None
        self.handle = None
    def attach(self, module):
        def _hook(_, __, out):
            if isinstance(out, tuple):
                out = out[0]
            self.tensor = out
            if torch.is_tensor(out):
                out.retain_grad()  
        self.handle = module.register_forward_hook(_hook)
        return self
    def grad(self):
        if self.tensor is None:
            return None
        else:
            return self.tensor.grad
    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class GradVizCallback(TrainerCallback):
    def __init__(self, tap, every_n_steps=10, out_dir="grad_viz", reduce="l1", original_iamge_path=None):
        self.tap = tap
        self.every_n_steps = every_n_steps
        self.out_dir = out_dir
        self.reduce = reduce
        self.original_iamge_path = original_iamge_path
        self.image_name = os.path.basename(original_iamge_path).split('.')[0]
        os.makedirs(self.out_dir, exist_ok=True)

    def _reduce_tokens(self, g_tok):  # [T_img, D] -> [T_img]
        if self.reduce == "l2":
            return g_tok.pow(2).sum(dim=-1).sqrt()
        return g_tok.abs().sum(dim=-1)

    def _save_heatmaps(self, step, g):  # g: [B, T_img, D] or [T_img, D]
        if g is None:
            return
        g = g.detach().float().cpu()
        if g.dim() == 2:
            g = g.unsqueeze(0)  # -> [1, T_img, D]
        B, T, D = g.shape
        b = 0 
        
        s = self._reduce_tokens(g[b])   # [T_img]
        T_img = s.numel()
        grid = int(math.sqrt(T_img))
        assert grid * grid == T_img, f"T_img({T_img}) is not a perfect square"
        
        heat = s.reshape(grid, grid)

        # Remove border values to eliminate the effect of register (https://arxiv.org/abs/2309.16588)
        H, W = heat.shape
        src = heat.clone()  
        cols = torch.arange(W, device=src.device)
        choice = torch.randint(0, 2, (W,), device=src.device)  
        heat[0, cols] = src[1 + choice, cols]
        choice = torch.randint(0, 2, (W,), device=src.device)
        heat[H - 1, cols] = src[H - 2 - choice, cols]
        rows = torch.arange(H, device=src.device)
        choice = torch.randint(0, 2, (H,), device=src.device)
        heat[rows, 0] = src[rows, 1 + choice]
        choice = torch.randint(0, 2, (H,), device=src.device)
        heat[rows, W - 1] = src[rows, W - 2 - choice]

        path = os.path.join(self.out_dir, f"{self.image_name}_step{step:02d}.png")
        print(f"Saving heatmap to {path}")
        print(f"original image path: {self.original_iamge_path}")

        mn, mx = float(heat.min()), float(heat.max())
        if mx > mn:
            heat = (heat - mn) / (mx - mn)

        GAMMA = 1.1
        heat = heat.clamp(0.0, 1.0).pow(GAMMA)  

        plt.figure()
        bg_pil = Image.open(self.original_iamge_path).convert("RGB")
        bg_pil = bg_pil.resize((384, 384), Image.BILINEAR)
        enhancer = ImageEnhance.Brightness(bg_pil)
        bg_pil = enhancer.enhance(0.7)
        bg = np.array(bg_pil)

        heat_pil = Image.fromarray((heat.numpy() * 255).astype(np.uint8))
        heat_pil = heat_pil.resize((bg.shape[1], bg.shape[0]), Image.BILINEAR) 
        heat_resized = np.array(heat_pil).astype(np.float32) / 255.0  

        overlay_alpha = 0.6
        
        dpi = 150
        fig = plt.figure(figsize=(bg.shape[1]/dpi, bg.shape[0]/dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])  
        ax.axis("off")

        ax.imshow(bg)
        ax.imshow(
            heat_resized, cmap="jet", alpha=overlay_alpha,
            interpolation="nearest",
            extent=[0, bg.shape[1], bg.shape[0], 0]
        )

        plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor='none')
        plt.close(fig)


    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % self.every_n_steps) != 0:
            return
        if args.process_index != 0:
            return
        if  state.global_step >= 35:
            raise NotImplementedError("Few iterations required for Grad-CAM visualization")
        self._save_heatmaps(state.global_step, self.tap.grad())


def unwraped_and_tapping(trainer, training_args, original_iamge_path):
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)

    def _find_mm_projector(m):
        base = m.get_model() if hasattr(m, "get_model") and callable(m.get_model) else m
        if hasattr(base, "mm_projector"):
            return getattr(base, "mm_projector")
        for name, module in base.named_modules():
            if "mm_projector" in name:
                return module
        raise RuntimeError("mm_projector not found")

    projector = _find_mm_projector(unwrapped)

    tap = _FeatureTap().attach(projector)

    viz_cb = GradVizCallback(
        tap=tap,
        every_n_steps=10,
        out_dir=os.path.join(training_args.output_dir, "grad_viz"),
        reduce="l1", 
        original_iamge_path=original_iamge_path
    )
    trainer.add_callback(viz_cb)
    return trainer
