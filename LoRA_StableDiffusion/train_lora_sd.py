import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import get_peft_model, LoraConfig

IMAGE_FOLDER = "./training_data/gurukul_hologram_style"
METADATA_FILE = "./training_data/metadata.jsonl"

class GurukulDataset(Dataset):
    def __init__(self, metadata_path, image_folder, tokenizer, image_size=512):
        self.samples = []
        with open(metadata_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.samples.append(item)

        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = os.path.join(self.image_folder, item["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        prompt = item["text"]
        tokens = self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        return {
            "pixel_values": image,
            "input_ids": tokens["input_ids"].squeeze(0)
        }

def train(unet, text_encoder, vae, noise_scheduler, dataloader, device, weight_dtype):
    vae.eval()
    text_encoder.eval()
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(weight_dtype != torch.float32))  # ✅ enable AMP only if not float32

    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
        input_ids = batch["input_ids"].to(device)

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = torch.clamp(noisy_latents, -10.0, 10.0).to(device, dtype=weight_dtype)

        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0].to(device, dtype=weight_dtype)

        print(f"Step {step} | Latents Max: {latents.abs().max().item():.4f}, Noise Max: {noise.abs().max().item():.4f}, Noisy Latents Max: {noisy_latents.abs().max().item():.4f}")

        with torch.cuda.amp.autocast(enabled=(weight_dtype != torch.float32)):  # ✅ AMP context
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(noise_pred, noise, reduction="mean")

        if torch.isnan(loss):
            print(f"⚠️ Loss became NaN at step {step}, skipping this step.")
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()  # ✅ scaled backward
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
        scaler.step(optimizer)         # ✅ scaled step
        scaler.update()                # ✅ update scaler

        for name, param in unet.named_parameters():
            if torch.isnan(param).any():
                print(f"⚠️ NaN detected in {name}")
                return

        print(f"Step {step} - Loss: {loss.item()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32  # ← safer, for debugging

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    text_encoder = CLIPTextModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=weight_dtype
    ).to(device)
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="vae",
        torch_dtype=weight_dtype
    ).to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        torch_dtype=weight_dtype
    ).to(device)

    noise_scheduler = DDPMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["attn2.to_q", "attn2.to_k", "attn2.to_v"],
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet = unet.to(torch.float32)
    unet.print_trainable_parameters()

    dataset = GurukulDataset(
        metadata_path=METADATA_FILE,
        image_folder=IMAGE_FOLDER,
        tokenizer=tokenizer
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size = 1 for stability

    print("Starting training...")
    train(unet, text_encoder, vae, noise_scheduler, dataloader, device, weight_dtype)

    adapter_output_dir = "./lora_output"
    os.makedirs(adapter_output_dir, exist_ok=True)
    unet.save_pretrained(adapter_output_dir)
    print(f"✅ LoRA adapter saved to {adapter_output_dir}")

if __name__ == "__main__":
    main()
