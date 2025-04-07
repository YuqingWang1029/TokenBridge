import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torchvision.datasets as datasets
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import torch_fidelity
import json
import argparse
from models.vae import AutoencoderKL
from scipy.stats import norm
import math
import multiprocessing as mp


class CenterCropTransform:
    """
    Center crop transform with resizing for consistent input dimensions.
    """
    def __init__(self, image_size):
        self.image_size = image_size
        
    def __call__(self, pil_image):
        while min(*pil_image.size) >= 2 * self.image_size:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.image_size / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image)
        crop_y = (arr.shape[0] - self.image_size) // 2
        crop_x = (arr.shape[1] - self.image_size) // 2
        return Image.fromarray(arr[crop_y: crop_y + self.image_size, crop_x: crop_x + self.image_size])


class ClasswiseImageFolder(datasets.ImageFolder):
    """
    Extension of ImageFolder that provides access to samples by class index.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.samples_by_class = {}
        for class_idx in range(len(self.classes)):
            class_indices = [idx for idx, (_, target) in enumerate(self.samples) if target == class_idx]
            self.samples_by_class[class_idx] = class_indices


def post_training_quantize(x, bits, min_range, max_range, std_range=3.0):
    """
    Apply post-training quantization to continuous features based on Gaussian distribution.
    
    Args:
        x (torch.Tensor): Input tensor to quantize
        bits (int): Number of quantization bits
        min_range (float): Minimum value of input data range
        max_range (float): Maximum value of input data range
        std_range (float): Standard normal distribution range (default 3.0, covering 99.7% of data)
    
    Returns:
        tuple: Quantized indices and their corresponding dequantized values
    """
    if bits < 0:
        return x, x
    
    n = 2 ** bits
    device = x.device
    dtype = x.dtype
    
    # Calculate quantization boundaries based on Gaussian CDF
    probs = torch.linspace(0, 1, n+1, device=device, dtype=dtype)
    boundaries = torch.tensor(norm.ppf(probs.cpu()), device=device, dtype=dtype)
    boundaries = torch.clamp(boundaries, -std_range, std_range)
    
    # Calculate reconstruction values using truncated normal mean
    def truncated_normal_mean(a, b):
        """
        Calculate conditional expectation E[X|a < X < b] for standard normal distribution.
        
        Args:
            a: Lower boundary
            b: Upper boundary
        
        Returns:
            float: Conditional expectation in interval (a,b)
        """
        sqrt_2 = math.sqrt(2)
        sqrt_2pi = math.sqrt(2 * math.pi)
        
        # Calculate PDF values
        phi_a = torch.exp(-0.5 * a**2) / sqrt_2pi
        phi_b = torch.exp(-0.5 * b**2) / sqrt_2pi
        
        # Calculate CDF values
        Phi_a = 0.5 * (1 + torch.erf(a / sqrt_2))
        Phi_b = 0.5 * (1 + torch.erf(b / sqrt_2))
        
        # Avoid division by zero
        denominator = Phi_b - Phi_a
        denominator = torch.where(denominator == 0, 
                                torch.tensor(1e-10, device=device, dtype=dtype), 
                                denominator)
        
        return (phi_a - phi_b) / denominator
    
    # Calculate reconstruction values for each interval
    reconstruction_values = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i+1]
        mean = truncated_normal_mean(a, b)
        reconstruction_values.append(mean)
    reconstruction_values = torch.tensor(reconstruction_values, device=device, dtype=dtype)
    
    # Map input to standard normal range
    x_normalized = (x - min_range) / (max_range - min_range) * (2 * std_range) - std_range
    x_clamped = x_normalized.clamp(-std_range, std_range)
    
    # Find nearest reconstruction value
    x_expanded = x_clamped.unsqueeze(-1)
    dists = (x_expanded - reconstruction_values).abs()
    indices = dists.argmin(dim=-1)
    
    # Map back to original range
    normalized_values = reconstruction_values
    values = (normalized_values + std_range) / (2 * std_range) * (max_range - min_range) + min_range
    dequant = values[indices]
    
    return indices, dequant


def save_image(tensor, path):
    """
    Save tensor as image file.
    
    Args:
        tensor (torch.Tensor): Image tensor in range [-1, 1]
        path (str): Path to save image
    """
    image = (tensor + 1) / 2.0
    image = torch.clamp(image, 0, 1)
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def calculate_metrics(generated_path, img_size=256):
    """
    Calculate FID and Inception Score metrics.
    
    Args:
        generated_path (str): Path to generated images
        img_size (int): Image size
    
    Returns:
        dict: Dictionary with calculated metrics
    """
    if img_size == 256:
        input2 = None
        fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
    else:
        raise NotImplementedError(f"Image size {img_size} not supported")
    
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=generated_path,
        input2=input2,
        fid_statistics_file=fid_statistics_file,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    return metrics_dict


def process_classes(rank, args, start_class, end_class):
    """
    Process a subset of classes for reconstruction evaluation.
    
    Args:
        rank (int): GPU rank
        args (Namespace): Command line arguments
        start_class (int): Starting class index
        end_class (int): Ending class index
    
    Returns:
        tuple: Path to reconstructed images and quantization statistics
    """
    device = f"cuda:{rank}"
    
    # Initialize VAE
    vae = AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=args.vae_path
    ).to(device)
    vae.eval()

    # Setup data loading
    transform = transforms.Compose([
        CenterCropTransform(256),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    dataset = ClasswiseImageFolder(args.image_dir, transform=transform)
    
    exp_name = f"quant_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    real_dir = os.path.join(exp_dir, 'real')
    recon_dir = os.path.join(exp_dir, 'recon')
    
    # Create directories (in a multi-process safe way)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    min_range, max_range = -args.range, args.range
    stats = {
        'out_of_range_count': 0,
        'total_values': 0,
        'unique_values': set()
    }

    with torch.no_grad():
        for class_idx in tqdm(range(start_class, end_class), 
                            desc=f'GPU {rank} processing classes {start_class}-{end_class-1}'):
            class_indices = dataset.samples_by_class.get(class_idx, [])[:args.images_per_class]
            if not class_indices:
                continue
                
            class_subset = Subset(dataset, class_indices)
            class_loader = DataLoader(
                class_subset,
                batch_size=args.images_per_class,
                num_workers=1,
                pin_memory=True
            )
            
            for images, _ in class_loader:
                images = images.to(device)
                
                posterior = vae.encode(images)
                latents = posterior.sample()
                latents_norm = latents.mul(0.2325)
                
                quant, dequant = post_training_quantize(latents_norm, args.bits, min_range, max_range)

                recon = vae.decode(dequant.div(0.2325))
                
                stats['out_of_range_count'] += ((latents_norm < min_range) | (latents_norm > max_range)).sum().item()
                stats['total_values'] += latents_norm.numel()
                stats['unique_values'].update(quant.cpu().unique().tolist())
                
                for idx, (img, rec) in enumerate(zip(images, recon)):
                    # Use GPU rank as part of filename to avoid conflicts
                    save_image(img, os.path.join(real_dir, f'class_{class_idx:04d}_gpu{rank}_img_{idx:02d}.png'))
                    save_image(rec, os.path.join(recon_dir, f'class_{class_idx:04d}_gpu{rank}_img_{idx:02d}.png'))
    
    return recon_dir, stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate post-training quantization for VAE latents')
    
    parser.add_argument('--vae_path', type=str, 
                        default="pretrained_models/vae/kl16.ckpt",
                        help='Path to VAE checkpoint')
    parser.add_argument('--image_dir', type=str, 
                        default="./data/imagenet/train",
                        help='Path to ImageNet training set')
    parser.add_argument('--output_dir', type=str, 
                        default="quantization_results",
                        help='Base directory for output')
    parser.add_argument('--bits', type=int, default=6,
                        help='Number of bits for quantization')
    parser.add_argument('--range', type=float, default=5.0,
                        help='Absolute range for quantization')
    parser.add_argument('--images_per_class', type=int, default=50,
                        help='Number of images per class')
    
    args = parser.parse_args()
    
    # Get available GPU count
    world_size = torch.cuda.device_count()
    
    # Calculate class range per GPU
    classes_per_gpu = 1000 // world_size
    processes = []
    
    for rank in range(world_size):
        start_class = rank * classes_per_gpu
        end_class = start_class + classes_per_gpu if rank != world_size - 1 else 1000
        p = mp.Process(target=process_classes, args=(rank, args, start_class, end_class))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Calculate final metrics
    exp_name = f"quant_b{args.bits}_r{abs(args.range)}_n{args.images_per_class}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    recon_dir = os.path.join(exp_dir, 'recon')
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(recon_dir, img_size=256)
    
    results = {
        'parameters': vars(args),
        'fid_score': metrics['frechet_inception_distance'],
        'inception_score': {
            'mean': metrics['inception_score_mean'],
            'std': metrics['inception_score_std']
        }
    }
    
    # Save results
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults for experiment {exp_name}:")
    print(f"FID Score: {results['fid_score']:.4f}")
    print(f"Inception Score: {results['inception_score']['mean']:.4f} ± {results['inception_score']['std']:.4f}")


if __name__ == "__main__":
    main()