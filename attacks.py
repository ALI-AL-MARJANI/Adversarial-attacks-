import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Denormalizsation 
def denorm(batch, mean=[0.1307], std=[0.3081]):
    
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


# FGSM Attack
def fgsm_attack(image, epsilon, data_grad):
    
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image, epsilon * sign_data_grad


# PGD Attack 
def pgd_attack(model, images, labels, eps=0.3, alpha=2/255, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for _ in range(iters):
        images = images.detach()
        images.requires_grad = True

        outputs = model(images)
        model.zero_grad()

        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()

        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, 0, 1).detach()

    return images, adv_images - ori_images


# DeepFool Attack 
def deepfool(model, image, num_classes=10, max_iter=50 , overshoot=0.02):
    """
    We Compute the minimal L2 perturbation required to change classifier decision
    """
    model.eval()
    x = image.clone().detach().to(device)
    x.requires_grad_(True)

    with torch.no_grad():
        logits = model(x)
    label = logits.argmax().item()

    total_perturb = torch.zeros_like(x).to(device)

    for _ in range(max_iter):
        logits = model(x)
        pred = logits.argmax().item()

        if pred != label:
            break

        logits[0, label].backward(retain_graph=True)
        grad_orig = x.grad.data.clone()

        min_pert = float("inf")
        w_best = None

        for k in range(num_classes):
            if k == label:
                continue

            x.grad.zero_()
            logits[0, k].backward(retain_graph=True)
            grad_k = x.grad.data.clone()

            w_k = grad_k - grad_orig
            f_k = (logits[0, k] - logits[0, label]).item()

            pert_k = abs(f_k) / (w_k.flatten().norm() + 1e-8)

            if pert_k < min_pert:
                min_pert = pert_k
                w_best = w_k

        r_i = (min_pert + 1e-4) * w_best / (w_best.flatten().norm() + 1e-8)
        total_perturb += r_i

        x = (image + (1 + overshoot) * total_perturb).detach()
        x.requires_grad_(True)

    x_adv = torch.clamp(x, 0, 1)
    return x_adv.detach(), total_perturb.detach()
