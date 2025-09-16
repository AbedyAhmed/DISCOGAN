import os
import argparse
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import time

# Ajouter le chemin au PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from discogan.model import Generator, Discriminator, weights_init
from discogan.dataset import Pix2PixUnpairedDataset
from discogan.utils import save_sample

# ---- Fonction de sauvegarde ----
def save_ckpt(state, ckpt_dir, name="latest.pt"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, name)
    torch.save(state, path)
    print(f"[INFO] Checkpoint saved at {path}")

# ---- Fonction de monitoring ----
def print_training_info(epoch, step, loss_G, loss_D_A, loss_D_B, loss_GAN_AB, loss_GAN_BA, loss_cycle, loss_id, lr):
    print(f"Epoch {epoch:03d} | Step {step:06d}")
    print(f"G Total: {loss_G:.4f} | D_A: {loss_D_A:.4f} | D_B: {loss_D_B:.4f}")
    print(f"GAN AB: {loss_GAN_AB:.4f} | GAN BA: {loss_GAN_BA:.4f}")
    print(f"Cycle: {loss_cycle:.4f} | Identity: {loss_id:.4f}")
    print(f"Learning Rate: {lr:.6f}")
    print("-" * 60)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_name', type=str, required=True, choices=['edges2handbags','edges2shoes','facescrub'])
    p.add_argument('--data_root', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'datasets'))
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--sample_every', type=int, default=500)
    p.add_argument('--save_every', type=int, default=50)  # Sauvegarde tous les 10 epochs
    p.add_argument('--lambda_cyc', type=float, default=10.0)
    p.add_argument('--lambda_id', type=float, default=5.0)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()

def main():
    args = get_args()
    
    # Afficher les paramètres
    print("⚙️  Paramètres d'entraînement:")
    print(f"   Task: {args.task_name}")
    print(f"   Device: {args.device}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Image Size: {args.size}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Lambda Cycle: {args.lambda_cyc}")
    print(f"   Lambda Identity: {args.lambda_id}")
    
    task_dir = os.path.join(args.data_root, args.task_name)
    
    # Chargement du dataset en mode VERTICAL
    print("📦 Chargement du dataset...")
    train_ds = Pix2PixUnpairedDataset(task_dir, 'train', size=args.size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             num_workers=args.num_workers, drop_last=True, pin_memory=True)
    
    print(f"✅ Dataset chargé: {len(train_ds)} images, {len(train_loader)} batches par epoch")

    # Models
    print("🧠 Initialisation des modèles...")
    G_AB = Generator().to(args.device)
    G_BA = Generator().to(args.device)
    D_A  = Discriminator().to(args.device)
    D_B  = Discriminator().to(args.device)
    
    # Initialisation des poids
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)
    
    print(f"✅ Modèles initialisés sur {args.device}")

    # Losses
    adv_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    id_loss = nn.L1Loss()

    # Optims
    opt_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), 
                      lr=args.lr, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Learning Rate Schedulers
    scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=50, gamma=0.5)
    scheduler_D_A = optim.lr_scheduler.StepLR(opt_D_A, step_size=50, gamma=0.5)
    scheduler_D_B = optim.lr_scheduler.StepLR(opt_D_B, step_size=50, gamma=0.5)

    # Targets
    def real_labels(shape): 
        return torch.ones(shape, device=args.device)
    def fake_labels(shape): 
        return torch.zeros(shape, device=args.device)

    # Dossiers de résultats
    step = 0
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', args.task_name, 'samples')
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', args.task_name)
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Pour sauvegarde du meilleur modèle
    best_loss = float('inf')
    start_time = time.time()
    
    print("🚀 Début de l'entraînement...")
    print("-" * 80)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            A = batch['A'].to(args.device)
            B = batch['B'].to(args.device)

            # ------------------
            #  Train Generators
            # ------------------
            opt_G.zero_grad()

            # GAN losses
            fake_B = G_AB(A)
            pred_fake_B = D_B(fake_B)
            loss_GAN_AB = adv_loss(pred_fake_B, real_labels(pred_fake_B.shape))

            fake_A = G_BA(B)
            pred_fake_A = D_A(fake_A)
            loss_GAN_BA = adv_loss(pred_fake_A, real_labels(pred_fake_A.shape))

            # Cycle consistency losses
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            loss_cycle = cycle_loss(rec_A, A) + cycle_loss(rec_B, B)

            # Identity losses
            idt_A = G_BA(A)
            idt_B = G_AB(B)
            loss_id = id_loss(idt_A, A) + id_loss(idt_B, B)

            # Total generator loss
            loss_G = loss_GAN_AB + loss_GAN_BA + args.lambda_cyc * loss_cycle + args.lambda_id * loss_id
            loss_G.backward()
            opt_G.step()

            # ------------------
            #  Train D_A
            # ------------------
            opt_D_A.zero_grad()
            pred_real_A = D_A(A)
            loss_D_A_real = adv_loss(pred_real_A, real_labels(pred_real_A.shape))
            pred_fake_A = D_A(fake_A.detach())
            loss_D_A_fake = adv_loss(pred_fake_A, fake_labels(pred_fake_A.shape))
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # ------------------
            #  Train D_B
            # ------------------
            opt_D_B.zero_grad()
            pred_real_B = D_B(B)
            loss_D_B_real = adv_loss(pred_real_B, real_labels(pred_real_B.shape))
            pred_fake_B = D_B(fake_B.detach())
            loss_D_B_fake = adv_loss(pred_fake_B, fake_labels(pred_fake_B.shape))
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            # Sauvegarde des samples
            if step % args.sample_every == 0:
                with torch.no_grad():
                    save_sample(A[:4], B[:4], fake_B[:4], fake_A[:4], results_dir, step)
                    if step > 0:  # Ne pas afficher au step 0
                        print(f"📸 Sample sauvegardé à step {step}")

            # Monitoring
            if step % 100 == 0 and step > 0:
                current_lr = scheduler_G.get_last_lr()[0]
                print_training_info(epoch, step, loss_G.item(), loss_D_A.item(), 
                                  loss_D_B.item(), loss_GAN_AB.item(), 
                                  loss_GAN_BA.item(), loss_cycle.item(), 
                                  loss_id.item(), current_lr)

            step += 1
            epoch_loss_G += loss_G.item()
            epoch_loss_D += (loss_D_A.item() + loss_D_B.item()) / 2

        # Mise à jour des learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Calcul des moyennes pour l'epoch
        avg_loss_G = epoch_loss_G / len(train_loader)
        avg_loss_D = epoch_loss_D / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Sauvegarde du checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_ckpt({
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D_A': opt_D_A.state_dict(),
                'opt_D_B': opt_D_B.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss_G': avg_loss_G,
                'loss_D': avg_loss_D,
            }, ckpt_dir, name=f'epoch_{epoch:03d}.pt')

        # Sauvegarde du meilleur modèle
        if avg_loss_G < best_loss:
            best_loss = avg_loss_G
            save_ckpt({
                'G_AB': G_AB.state_dict(),
                'G_BA': G_BA.state_dict(),
                'D_A': D_A.state_dict(),
                'D_B': D_B.state_dict(),
                'epoch': epoch,
                'step': step,
                'loss': best_loss,
            }, ckpt_dir, name='best_model.pt')
            print(f"🎯 Nouveau meilleur modèle! Loss: {best_loss:.4f}")

        # Affichage du résumé de l'epoch
        print(f"⏰ Epoch {epoch:03d}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"G: {avg_loss_G:.4f} | D: {avg_loss_D:.4f} | "
              f"LR: {scheduler_G.get_last_lr()[0]:.2e}")

    # Fin de l'entraînement
    total_time = time.time() - start_time
    print(f"✅ Entraînement terminé en {total_time:.1f} secondes!")
    print(f"📊 Meilleur loss: {best_loss:.4f}")
    print(f"💾 Modèles sauvegardés dans: {ckpt_dir}")

# Fonction pour générer des résultats de test
def generate_test_results(args):
    """Génère des résultats de test après l'entraînement"""
    from discogan.utils import save_test_batch
    
    print("🧪 Génération des résultats de test...")
    
    task_dir = os.path.join(args.data_root, args.task_name)
    test_ds = Pix2PixUnpairedDataset(task_dir, 'test', size=args.size)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', args.task_name, 'test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger le meilleur modèle
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', args.task_name)
    best_model_path = os.path.join(ckpt_dir, 'best_model.pt')
    
    if os.path.exists(best_model_path):
        print(f"📦 Chargement du meilleur modèle: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=args.device)
        
        G_AB = Generator().to(args.device)
        G_BA = Generator().to(args.device)
        
        G_AB.load_state_dict(checkpoint['G_AB'])
        G_BA.load_state_dict(checkpoint['G_BA'])
        
        G_AB.eval()
        G_BA.eval()
        
        print("🎨 Génération des résultats de test...")
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                A = batch['A'].to(args.device)
                B = batch['B'].to(args.device)
                
                fake_B = G_AB(A)
                fake_A = G_BA(B)
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)
                
                # Sauvegarder les résultats
                save_test_batch(A, B, fake_B, fake_A, rec_A, rec_B, output_dir, i)
                
                if i % 10 == 0:
                    print(f"   Batch {i} traité")
        
        print(f"✅ Résultats de test sauvegardés dans: {output_dir}")
    else:
        print("❌ Aucun modèle trouvé pour les tests")

if __name__ == '__main__':
    args = get_args()
    
    # Vérification du device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA non disponible, passage sur CPU")
        args.device = 'cpu'
    
    print(f"🔧 Device utilisé: {args.device}")
    if args.device == 'cuda':
        print(f"💻 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🧠 Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Entraînement principal
    main()
    
    # Génération des résultats de test
    try:
        generate_test_results(args)
    except Exception as e:
        print(f"❌ Erreur lors de la génération des tests: {e}")
        print("⚠️  Continuez sans les tests ou vérifiez votre dataset de test")
    
    print("🎉 Processus terminé avec succès!")