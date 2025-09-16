import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch

class Pix2PixUnpairedDataset(Dataset):
    """
    Charge des images depuis deux structures possibles :
    1. Structure concaténée : images A+B horizontalement dans un seul dossier
    2. Structure séparée : dossiers distincts pour A et B (trainA, trainB, etc.)
    """
    def __init__(self, root_dir, split='train', size=256):
        self.root_dir = root_dir
        self.split = split
        self.size = size
        
        # Vérifier que le dossier racine existe
        if not os.path.exists(self.root_dir):
            raise ValueError(f"❌ Le dossier {self.root_dir} n'existe pas")
        
        # Détecter la structure du dataset
        self.structure = self._detect_structure()
        
        if self.structure == 'separate':
            # Structure avec dossiers séparés (trainA, trainB)
            self.dir_A = os.path.join(root_dir, split + 'A')
            self.dir_B = os.path.join(root_dir, split + 'B')
            
            # Vérifier que les dossiers existent
            if not os.path.exists(self.dir_A):
                raise ValueError(f"❌ Dossier {self.dir_A} non trouvé")
            if not os.path.exists(self.dir_B):
                raise ValueError(f"❌ Dossier {self.dir_B} non trouvé")
                
            # Lister les fichiers
            self.files_A = sorted([f for f in os.listdir(self.dir_A) 
                                 if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
            self.files_B = sorted([f for f in os.listdir(self.dir_B) 
                                 if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
            
            print(f"📁 Structure séparée détectée:")
            print(f"   A: {self.dir_A} ({len(self.files_A)} images)")
            print(f"   B: {self.dir_B} ({len(self.files_B)} images)")
            
        else:
            # Structure concaténée (ancien format)
            self.data_dir = os.path.join(root_dir, split)
            
            if not os.path.exists(self.data_dir):
                raise ValueError(f"❌ Le dossier {self.data_dir} n'existe pas")

            self.files = sorted([f for f in os.listdir(self.data_dir)
                               if f.lower().endswith(('.jpg','.png','.jpeg','.bmp'))])
            
            print(f"📁 Structure concaténée détectée:")
            print(f"   Dossier: {self.data_dir}")
            print(f"   Images concaténées: {len(self.files)}")

        # Vérifier qu'il y a des images
        if (self.structure == 'separate' and (len(self.files_A) == 0 or len(self.files_B) == 0)) or \
           (self.structure == 'concat' and len(self.files) == 0):
            raise ValueError("❌ Aucune image trouvée")

        # Transformations
        self.is_training = split == 'train'
        self._setup_transforms()
        random.seed(42)

    def _detect_structure(self):
        """Détecte automatiquement la structure du dataset"""
        # Vérifier si les dossiers séparés existent
        if os.path.exists(os.path.join(self.root_dir, 'trainA')) and \
           os.path.exists(os.path.join(self.root_dir, 'trainB')):
            return 'separate'
        # Vérifier si les dossiers test séparés existent
        elif os.path.exists(os.path.join(self.root_dir, 'testA')) and \
             os.path.exists(os.path.join(self.root_dir, 'testB')):
            return 'separate'
        # Vérifier si le dossier concaténé train existe
        elif os.path.exists(os.path.join(self.root_dir, 'train')):
            return 'concat'
        # Vérifier si le dossier concaténé test existe
        elif os.path.exists(os.path.join(self.root_dir, 'test')):
            return 'concat'
        else:
            # Afficher les dossiers disponibles pour le débogage
            available_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            print(f"📂 Dossiers disponibles dans {self.root_dir}: {available_dirs}")
            
            # Essayer de deviner la structure basée sur les noms de dossiers
            a_dirs = [d for d in available_dirs if d.endswith('A')]
            b_dirs = [d for d in available_dirs if d.endswith('B')]
            
            if a_dirs and b_dirs:
                print(f"🤔 Structure détectée par pattern (A/B): {a_dirs}, {b_dirs}")
                return 'separate'
            else:
                raise ValueError(f"❌ Structure de dataset non reconnue dans {self.root_dir}")

    def _setup_transforms(self):
        """Configure les transformations en fonction de la structure"""
        if self.structure == 'concat':
            # Transformations pour images concaténées
            self.augmentation = T.Compose([
                T.Resize((self.size + 30, self.size * 2 + 60)),
                T.RandomCrop((self.size, self.size * 2)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.transform_basic = T.Compose([
                T.Resize((self.size, self.size * 2)),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            # Transformations pour images séparées
            self.augmentation = T.Compose([
                T.Resize(self.size + 30),
                T.RandomCrop(self.size),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.transform_basic = T.Compose([
                T.Resize(self.size),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        if self.structure == 'separate':
            return max(len(self.files_A), len(self.files_B))
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if self.structure == 'separate':
            # Pour la structure séparée, on charge des images aléatoires de A et B
            idx_A = idx % len(self.files_A)
            idx_B = random.randint(0, len(self.files_B) - 1)
            
            fname_A = self.files_A[idx_A]
            fname_B = self.files_B[idx_B]
            
            path_A = os.path.join(self.dir_A, fname_A)
            path_B = os.path.join(self.dir_B, fname_B)
            
            try:
                img_A = Image.open(path_A).convert("RGB")
                img_B = Image.open(path_B).convert("RGB")
                
                if self.is_training:
                    img_A = self.augmentation(img_A)
                    img_B = self.augmentation(img_B)
                else:
                    img_A = self.transform_basic(img_A)
                    img_B = self.transform_basic(img_B)
                    
                return {'A': img_A, 'B': img_B, 'name': f"{fname_A}_{fname_B}"}
                
            except Exception as e:
                print(f"❌ Erreur lors du chargement de {path_A} ou {path_B}: {e}")
                empty_tensor = torch.zeros((3, self.size, self.size))
                return {'A': empty_tensor, 'B': empty_tensor, 'name': 'error'}
                
        else:
            # Ancienne méthode pour structure concaténée
            fname = self.files[idx]
            path = os.path.join(self.data_dir, fname)

            try:
                full_image = Image.open(path).convert("RGB")

                if self.is_training:
                    full_tensor = self.augmentation(full_image)
                else:
                    full_tensor = self.transform_basic(full_image)

                # Séparer en gauche/droite
                _, h, w = full_tensor.shape
                half_w = w // 2

                A = full_tensor[:, :, :half_w]   # Partie gauche
                B = full_tensor[:, :, half_w:]   # Partie droite

                # Vérification dimensions finales
                if A.shape[1] != self.size or A.shape[2] != self.size:
                    A = T.functional.resize(A, (self.size, self.size))
                    B = T.functional.resize(B, (self.size, self.size))

                return {'A': A, 'B': B, 'name': fname}

            except Exception as e:
                print(f"❌ Erreur lors du chargement de {path}: {e}")
                empty_tensor = torch.zeros((3, self.size, self.size))
                return {'A': empty_tensor, 'B': empty_tensor, 'name': 'error'}

    def show_sample(self, idx=0):
        """Affiche les infos d'un sample"""
        sample = self[idx]
        print(f"📋 Sample {idx}: {sample['name']}")
        print(f"📐 Shape A: {sample['A'].shape}, Shape B: {sample['B'].shape}")
        print(f"📊 A range: [{sample['A'].min():.3f}, {sample['A'].max():.3f}]")
        print(f"📊 B range: [{sample['B'].min():.3f}, {sample['B'].max():.3f}]")
        return sample

    def visualize_sample(self, idx=0):
        """Retourne A et B comme images PIL"""
        sample = self[idx]

        def tensor_to_image(tensor):
            tensor = tensor.clone().detach()
            tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
            return T.ToPILImage()(tensor.cpu())

        return tensor_to_image(sample['A']), tensor_to_image(sample['B']), sample['name']


# Test rapide
if __name__ == '__main__':
    try:
        dataset = Pix2PixUnpairedDataset(
            root_dir='../datasets/facescrub',
            split='train',
            size=128
        )
        print(f"✅ Dataset chargé avec {len(dataset)} images")
        dataset.show_sample(0)
    except Exception as e:
        print(f"❌ Erreur: {e}")