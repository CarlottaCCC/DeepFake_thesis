from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from test_generic import test_attack

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
    transform_small = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
       ])
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    test_dataset_sub = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform_small)
    # I get a smaller subset of 500 images
    test_small = balanced_subset(test_dataset, n_per_class=5)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_loader_small = DataLoader(test_small, batch_size=32, shuffle=True)
    test_loader_zoo = DataLoader(test_dataset_sub, batch_size=2, shuffle=True)

    # LOADING MODELS
    '''
    RIFARE TRAINING QUESTI
    'resnet50_fgsm_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.01568627450980392_fine_tuned',
    'resnet50_square_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.01568627450980392_fine_tuned'
    '''

    square_models = [
        'resnet50_square_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned',
        'resnet50_square_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_fine_tuned'
    ]

    fgsm_models = [
        'resnet50_fgsm_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_fine_tuned',
        'resnet50_fgsm_epoch_5_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_fine_tuned'
    ]

    clean_model = 'resnet50_clean_epoch_2_LR_0.0001_batchsize_32_WD_0.01_DROPOUT_0.0_hor_flip'

    checkpoint_path = f"models_10/clean_resnet50/{clean_model}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Starting testing {clean_model}")

    #clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader, 'fgsm', 2/255, 'foolbox', clean_model, device)
    #clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader, 'fgsm', 4/255, 'foolbox', clean_model, device)
    #clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader, 'fgsm', 8/255, 'foolbox', clean_model, device)
    #clean_metrics, ifgsm_metrics = test_attack(model, test_loader, 'ifgsm', 8/255, 'foolbox', clean_model, device)
    #clean_metrics, pgd_metrics = test_attack(model, test_loader, 'pgd', 8/255, 'foolbox', clean_model, device)
    #clean_metrics, tjsma_metrics = test_attack(model, test_loader, 'tjsma', 2/255, 'None', clean_model, device)
    #clean_metrics, wjsma_metrics = test_attack(model, test_loader, 'wjsma', 2/255, 'None', clean_model, device)
    clean_metrics, square_metrics = test_attack(model, test_loader, 'square', 16/255, 'art', clean_model, device)
    clean_metrics, zoo_metrics = test_attack(model, test_loader, 'zoo', 2/255, 'art', clean_model, device)
    clean_metrics, autozoo_metrics = test_attack(model, test_loader, 'autozoom', 2/255, 'None', clean_model, device)
    clean_metrics, gen_metrics = test_attack(model, test_loader_small, 'genattack', 8/255, 'foolbox', clean_model, device)
    clean_metrics, nes_metrics = test_attack(model, test_loader_small, 'nes', 8/255, 'None', clean_model, device)


    for model_name in fgsm_models:
        checkpoint_path = f"models_10/fgsm_resnet50/{model_name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Starting testing {model_name}")
    
        #clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader, 'fgsm', 2/255, 'foolbox', model_name, device)
        #clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader, 'fgsm', 4/255, 'foolbox', model_name, device)
        #clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader, 'fgsm', 8/255, 'foolbox', model_name, device)
        #clean_metrics, ifgsm_metrics = test_attack(model, test_loader, 'ifgsm', 8/255, 'foolbox', model_name, device)
        #clean_metrics, pgd_metrics = test_attack(model, test_loader, 'pgd', 8/255, 'foolbox', model_name, device)
        clean_metrics, tjsma_metrics = test_attack(model, test_loader, 'tjsma', 2/255, 'None', model_name, device)
        clean_metrics, wjsma_metrics = test_attack(model, test_loader, 'wjsma', 2/255, 'None', model_name, device)
        clean_metrics, square_metrics = test_attack(model, test_loader, 'square', 16/255, 'art', model_name, device)
        clean_metrics, zoo_metrics = test_attack(model, test_loader, 'zoo', 2/255, 'art', model_name, device)
        clean_metrics, autozoo_metrics = test_attack(model, test_loader, 'autozoom', 2/255, 'None', model_name, device)
        clean_metrics, gen_metrics = test_attack(model, test_loader_small, 'genattack', 8/255, 'foolbox', model_name, device)
        clean_metrics, nes_metrics = test_attack(model, test_loader_small, 'nes', 8/255, 'None', model_name, device)

    for model_name in square_models:
        checkpoint_path = f"models_10/square_resnet50/{model_name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Starting testing {model_name}")
    
        #clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader, 'fgsm', 2/255, 'foolbox', model_name, device)
        #clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader, 'fgsm', 4/255, 'foolbox', model_name, device)
        #clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader, 'fgsm', 8/255, 'foolbox', model_name, device)
        #clean_metrics, ifgsm_metrics = test_attack(model, test_loader, 'ifgsm', 8/255, 'foolbox', model_name, device)
        #clean_metrics, pgd_metrics = test_attack(model, test_loader, 'pgd', 8/255, 'foolbox', model_name, device)
        clean_metrics, tjsma_metrics = test_attack(model, test_loader, 'tjsma', 2/255, 'None', model_name, device)
        clean_metrics, wjsma_metrics = test_attack(model, test_loader, 'wjsma', 2/255, 'None', model_name, device)
        clean_metrics, square_metrics = test_attack(model, test_loader, 'square', 16/255, 'art', model_name, device)
        clean_metrics, zoo_metrics = test_attack(model, test_loader, 'zoo', 2/255, 'art', model_name, device)
        clean_metrics, autozoo_metrics = test_attack(model, test_loader, 'autozoom', 2/255, 'None', model_name, device)
        clean_metrics, gen_metrics = test_attack(model, test_loader_small, 'genattack', 8/255, 'foolbox', model_name, device)
        clean_metrics, nes_metrics = test_attack(model, test_loader_small, 'nes', 8/255, 'None', model_name, device)

    