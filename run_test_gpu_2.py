import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from test_attacks import test_attack

if __name__ == "__main__":

    device = torch.device("cuda")
    print(device)
    # Modello ResNet50 senza pesi pretrained
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # I modify the last layer for binary classification
    model.fc = nn.Sequential(
    nn.Dropout(DROPOUT),
    nn.Linear(model.fc.in_features, 2)
    )
    #model = model.to(device)
    model = model.cuda()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
       ])
    
        
    print("Initializing testing dataset....")
    test_dataset = FFDataset(root_dir=ROOT_DIR, split="test", transform=transform)
    # I get a subset of 1000 images (instead of 2800 total)
    test_small, _ = balanced_subset(test_dataset, n_per_class=5)

    #with open("sampled_test_set.json") as f:
    #    sampled_test_set_paths = json.loads(f.read())
    #test_small = get_imgs_by_filepath(test_dataset, sampled_test_set_paths)

    print("Initializing test loader...")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    test_loader_small = DataLoader(test_small, batch_size=64, shuffle=True)

    # LOADING MODELS

    #models_no_eps_sched= {
    #    "FGSM-AT (eps=2/255)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_3.pt",
    #    "FGSM-AT (eps=8/255)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_3_prova.pt",
    #    "FGSM-AT + entropy (eps=2/255)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_3.pt",
    #    "FGSM-AT + entropy (eps=8/255)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_3.pt"}

    #models_prova = {
    #    "FGSM-AT (eps=8/255) 1": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_3_prova.pt",
    #    "FGSM-AT (eps=8/255) 2": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_3.pt",
    #    "FGSM-AT (eps=8/255) 3": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_2.pt",
    #    "FGSM-AT (eps=2/255) 1": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_2.pt",
    #    "FGSM-AT (eps=2/255) 2": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_3.pt"}

    #models_eps_sched = {
    #    "FGSM-AT (cosine epsilon scheduler)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_3.pt",
    #    "FGSM-AT (linear epsilon scheduler)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_3.pt",
    #    "FGSM-AT + entropy (cosine epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_3.pt",
    #    "FGSM-AT + entropy (linear epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_3.pt",
    #}

    models_init_no_sched = {
        "FGSM-AT (eps=8/255)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.pt",
        "FGSM-AT (eps=2/255)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.pt",
        "FGSM-AT + entropy (eps=2/255)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.00784313725490196_seed_42_None_sched_4_init.pt",
        "FGSM-AT + entropy (eps=8/255)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_EPS_0.03137254901960784_seed_42_None_sched_4_init.pt"
    }

    models_init_eps_sched = {
        #"FGSM-AT (cosine epsilon scheduler)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.pt",
        #"FGSM-AT (linear epsilon scheduler)": "resnet50_fgsm_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.pt",
        #"FGSM-AT + entropy (cosine epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.pt",
        #"FGSM-AT + entropy (linear epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.pt",
        "FGSM-AT + entropy (cosine epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_sched_4_init.pt",
        #"FGSM-AT + entropy (linear epsilon scheduler)": "resnet50_square_epoch_11_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_sched_4_init.pt",
    }

    models_adaptive_fgsm = {
        "FGSM-AT (adaptive cosine scheduler)": "resnet50_fgsm_epoch_29_LR_0.0001_batchsize_32_WD_0.01_seed_42_cosine_adaptive.pt",
        "FGSM-AT (adaptive linear scheduler)": "resnet50_fgsm_epoch_29_LR_0.0001_batchsize_32_WD_0.01_seed_42_linear_adaptive.pt"
    }

    # test clean model:
    #model_name = "resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
    #model_data = "Clean Model"
    #checkpoint_path = f"{MODELS_DIR}/{model_name}"
    #checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"Starting testing {model_name}")
    ##clean_metrics, zoo_metrics = test_attack(model, test_loader_small, 'zoo', 2/255, 'art', model_name, model_data, "ZOO", device)
    #clean_metrics, autozoo_metrics = test_attack(model, test_loader_small, 'autozoom', 2/255, 'art', model_name, model_data, "AutoZOOM", device)


    for model_data, model_name in models_init_eps_sched.items():
        checkpoint_path = f"{MODELS_DIR}/with_eps_scheduler/{model_name}"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Starting testing {model_name}")

    #    clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader_small, 'fgsm', 2/255, 'foolbox', model_name, model_data, "FGSM (eps=2/255)", device)
    #    clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader_small, 'fgsm', 4/255, 'foolbox', model_name, model_data, "FGSM (eps=4/255)",device)
    #    clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader_small, 'fgsm', 8/255, 'foolbox', model_name, model_data, "FGSM (eps=8/255)",device)
    #    clean_metrics, ifgsm_metrics = test_attack(model, test_loader_small, 'ifgsm', 8/255, 'foolbox', model_name, model_data, "IFGSM",device)
    #    clean_metrics, pgd_metrics = test_attack(model, test_loader_small, 'pgd', 8/255, 'foolbox', model_name, model_data, "PGD",device)
        #clean_metrics, square_metrics = test_attack(model, test_loader_small, 'square', 16/255, 'art', model_name, model_data, "SQUARE (1000 iterations, eps=16/255)", device)
        #clean_metrics, jsma_metrics = test_attack(model, test_loader_small, 'jsma', 16/255, 'art', model_name, model_data, "JSMA", device)
        #clean_metrics, gen_metrics = test_attack(model, test_loader_small, 'genattack', 8/255, 'foolbox', model_name, model_data, "GenAttack", device)
        #clean_metrics, nes_metrics = test_attack(model, test_loader_small, 'nes', 8/255, 'None', model_name, model_data, "NES", device)
        #clean_metrics, autozoo_metrics = test_attack(model, test_loader_small, 'autozoom', 2/255, 'art', model_name, model_data, "AutoZOOM", device)
        clean_metrics, zoo_metrics = test_attack(model, test_loader_small, 'zoo', 2/255, 'art', model_name, model_data, "ZOO", device)

    
    #for model_data, model_name in models_init_no_sched.items():
    #    checkpoint_path = f"{MODELS_DIR}/no_eps_scheduler/{model_name}"
    #    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    #    model.load_state_dict(checkpoint['model_state_dict'])
    #    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #    print(f"Starting testing {model_name}")
#
    #    clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader_small, 'fgsm', 2/255, 'foolbox', model_name, model_data, "FGSM (eps=2/255)", device)
    #    clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader_small, 'fgsm', 4/255, 'foolbox', model_name, model_data, "FGSM (eps=4/255)",device)
    #    clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader_small, 'fgsm', 8/255, 'foolbox', model_name, model_data, "FGSM (eps=8/255)",device)
    #    clean_metrics, ifgsm_metrics = test_attack(model, test_loader_small, 'ifgsm', 8/255, 'foolbox', model_name, model_data, "IFGSM",device)
    #    clean_metrics, pgd_metrics = test_attack(model, test_loader_small, 'pgd', 8/255, 'foolbox', model_name, model_data, "PGD",device)
    #    clean_metrics, square_metrics = test_attack(model, test_loader_small, 'square', 16/255, 'art', model_name, model_data, "SQUARE (1000 iterations, eps=16/255)", device)
    #    clean_metrics, jsma_metrics = test_attack(model, test_loader_small, 'jsma', 16/255, 'art', model_name, model_data, "JSMA", device)
    #    clean_metrics, gen_metrics = test_attack(model, test_loader_small, 'genattack', 8/255, 'foolbox', model_name, model_data, "GenAttack", device)
    #    clean_metrics, nes_metrics = test_attack(model, test_loader_small, 'nes', 8/255, 'None', model_name, model_data, "NES", device)
    #    #clean_metrics, autozoo_metrics = test_attack(model, test_loader_small, 'autozoom', 2/255, 'None', model_name, model_data, "AutoZOOM", device)
    #    #clean_metrics, zoo_metrics = test_attack(model, test_loader_small, 'zoo', 2/255, 'art', model_name, model_data, "ZOO", device)


    