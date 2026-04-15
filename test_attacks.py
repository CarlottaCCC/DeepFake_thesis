from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FFDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import foolbox as fb
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
import csv
from attacks import *
from attacks_implementation.nes import nes_attack
from attacks_implementation.jsma_accurate import tjsma_binary, wjsma_binary, wjsma_binary_debug
from attacks_implementation.autozoom import AutoZOOMAttack
from attacks_implementation.ifgsm import ifgsm_attack


def test_attack(model, test_loader, attack_type, epsilon, library, model_name, model_data, attack_label_data, device):

    clean_metrics = Metrics()
    adv_metrics = Metrics()
    total_samples = 0

    model.eval()
    criterion = nn.CrossEntropyLoss()

    preprocessing = dict(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    axis=-3  # per PyTorch (C, H, W)
    )
    fmodel = fb.PyTorchModel(model, bounds=(0,1), preprocessing=preprocessing, device=device)

    classifier = PyTorchClassifier(
    model=model,
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.Adam(model.parameters()),
    input_shape=(3, 224, 224),
    nb_classes=2,
    clip_values=(0.0, 1.0),
    preprocessing=(
        [0.485, 0.456, 0.406],  # mean ImageNet
        [0.229, 0.224, 0.225]   # std ImageNet
    ),
    device_type="gpu"
    )

    #black box model wrapper for AutoZOOm attack
    model_fn = make_model_fn(model, device='cuda')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    print(f"Starting {attack_type} test!")
    pbar = tqdm(test_loader, desc=f"Testing", unit="batch")
    for batch in pbar:
        if batch is None:
            continue
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device).long().squeeze()
        batch_size = imgs.size(0)
        #print(y.shape, y.dtype)

        # ATTACKS
        #if attack_type == 'fgsm' and library == 'None':
        #    grad_imgs = torch.autograd.grad(loss_clean, imgs, retain_graph=True, create_graph=False)[0]
        #    imgs_adv = imgs +  epsilon * grad_imgs.sign()
        #    #imgs_adv = torch.clamp(imgs_adv, 0, 1)
        #    imgs_adv = torch.clamp(imgs_adv, imgs - epsilon, imgs + epsilon)

        if library == 'foolbox' and attack_type != 'genattack':
            attack_fn = get_attack_foolbox(attack_type)
            eps = epsilon
            _, imgs_adv, _ = attack_fn(fmodel, imgs, labels, epsilons=eps)
        elif library == 'art' and (attack_type!= 'jsma'):
            attack_fn = get_attack_art(attack_type, classifier)
            print(classifier._device)
            #target_labels = 1 - labels
            imgs_adv = attack_fn.generate(x=imgs.cpu().numpy(), y=labels.cpu().numpy())
            # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
            imgs_adv = torch.from_numpy(imgs_adv).float().to(device)
        elif library == 'art' and (attack_type == 'jsma'):
            attack_fn = get_attack_art(attack_type, classifier)
            target_labels = 1 - labels
            imgs_adv = attack_fn.generate(x=imgs.cpu().numpy(), y=target_labels.cpu().numpy())
            # ART returns numpy, need to convert imgs_square to tensor to pass it to the model
            imgs_adv = torch.from_numpy(imgs_adv).float().to(device)
        elif library == 'foolbox' and attack_type == 'genattack':
            # CHIAVE: Crea target labels (classe opposta per classificazione binaria)
            target_labels = 1 - labels  # Se binario: 0→1, 1→0
            # Crea criterion TARGETED 
            criterion = fb.criteria.TargetedMisclassification(target_labels)
            attack_fn = get_attack_foolbox(attack_type)
            eps = epsilon
            _, imgs_adv, _ = attack_fn(fmodel, imgs, criterion, epsilons=eps)
        elif attack_type == 'tjsma':
            adv_list = []
            target_labels = 1 - labels
            imgs_adv = torch.zeros_like(imgs)
            for i in range(imgs.size(0)):
                x_adv = tjsma_binary(model, imgs[i], target_labels[i], max_iter=300, theta=0.05)
                imgs_adv[i] = x_adv

            # since tjsma binary returns image.detach().squeeze(0) the output has dimention [3,244,244]
            # so with torch.stack I add one dimension creating a batch [32,3,224,224]
            #imgs_adv = torch.stack(adv_list, dim=0) 
            #adv_list = []

        elif attack_type == 'wjsma':
            target_labels = 1 - labels
            imgs_adv = torch.zeros_like(imgs)
            for i in range(imgs.size(0)):
                x_adv = wjsma_binary(model, imgs[i], target_labels[i], max_iter=300, theta=0.05)
                imgs_adv[i] = x_adv

            #imgs_adv = torch.stack(adv_list, dim=0)
            #adv_list = []
            #imgs_adv[i] = x_adv

        elif attack_type == 'nes':
            norm_imgs = normalize(imgs)
            target_labels = 1 - labels
            imgs_adv = nes_attack(model, norm_imgs, target_labels)

        #elif attack_type == 'ifgsm':
        #    imgs_adv = ifgsm_attack()

        #elif attack_type == 'autozoom':
        #    norm_imgs = normalize(imgs)
        #    attack = AutoZOOMAttack(model_fn=model_fn, attack_mode='bilin', img_shape=(3,224,224))
        #    imgs_adv = attack.generate(norm_imgs.cpu().numpy(), labels.cpu().numpy())
        #    imgs_adv = torch.from_numpy(imgs_adv).float().to(device)

        # compute L2 and Linf metrics
        delta = imgs_adv - imgs
        l2, linf = batch_norms(delta)
        adv_metrics.total_l2 += l2.sum().item()
        adv_metrics.total_linf += linf.sum().item()

        # normalize
        imgs = normalize(imgs)
        imgs_adv = normalize(imgs_adv)
        #print(f"min: {imgs.min():.3f}, max: {imgs.max():.3f}")
        #print(f"min: {imgs_adv.min():.3f}, max: {imgs_adv.max():.3f}")

        # inferenza
        with torch.no_grad():
            logits_clean = model(imgs)
            logits_adv = model(imgs_adv)

            probs_clean = torch.softmax(logits_clean, dim=1)[:,1].detach().cpu().numpy()
            #debug
            preds = torch.argmax(logits_clean, dim=1)
            clean_metrics.update(labels, probs_clean)
            probs_adv = torch.softmax(logits_adv, dim=1)[:,1].detach().cpu().numpy()
            adv_metrics.update(labels, probs_adv)

        total_samples += batch_size
    
    #Attack success rate
    adv_metrics.attack_success_rate(clean_metrics.all_probs)
    # Average L2 and L_inf norm
    adv_metrics.avg_l2 = adv_metrics.total_l2/total_samples
    adv_metrics.avg_linf = adv_metrics.total_linf/total_samples
   
    clean_results = clean_metrics.compute()
    adv_results = adv_metrics.compute()
    
    print("CLEAN RESULTS")
    clean_metrics.print(0)
    print(f"{attack_type} ATTACK RESULTS")
    adv_metrics.print(0)
    print(f"Attack Success Rate:  {adv_metrics.asr_list[0]}")
    print(f"Average L2:  {adv_metrics.avg_l2}")
    print(f"Average Linf:  {adv_metrics.avg_linf}")
   

    #saving metrics history
    history = {
        "number_of_samples": total_samples,
        "clean_auc": clean_metrics.auc_list,
        f"{attack_type}_auc": adv_metrics.auc_list,
        "clean_auc": clean_metrics.auc_list,
        f"{attack_type}_auc": adv_metrics.auc_list,
        "clean_f1": clean_metrics.f1_list,
        f"{attack_type}_f1": adv_metrics.f1_list,
        "clean_precision": clean_metrics.precision_list,
        f"{attack_type}_precision": adv_metrics.precision_list,
        "clean_recall": clean_metrics.recall_list,
        f"{attack_type}_recall": adv_metrics.recall_list,
        "clean_accuracy": clean_metrics.accuracy_list,
        f"{attack_type}_accuracy": adv_metrics.accuracy_list,
        f"{attack_type}_asr": adv_metrics.asr_list,
        "fgsm_epsilon_train": EPS,
        f"{attack_type}_epsilon_test": epsilon,
        f"{attack_type}_avg_l2": adv_metrics.avg_l2,
        f"{attack_type}_avg_linf": adv_metrics.avg_linf
    }
    save_history_json(history,f"test_results/{model_name}/test_results_{attack_type}/results_{attack_type}_eps_{epsilon}_num_samples_{total_samples}_norm.json")

    save_results_for_data(
        model_name=model_data,
        attack_label=attack_label_data,
        accuracy=adv_metrics.accuracy_list[0],
        auc=adv_metrics.auc_list[0],
        asr=adv_metrics.asr_list[0],
        avg_l2=adv_metrics.avg_l2,
        avg_linf=adv_metrics.avg_linf,
        output_path=f"test_results_for_data/{model_name}/test_results_{attack_type}/results_{attack_type}_eps_{epsilon}_num_samples_{total_samples}_norm.json"
    )

    return clean_metrics, adv_metrics


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

    # I load the trained clean model
    model_name = 'resnet50_fgsm_epoch_4_LR_0.0001_batchsize_32_WD_0.01_EPS_2_fine_tuned'
    checkpoint_path = f"models_10/fgsm_resnet50/{model_name}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
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
    test_small = balanced_subset(test_dataset, n_per_class=500)
    
    print("Initializing test loader...")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader_small = DataLoader(test_small, batch_size=32, shuffle=True)
    #test_loader_zoo = DataLoader(test_dataset_sub, batch_size=2, shuffle=True)


    clean_metrics, fgsm_metrics_1 = test_attack(model, test_loader, 'fgsm', 2/255, 'foolbox', model_name, device)
    clean_metrics, fgsm_metrics_2 = test_attack(model, test_loader, 'fgsm', 4/255, 'foolbox', model_name, device)
    clean_metrics, fgsm_metrics_3 = test_attack(model, test_loader, 'fgsm', 8/255, 'foolbox', model_name, device)
    clean_metrics, ifgsm_metrics = test_attack(model, test_loader, 'ifgsm', 8/255, 'foolbox', model_name, device)
    clean_metrics, pgd_metrics = test_attack(model, test_loader, 'pgd', 8/255, 'foolbox', model_name, device)
    #clean_metrics, square_metrics = test_attack(model, test_loader_small, 'square',TEST_EPS_SQUARE, 'art', model_name, device)
    #clean_metrics, tjsma_metrics = test_attack(model, test_loader, 'tjsma', 0.3, 'None', model_name, device)
    #clean_metrics, wjsma_metrics = test_attack(model, test_loader, 'wjsma', 0.3, 'None', model_name, device)
    #clean_metrics, gen_metrics = test_attack(model, test_loader, 'genattack', 0.3, 'foolbox', model_name, device)

    # plot roc curve 
    #plot_roc(clean_metrics.fpr, clean_metrics.tpr, clean_metrics.auc_list[0], "(test)", "clean_test")
    #plot_roc(fgsm_metrics.fpr, fgsm_metrics.tpr, fgsm_metrics.auc_list[0], "(test)", "fgsm_test")