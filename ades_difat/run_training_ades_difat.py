import os
os.environ['MPLCONFIGDIR'] = "/work/project"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
from torchvision import transforms
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from utils import *
import itertools
from test_attacks import test_attack
from diffusers import DDPMPipeline, DDPMScheduler
from training_ades_difat import train_pgd_at

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

seed=42
set_seed(42)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
   ])
params = {"lr": 1e-03,
"weight_decay": 5e-4,
"batch_size": 16
}

mode = 'baseline'

results     = []
    
print("Initializing Data loaders ......")
train_loader, val_loader, test_loader = get_data_loaders(transform, params['batch_size'])
checkpoint_path = f"{MODELS_DIR}/resnet50_clean_epoch_12_LR_0.0001_batchsize_32_WD_0.01_aug.pt"
#mode = 'baseline'
epsilon_scheduler = None
ades_scheduler = None
ades_optimizer = None
difat_purifier = None
lambda_ades = 0
lambda_mean = 4.0
beta = 0.01
train_metrics_clean = Metrics()
train_metrics_adv = Metrics()
val_metrics_clean = Metrics()
val_metrics_adv = Metrics()
start_epoch = 0
train_losses = []
num_epochs = 40
alpha_adv = 0.5
epsilon = 8/255
print(f"Start training PGD-AT with mode {mode}")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# I modify the last layer for binary classification
model.fc = nn.Sequential(
nn.Dropout(DROPOUT),
nn.Linear(model.fc.in_features, 2)
)
model = model.to(device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
criterion = nn.CrossEntropyLoss()
lr = 1e-4
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
# OPTIMIZER AND LR SCHEDULER RESNET50
optimizer = torch.optim.SGD(
model.parameters(),
lr=params['lr'],              # your empirically-successful ceiling was ~1e-5 under CyclicLR;
                       # SGD+momentum can typically tolerate a somewhat higher peak LR
                       # than Adam-family optimizers for the same model, but I'd still
                       # start conservatively and treat this as a value to sweep
momentum=0.9,
weight_decay=5e-4,    # matches the paper directly, no scaling needed here
)
# paper: decay at epochs 75/90 out of 100 → i.e. at 75% and 90% of total training
milestones = [int(0.75 * num_epochs), int(0.90 * num_epochs)]  # e.g. [15, 18] for your 20 epochs
LRscheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
# LEARNABLE ADES EPSILON SCHEDULER
if mode == "ades":
    print("ADES CHECK MAIN")
    ades_scheduler = LearnableEpsilonScheduler()
    ades_scheduler = ades_scheduler.to(device)
    ades_optimizer = torch.optim.AdamW(ades_scheduler.parameters(), weight_decay=1e-5)
    lambda_ades = 0.001
#DIFAT
if mode == "difat":
    pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    difat_purifier = DiffusionPurifier(pipe.unet, pipe.scheduler, device=device)
#Baseline with linear epsilon scheduler
# EPSILON SCHEDULER
#if mode == "baseline":
type_sched = 'linear'
num_epochs_rampup = num_epochs/2
epsilon_scheduler = CurriculumEpsilonScheduler(
        eps_start=0/255, eps_end=8/255,
        num_epochs_rampup=int(num_epochs_rampup), type=type_sched,
        adaptive=True, patience=5, num_epochs_per_eps=1
    )
##############################
print(f"Starting training PGD-AT with mode: {mode}")

trained_model, train_metrics_clean, train_metrics_adv, val_metrics_clean, val_metrics_adv, train_losses, epoch_total_times, loss_type = train_pgd_at(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=num_epochs,
    optimizer=optimizer,
    LRscheduler=LRscheduler,
    criterion=criterion,
    device=device,
    train_losses=train_losses,
    train_metrics_clean=train_metrics_clean,
    train_metrics_adv=train_metrics_adv,
    val_metrics_clean=val_metrics_clean,
    val_metrics_adv=val_metrics_adv,
    lr=params["lr"],
    mode=mode,
    epsilon_scheduler=epsilon_scheduler,
    ades_scheduler=ades_scheduler,
    ades_optimizer=ades_optimizer,
    lambda_ades=lambda_ades,
    lambda_mean=lambda_mean,
    beta =beta,
    seed=seed,
    difat_purifier=difat_purifier,
    save_model=True
)
val_adv_acc = val_metrics_adv.accuracy_list[num_epochs-1]
val_clean_acc = val_metrics_clean.accuracy_list[num_epochs-1]
torch.cuda.synchronize(device)
torch.cuda.empty_cache()
 # testing
clean_metrics, fgsm_metrics_1 = test_attack(trained_model, test_loader, 'fgsm', 2/255, 'foolbox', " ", " ", "FGSM (eps=2/255)", device, save_results=False)
clean_metrics, fgsm_metrics_2 = test_attack(trained_model, test_loader, 'fgsm', 8/255, 'foolbox', " ", " ", "FGSM (eps=8/255)", device, save_results=False)
clean_metrics, ifgsm_metrics_2 = test_attack(trained_model, test_loader, 'ifgsm', 8/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
clean_metrics, pgd_metrics_2 = test_attack(trained_model, test_loader, 'pgd', 8/255, 'None', " ", " ", "PGD", device, save_results=False)
clean_metrics, ifgsm_metrics_1 = test_attack(trained_model, test_loader, 'ifgsm', 2/255, 'foolbox', " ", " ", "IFGSM",device, save_results=False)
clean_metrics, pgd_metrics_1 = test_attack(trained_model, test_loader, 'pgd', 2/255, 'None', " ", " ", "PGD", device, save_results=False)
if torch.isnan(torch.tensor(val_clean_acc)):
    results.append({
        'params': params, 
        'val_clean_acc': None, 
        'val_adv_acc':  None,
        'test_clean_acc':  None,
        'test_fgsm_small_acc':  None,
        'test_fgsm_big_acc':  None,
        'test_ifgsm_big_acc':  None,
        'test_pgd_big_acc':  None,
        'status': 'failed_nan'})
else:
    results.append({
        'params': params, 
        'loss_type': loss_type,
        'lambda_mean': lambda_mean,
        'beta': beta,
        'val_clean_acc': val_clean_acc, 
        'val_adv_acc': val_adv_acc,
        'test_clean_acc': clean_metrics.accuracy_list[0],
        'test_clean_auc': clean_metrics.auc_list[0],
        'test_clean_asr': 0,
        'test_fgsm_small_acc': fgsm_metrics_1.accuracy_list[0],
        'test_fgsm_small_auc': fgsm_metrics_1.auc_list[0],
        'test_fgsm_small_asr': fgsm_metrics_1.asr_list[0],
        'test_fgsm_big_acc': fgsm_metrics_2.accuracy_list[0],
        'test_fgsm_big_auc': fgsm_metrics_2.auc_list[0],
        'test_fgsm_big_asr': fgsm_metrics_2.asr_list[0],
        'test_ifgsm_small_acc': ifgsm_metrics_1.accuracy_list[0],
        'test_ifgsm_small_auc': ifgsm_metrics_1.auc_list[0],
        'test_ifgsm_small_asr': ifgsm_metrics_1.asr_list[0],
        'test_ifgsm_big_acc': ifgsm_metrics_2.accuracy_list[0],
        'test_ifgsm_big_auc': ifgsm_metrics_2.auc_list[0],
        'test_ifgsm_big_asr': ifgsm_metrics_2.asr_list[0],
        'test_pgd_small_acc': pgd_metrics_1.accuracy_list[0],
        'test_pgd_small_auc': pgd_metrics_1.auc_list[0],
        'test_pgd_small_asr': pgd_metrics_1.asr_list[0],
        'test_pgd_big_acc': pgd_metrics_2.accuracy_list[0],
        'test_pgd_big_auc': pgd_metrics_2.auc_list[0],
        'test_pgd_big_asr': pgd_metrics_2.asr_list[0],
        'epoch_total_times': epoch_total_times,
        'status': 'ok'})
    
out_dir = f'pgd_{mode}'
os.makedirs(out_dir, exist_ok=True)
eps_sched_label = "_"
if mode == "baseline" and epsilon_scheduler != None:
    eps_sched_label = f"_{type_sched}_eps_sched_neprampup_{num_epochs_rampup}"
elif mode == "baseline" and epsilon_scheduler == None:
    eps_sched_label = f"_fixed_eps_{epsilon}"
elif mode == "ades":
    mode = f"ades_{lambda_ades}_{loss_type}"

with open(f'{out_dir}/grid_search_pgd_{mode}_{eps_sched_label}_lr_{lr}_{num_epochs}_alpha_{alpha_adv}_freeze_2.json', 'w') as f:
    json.dump(results, f, indent=4)
print(f"Clean acc: {val_clean_acc:.4f} | Adv acc: {val_adv_acc:.4f}")