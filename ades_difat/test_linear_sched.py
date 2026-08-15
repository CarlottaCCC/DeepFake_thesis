
def linear_scheduler(current_epoch, num_epochs):
    if current_epoch < num_epochs // 2:
        target_eps = (current_epoch / (num_epochs // 2)) * (8/255)
    else:
        target_eps = 8/255
    return target_eps

for epoch in range(0,24):
    target_eps = linear_scheduler(epoch, 25)
    print(target_eps)