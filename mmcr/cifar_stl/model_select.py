import os
from mmcr.cifar_stl.train_linear_classifier import train_classifier

def select_model(
    checkpoint_directory: str,
    dataset: str,
    save_dir: str,
):
    checkpoints = os.listdir(checkpoint_directory)
    accs = []
    max_acc = 0
    for checkpoint in checkpoints:
        acc_chkp = float(checkpoint.split('_')[-1][:-4])
        if acc_chkp > max_acc:
            max_acc = acc_chkp

    # will train classifiers for models with monitor accuracy within 1% of max accuracy
    checkpoints_to_test = []
    for checkpoint in checkpoints:
        acc_chkp = float(checkpoint.split('_')[-1][:-4])
        if acc_chkp >= max_acc - 1.0:
            checkpoints_to_test.append(checkpoint)

    print('Number of checkpoints to test: ', len(checkpoints_to_test))
    best_acc = 0.0
    for checkpoint in checkpoints_to_test:
        model, acc = train_classifier(
            checkpoint_directory + checkpoint,
            dataset=dataset,
            batch_size=1024,
            epochs=50,
            lr=0.1,
        )

        if acc > best_acc:
            best_acc = acc
            best_model = model

            print()
            print('New best accuracy: ', best_acc)
            print()

    if save_dir is not None:
        to_save = {"model": best_model.state_dict(), "acc": best_acc}
        torch.save(to_save, save_dir + "best_model.pth")

    return best_acc