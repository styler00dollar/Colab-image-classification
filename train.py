import argparse
from dataloader import DataModule
from CustomTrainClass import CustomTrainClass
import pytorch_lightning as pl

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_path', type=str, required=True, help='Input folder.')
    parser.add_argument('--validation_path', type=str, required=True, help='Validation data path.')
    parser.add_argument('--default_root_dir', type=str, required=True)
    parser.add_argument('--means', type=str, nargs='+', required=True)
    parser.add_argument('--std', type=str, nargs='+', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--aug', type=str, required=True)
    parser.add_argument('--size', type=int)

    parser.add_argument('--model_train', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--diffaug_activate', type=bool)
    parser.add_argument('--policy', type=str)
    parser.add_argument('--num_workers', type=int, default = 4)


    #parser.add_argument('--fp16_mode', type=bool, default=False, required=False)
    args = parser.parse_args()

    # converting strings to float inside array
    args.means = [float(i) for i in args.means]
    args.std = [float(i) for i in args.std]

    dm = DataModule(training_path=args.training_path, validation_path=args.validation_path, test_path=args.validation_path, num_workers = args.num_workers, size = args.size, batch_size=args.batch_size, means=args.means, std=args.std)
    model = CustomTrainClass(model_train=args.model_train, num_classes=args.num_classes, diffaug_activate=args.diffaug_activate, policy=args.policy, aug=args.aug)
    # skipping validation with limit_val_batches=0
    #gpus=1, limit_val_batches=0,
    trainer = pl.Trainer(gpus=1, max_epochs=800, progress_bar_refresh_rate=20, default_root_dir=args.default_root_dir)

    # For resuming training
    """
    checkpoint_path = None #'test.ckpt'

    if checkpoint_path is not None:
        trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path, gpus=1, max_epochs=800, progress_bar_refresh_rate=20, default_root_dir=args.default_root_dir)

        model = model.load_from_checkpoint(checkpoint_path)
        dm = DataModule(batch_size=16, training_path=training_path, validation_path=args.validation_path, num_workers = 16, size = 256)
        checkpoint = torch.load(checkpoint_path)
        trainer.checkpoint_connector.restore(checkpoint, on_gpu=True)
        trainer.checkpoint_connector.restore_training_state(checkpoint)
        pl.Trainer.global_step = checkpoint['global_step']
        pl.Trainer.epoch = checkpoint['epoch']
        print("Checkpoint was loaded successfully.")

        #############################################
    """
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
