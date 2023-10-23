import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
import argparse

from configs.config_utilities import load_config
from models.pyramid import build_pon, build_hpon
from dataset import build_dataloaders
from criterion import OccupancyCriterion
from logger import TensorboardLogger
import utilities.torch as torch_utils
from utilities.line_notify_tracking import Send_notify_to_line


def create_experiment(
    config,
    args,
) -> str:
    if args.resume_experiment is not None:
        log_dir = f"{config.log_dir}/{args.resume_experiment}"
        print("Restoring experiment from: " + log_dir)
        if args.tag is not None:
            print(f"`--tag {args.tag}` is not used.")
    else:
        experiment = (
            f"{args.network}_{args.loss}_{config.nuscenes_version}_"
            + datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )
        if args.tag is not None:
            experiment = args.tag + "_" + experiment
        log_dir = f"{config.log_dir}/{experiment}"
        print("Creating new experiment at: " + log_dir)
    return log_dir


def main():
    parser = argparse.ArgumentParser(
        description="Training a model for bird's-eye-view map prediction."
    )
    parser.add_argument(
        "--network",
        choices=["H-PON", "PON"],
        default="H-PON",
        help="network to train, default: `H-PON`",
    )
    parser.add_argument(
        "--loss",
        choices=["occupancy", "bce"],
        default="occupancy",
        help="""loss function, default: `occupancy`; 
        `occupancy` - occupancy loss, `bce` - binary cross entropy loss""",
    )
    parser.add_argument(
        "--tag",
        help="tag included in front of experiment name (optional)",
    )
    parser.add_argument(
        "--resume-experiment",
        help="""name of experiment to load and resume training 
        (must use with `--resume-epoch`), 
        default format: [network]_[loss]_[nuscenes_version]_[datetime]""",
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        help="""saved checkpoint epoch to load and resume training 
        (must use with `--resume-experiment`)""",
    )
    parser.add_argument(
        "--line-notify",
        help="A Line nofity token for experiment tracking in Line application",
    )
    parser.add_argument("--save-best", help="save best epoch", action="store_true")
    args = parser.parse_args()
    config = load_config("configs/configs.yml")

    # Create directory for experiment
    log_dir = create_experiment(config, args)

    # Build dataset loader
    train_loader, validate_loader = build_dataloaders(config)

    # Detect device
    device = torch_utils.detect_device()
    print(f"Training on {device}")

    # Build network
    if args.network == "H-PON":
        network = build_hpon(config, htfm_method="stack").to(device)
    elif args.network == "PON":
        network = build_pon(config).to(device)

    # Build criterion
    if args.loss == "occupancy":
        criterion = OccupancyCriterion(
            config.prior,
            config.xent_weight,
            config.uncert_weight,
            config.weight_mode,
        ).to(device)
    elif args.loss == "bce":
        criterion = nn.BCEWithLogitsLoss().to(device)

    # Build optimizer
    optimizer = optim.Adam(network.parameters(), lr=config.lr)

    # Define Line notify module
    if args.line_notify is not None:
        line_notify = Send_notify_to_line(
            line_token=args.line_notify,
            exp_name=log_dir,
            model=args.network,
            batch_size=config.batch_size,
            loss=args.loss,
            optimizer=optimizer.__class__.__name__,
            lr=config.lr,
            total_epoch=config.epochs,
        )

    # Load checkpoint
    if args.resume_experiment is not None:
        load_checkpoint_path = f"{log_dir}/saved_{str(args.resume_epoch).zfill(4)}.pt"
        print(f"Loading checkpoint from {load_checkpoint_path}")
        checkpoint = torch.load(load_checkpoint_path)
        network.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_step = checkpoint["step"]
        initial_epoch = checkpoint["epoch"] + 1
        min_loss = checkpoint["min_loss"]
    else:
        initial_step = 0
        initial_epoch = 0
        min_loss = float("inf")

    logger = TensorboardLogger(
        device=device,
        log_dir=log_dir,
        validate_loader=validate_loader,
        criterion=criterion,
        loss=args.loss,
        num_classes=config.num_class,
        initial_step=initial_step,
        min_loss=min_loss,
    )

    # Log experiment config in case of creating new experiment
    if args.resume_experiment is None:
        config_log_table = f"""
            <table>
                <tr>
                    <th>Nuscenes Version</th>
                    <th>Batch Size</th>
                    <th>Num Workers</th>
                    <th>Learning Rate</th>
                    <th>Number of epochs</th>
                    <th>Device</th>
                    <th>Network</th>
                    <th>Loss</th>
                    <th>Optimizer</th>
                    <th>Augmented: hflip</th>
                </tr>
                <tr>
                    <td>{config.nuscenes_version}</td>
                    <td>{config.batch_size}</td>
                    <td>{config.num_workers}</td>
                    <td>{config.lr}</td>
                    <td>{config.epochs}</td>
                    <td>{device}</td>
                    <td>{args.network}</td>
                    <td>{args.loss}</td>
                    <td>{optimizer.__class__.__name__}</td>
                    <td>{config.hflip}</td>
                </tr>
            </table>
        """
        logger.writer.add_text(
            "Experiment Configurations", config_log_table, global_step=0
        )

    for epoch in tqdm(range(initial_epoch, config.epochs)):
        try:
            for batch in train_loader:
                images, labels, masks, calibs = batch
                images = images.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                calibs = calibs.to(device)

                logits = network(images, calibs)

                # Compute loss
                if args.loss == "occupancy":
                    loss = criterion(logits, labels, masks).to(device)
                elif args.loss == "bce":
                    loss = criterion(logits, labels.float()).to(device)

                # Compute gradient
                optimizer.zero_grad()
                loss.backward()

                # Update weights
                optimizer.step()

                logger.log_step(loss=loss.item())
            logger.log_epoch(network, epoch)

            if args.line_notify is not None:
                line_notify.send_message(current_epoch=epoch)
        except Exception as e:
            if args.line_notify is not None:
                line_notify.send_error(error_message=e)
            raise e

        # Save checkpoint every n epochs
        if (epoch + 1) % config.num_epochs_to_save_checkpoint == 0:
            print(f"Saving model at epoch {epoch}")
            checkpoint_path = log_dir + f"/saved_{str(epoch).zfill(4)}.pt"
            torch.save(
                dict(
                    epoch=epoch,
                    step=logger.training_step,
                    model_state_dict=network.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    min_loss=logger.min_loss,
                ),
                checkpoint_path,
            )

        # Save best epoch
        if args.save_best and logger.save_model:
            print(f"Saving best model at epoch {epoch}")
            checkpoint_path = log_dir + f"/saved_best.pt"
            torch.save(
                dict(
                    epoch=epoch,
                    step=logger.training_step,
                    model_state_dict=network.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    min_loss=logger.min_loss,
                ),
                checkpoint_path,
            )

    # Save last epoch
    if (epoch + 1) % config.num_epochs_to_save_checkpoint != 0:
        print(f"Saving model at epoch {epoch}")
        checkpoint_path = log_dir + f"/saved_{str(epoch).zfill(4)}.pt"
        torch.save(
            dict(
                epoch=epoch,
                step=logger.training_step,
                model_state_dict=network.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                min_loss=logger.min_loss,
            ),
            checkpoint_path,
        )


if __name__ == "__main__":
    main()
