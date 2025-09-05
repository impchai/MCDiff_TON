import numpy as np
import torch
from torch.optim import Adam, AdamW
from tqdm import tqdm
import pickle

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=50,
    foldername="",
):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad == True], lr=config["lr"], weight_decay=1e-3)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.5 * config["epochs"])
    p2 = int(0.75 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[p1, p2],
                                                        gamma=0.1)

    best_valid_loss = 1e10
    early_stop = 0
    for epoch_no in range(config["epochs"]):
        avg_loss = 0

        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1)

        if valid_loader is not None and (epoch_no +
                                         1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0,
                          maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss":
                                avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )

            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                torch.save(model.state_dict(), output_path)
                early_stop = 0

            else:
                early_stop = early_stop + 1
                print('Early stop ITER', early_stop)
        if early_stop > 5:
            break



def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * np.sum(
        np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return np.sum(np.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(np.quantile(forecast[j : j + 1], quantiles[i], axis=1))
        q_pred = np.array(q_pred)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)



def evaluate(model,
             test_loader,
             nsample=50,
             scaler=1,
             mean_scaler=0,
             foldername=""):
    print( '| nsample=', nsample)

    with torch.no_grad():
        model.eval()
        evalpoints_one_total = 0
        tv_distance_total = 0
        all_target = []
        all_observed_time = []
        all_generated_samples = []
        all_samples_list_all = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                out0 = model.evaluate(test_batch)
                sample0, c_targets, observed_time = out0

                samples_list = [sample0]  # list of [B,K,L] tensors
                for _ in range(nsample - 1):
                    sample_i, _, _ = model.evaluate(test_batch)
                    samples_list.append(sample_i)

                samples_tensor = torch.stack(samples_list, dim=0)  # [n_repeat,B,K,L]

                samples_mean_tensor = samples_tensor.mean(dim=0)  # [B,K,L]

                samples_mean = samples_mean_tensor.detach().cpu().numpy()
                c_target = c_targets.detach().cpu().numpy()

                B, K, L = c_target.shape


                evalpoints_one_total += B

                tv_distance = 0.0
                for i in range(B):
                    tv_distance += 0.5 * np.abs(samples_mean[i] - c_target[i])
                tv_distance_res = tv_distance / K
                tv_distance_total += tv_distance_res.sum().item()


                all_target.append(c_target)                    # [B,K,L]
                all_observed_time.append(observed_time)        # tensor
                all_generated_samples.append(samples_mean)     # [B,K,L]
                all_samples_list_all.append(samples_tensor.cpu().numpy())  # [n_repeat,B,K,L]

                it.set_postfix(
                    ordered_dict={
                        "tv_distance": tv_distance_total / max(1, evalpoints_one_total),
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )


            with open(foldername + f"/generated_outputs_nsample{nsample}.pk", "wb") as f:
                all_target = np.array(all_target)                  # [num_batches,B,K,L]
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = np.array(all_generated_samples)     # [num_batches,B,K,L]
                all_samples_list_all = np.array(all_samples_list_all)       # [num_batches,n_repeat,B,K,L]

                pickle.dump(
                    [all_generated_samples, all_target, all_observed_time,
                     scaler, mean_scaler, all_samples_list_all],
                    f,
                )

            CRPS = calc_quantile_CRPS(all_target, all_generated_samples, mean_scaler, scaler)


            print("tv-distance:", tv_distance_total / max(1, evalpoints_one_total))
            print("CRPS:", CRPS)

