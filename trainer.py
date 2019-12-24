import sys
import fire
import time
from itertools import count
import torch
import torch.optim as optim

from model import *
from data import *


epoch_l = -1 # for logging
training_log = '/fakepath4log/training.log'
recon_w = 0.005

def train(log_interval, model, train_loader, optimizer, epoch, num_sample, batch_size, ckpt_interval):
    model.train()
    t0 = time.time()
    ckpt_interval = count(ckpt_interval, ckpt_interval)
    ckpt_time = next(ckpt_interval)
    with open(training_log, 'a') as f:
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, _ = data  # _: mol_idx
            v, fp, ff, recon = model(*inputs)
            margin_loss = model.margin_loss(v, fp)
            recon_loss = model.reconstruction_loss(ff, recon, weight=recon_w)
            loss = margin_loss + recon_loss
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                log = 'Train Epoch: {} [{:.0f}/{} ({:.0f}%)]\tLoss: {:.6f}\t' \
                      'Margin Loss: {:.6f}\tRecon Loss: {:.6f}\tEclipsed Time: {:.1f}'.format(
                    epoch, batch_idx * len(inputs[1]), num_sample, 100. * batch_idx / (num_sample / batch_size),
                    loss.item(), margin_loss.item(), recon_loss.item(), time.time() - t0
                )
                global epoch_l  # it seems useless...
                epoch_l = round(epoch - 1 + batch_idx / (num_sample / batch_size), 2)

                print(log)
                f.write(log + '\n')
                f.flush()
            if int((time.time() - t0) / 3600) == ckpt_time:
                save(model, 'checkpoint', epoch_l, f)
                ckpt_time = next(ckpt_interval)


def test(model, train_loader, epoch, num_sample, batch_size):
    t0 = time.time()
    with torch.no_grad():
        model.eval()
        with open(training_log, 'a') as f:
            for batch_idx, data in enumerate(train_loader):
                inputs, _ = data  # _: mol_idx
                v, fp, ff, recon = model(*inputs)
                margin_loss = model.margin_loss(v, fp)
                recon_loss = model.reconstruction_loss(ff, recon, weight=recon_w)
                loss = margin_loss + recon_loss

                log = 'Test Epoch: {} [{:.0f}/{} ({:.0f}%)]\tLoss: {:.6f}\t' \
                      'Margin Loss: {:.6f}\tRecon Loss: {:.6f}\tEclipsed Time: {:.1f}'.format(
                    epoch, batch_idx * len(inputs[1]), num_sample, 100. * batch_idx / (num_sample / batch_size),
                    loss.item(), margin_loss.item(), recon_loss.item(), time.time() - t0
                )
                global epoch_l
                epoch_l = round(epoch - 1 + batch_idx / (num_sample / batch_size), 2)
                print(log)
                f.write(log + '\n')
                f.flush()


def save(model, state, epoch, log_file_obj):
    t = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    torch.save(model.state_dict(), "/fakepath4ckpt/ForceFieldCapsNet_Epoch{}_{}_{}.pt".format(str(epoch), state, t))
    log = 'Checkpoint Saved at ' + t + '.'
    print(log)
    log_file_obj.write(log + '\n')
    log_file_obj.flush()


def scatter(inputs, num_chunk):
    # version for array
    def chunk_it(seq, num):
        assert isinstance(num, (int, list, tuple))
        if isinstance(num, int):
            chunk_sizes = [len(seq) / float(num), ] * num
        else:
            chunk_sizes = map(int, num)
        out = []
        last = 0.0
        for size in chunk_sizes:
            out.append(seq[int(last):int(last + size)])
            last += size
        return out

    nums_atoms_ckd = chunk_it(inputs[3], num_chunk)
    chunk_sizes = [sum(num_atoms) for num_atoms in nums_atoms_ckd]
    gs_charge_ckd, atom_type_ckd, pos_ckd = [chunk_it(i, chunk_sizes) for i in inputs[:3]]
    inputs = list(zip(gs_charge_ckd, atom_type_ckd, pos_ckd, nums_atoms_ckd))
    return inputs


def main(grid_size=64, batch_size=27, epochs=1, lr=0.0001, device=0, device_ids=(0, 1, 2), seed=1,
         log_interval=2, ckpt_interval=6, save_model=True, num_workers=5, ckpt=None, num_sample=None, train_ratio=0.9):

    try:
        '''
        --grid-size: grid size of box (default: 64)
        --batch-size: input batch size for training (default: x)
        --epochs: number of epochs to train (default: x)
        --lr: learning rate (default: x)
        --device: master cuda device (default: 0)
        --device-ids: for data parallel training (default: [0, 1, 2])
        --seed: random seed (default: 1)
        --log-interval: how many batches to wait before logging training status (default: 2)
        --save-model: For Saving the current Model (default: True)
        --num-workers: number of workers for pytorch dataloder (default: 5)
        --ckpt: checkpoint file to continue training (default: None)
        --num-sample: the number of molecules sampled from the full ZINC dataset (default: None, using full dataset)
        --train-ratio: training/test ratio (default: 0.9)
        '''

        torch.manual_seed(seed)
        device = torch.device('cuda:' + str(device))

        # zinc h5 loader
        arrayh5 = [
            '/fakepath4zincdataset/Ctran_molarray.h5',
            '/fakepath4zincdataset/Dtran_molarray.h5',
            '/fakepath4zincdataset/Etran_molarray.h5',
        ]
        fph5 = [
            '/fakepath4zincdataset/Ctran_maccskey.h5',
            '/fakepath4zincdataset/Dtran_maccskey.h5',
            '/fakepath4zincdataset/Etran_maccskey.h5',
        ]
        zinc = ZINCH5Dataset(arrayh5, fph5, )
        num_sample = zinc.sample(num_samp=num_sample, train_ratio=train_ratio)
        num_sample = [int(num_sample * train_ratio), num_sample - int(num_sample * train_ratio)]
        loader = ZINCH5Dataloader(zinc, batch_size=batch_size, num_workers=num_workers)

        model = ForceFiledCapsNet(device_ids=device_ids, grid_size=grid_size).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # get training started
        with open(training_log, 'a') as f:
            # print('Training Start.')
            f.write('\n\nTraining started at ' + time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '\n')
            f.write(' '.join(sys.argv) + '\t')
            f.write('\n')
            f.write(str(model) + '\n')
            f.flush()

        for epoch in range(1, epochs + 1):
            # zinc h5
            train_loader = loader('train')
            test_loader = loader('test')

            train(log_interval, model, train_loader, optimizer, epoch, num_sample[0], batch_size, ckpt_interval)
            test(model, test_loader, epoch, num_sample[1], batch_size)

        if save_model:
            with open(training_log, 'a') as f:
                save(model, 'completed', epochs, f)

    except KeyboardInterrupt:
        save_model = input('save model? (y/n)')
        if save_model == 'y':
            with open(training_log, 'a') as f:
                save(model, 'interrupted', epoch_l, f)

    except Exception:
        with open(training_log, 'a') as f:
            save(model, 'error', epoch_l, f)
        raise


if __name__ == '__main__':
    fire.Fire(main)

