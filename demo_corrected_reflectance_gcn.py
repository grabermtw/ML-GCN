import argparse
from corrected_reflectance_engine import *
from corrected_reflectance_models import *
from corrected_reflectance import *
from corrected_reflectance_util import *

"""
python3 demo_corrected_reflectance_gcn.py
"""

parser = argparse.ArgumentParser(description='WILDCAT Training for Corrected Reflectance dataset')
parser.add_argument('--image-size', '-i', default=128, type=int,
                    metavar='N', help='image size (default: 128)')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=25, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


def main_corrected_reflectance():
    global args, best_prec1, use_gpu
    args = parser.parse_args()
    start_time = time.time()
    use_gpu = torch.cuda.is_available()

    training_data, validation_data = load_data("labeled_data.csv")

    train_dataset = CorrectedReflectanceDataset(training_data)
    val_dataset = CorrectedReflectanceDataset(validation_data)
    num_classes = 2

    model = gcn_resnet101(num_weather_classes=num_classes, num_terrain_classes=num_classes, in_channel=128, t=0.4, adj_file='corrected_reflectance.pkl')#'data/coco/coco_adj.pkl')

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes':num_classes}
    state['difficult_examples'] = True
    state['save_model_path'] = 'checkpoint/corrected_reflectance/'
    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    # state['device_ids'] = args.device_ids
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)

    print("Total runtime:", str(time.time() - start_time))

if __name__ == '__main__':
    main_corrected_reflectance()
