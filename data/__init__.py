'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(data_set,args, phase):
    '''create dataloader '''
    if phase == 'train':
        return torch.utils.data.DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)
    elif phase == 'val':
        return torch.utils.data.DataLoader(
            data_set, batch_size= 1, shuffle=False, num_workers=8, pin_memory=True)
    else:
        raise NotImplementedError(
            'Dataloader [{:s}] is not found.'.format(phase))


def create_dataset(args, phase):
    '''create dataset'''
    from data.dataset import UIEDataset_test as D

    dataset = D(dataroot=args.data_dir,
                resolution=256,
                split=phase,
                data_len=args.data_len,
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           'UIEB'))
    return dataset
def create_datasetforsample(data_dir, phase):
    '''create dataset'''
    from data.dataset import UIEDatasetsample as D
    dataset = D(dataroot=data_dir,
                resolution=256,
                split=phase,
                data_len=-1,
                )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           'UIEB'))
    return dataset
