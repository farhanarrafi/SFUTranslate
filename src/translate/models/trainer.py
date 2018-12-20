import math
from tqdm import tqdm
import sys
from translate.configs.loader import ConfigLoader
from translate.configs.utils import get_resource_file
from translate.models.RNN.estimator import STSEstimator
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.padder import get_padding_batch_loader
from translate.models.backend.utils import device
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader
from translate.readers.dummydata import DummyDataset
from translate.logging.utils import logger


def prepare_dummy_datasets(configs: ConfigLoader):
    train_ = DummyDataset(configs, ReaderType.TRAIN)
    test_ = DummyDataset(configs, ReaderType.TEST)
    dev_ = DummyDataset(configs, ReaderType.DEV)
    return train_, test_, dev_


def prepare_parallel_dataset(configs: ConfigLoader):
    raise NotImplementedError


def make_model(configs: ConfigLoader, train_dataset: AbsDatasetReader):
    created_model = SequenceToSequence(configs, train_dataset).to(device)
    created_estimator = STSEstimator(configs, created_model, train_dataset.compute_bleu)
    return created_model, created_estimator


if __name__ == '__main__':
    opts = ConfigLoader(get_resource_file("default.yaml"))
    dataset_type = opts.get("reader.dataset.type", must_exist=True)
    epochs = opts.get("trainer.optimizer.epochs", must_exist=True)
    if dataset_type == "dummy":
        train, test, dev = prepare_dummy_datasets(opts)
    elif dataset_type == "parallel":
        train, test, dev = prepare_parallel_dataset(opts)
    else:
        raise NotImplementedError
    model, estimator = make_model(opts, train)
    print_every = int(0.25 * int(math.ceil(float(len(train)/float(model.batch_size)))))

    for epoch in range(epochs):
        logger.info("Epoch {}/{} begins ...".format(epoch + 1, epochs))
        iter_ = 0
        train.allocate()
        itr_handler = tqdm(get_padding_batch_loader(train, model.batch_size), ncols=100,
                           desc="[E {}/{}]-[B {}]-[L {}]-#Batches Processed".format(
                               epoch + 1, epochs, model.batch_size, 0.0), total=math.ceil(len(train)/model.batch_size))
        for input_tensor_batch, target_tensor_batch in itr_handler:
            iter_ += 1
            loss_value, decoded_word_ids = estimator.step(input_tensor_batch, target_tensor_batch)
            itr_handler.set_description("[E {}/{}]-[B {}]-{}-#Batches Processed".format(
                epoch + 1, epochs, model.batch_size, str(estimator)))
            if iter_ % print_every == 0:
                dev.allocate()
                ref_sample, hyp_sample = "", ""
                for batch_i_tensor, batch_t_tensor in get_padding_batch_loader(dev, model.batch_size):
                    ref_sample, hyp_sample = estimator.step_no_grad(batch_i_tensor, batch_t_tensor)
                print("", end='\n', file=sys.stderr)
                logger.info(u"Sample: E=\"{}\", P=\"{}\"\n".format(ref_sample, hyp_sample))
                dev.deallocate()
        print("\n", end='\n', file=sys.stderr)
        train.deallocate()
