from torch.utils.data import DataLoader
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info
import torchvision.transforms as transforms
import tensorflow as tf
from tensorflow import nest


def torch_to_tf(torch_dataset):
    """
    Returns a tensorflow dataset with same data.
    :type torch_dataset: a torch Dataset
    """
    allowed_types = [Dataset, IterableDataset, DataLoader]
    if not any(isinstance(torch_dataset, tp) for tp in allowed_types):
        raise TypeError("`torch_dataset` must be a torch Dataset or a Dataloader")
    else:
        def to_tf_generator():
            for sample in torch_dataset:
                tf_sample = nest.map_structure(lambda t: tf.convert_to_tensor(t.numpy()), sample)
                if isinstance(tf_sample, list):
                    tf_sample = tuple(tf_sample)
                yield tf_sample

        batch = next(iter(torch_dataset))
        output_signature = nest.map_structure(lambda t: tf.TensorSpec(shape=tuple(t.shape),
                                                                      dtype=tf.as_dtype(str(t.dtype)[6:])), batch)
        if isinstance(output_signature, list):
            output_signature = tuple(output_signature)

        def unit_test():
            gen = to_tf_generator()
            gen_sample = next(gen)
            nest.assert_same_structure(gen_sample, tuple(output_signature), check_types=True, expand_composites=False)

        try:
            unit_test()
        except:
            raise AssertionError('Sorry, probably the dataset is of a complex type that this function cant handle :)')

        tf_dataset = tf.data.Dataset.from_generator(to_tf_generator, output_signature=output_signature)

    return tf_dataset.prefetch(1)


def tf_to_torch(tf_dataset: tf.data.Dataset, tf_ds_info, split:str, transforms_list=[]) -> IterableDataset:
    """
    Returns torch Dataset(iterable) with same data with multi-process loading.  
    :return_type torch_dataset which returns structure with numpy arrays.
    """
    allowed_types = [tf.data.Dataset]
    if not any(isinstance(tf_dataset, tp) for tp in allowed_types):
        raise TypeError("`tf_dataset` should be of the type `tf.data.Dataset`")

    else:
        class torch_ds(Dataset):

          def __init__(self):
            self.X = tf_dataset.repeat(-1).as_numpy_iterator()

          def __len__(self):
            return ds_info.splits[split].num_examples

          def __getitem__(self, index):
                image, label = self.X.next()
                for t in transfroms_list:
                    image = t(image)
                sample = (image, label)
                return sample
            

    return torch_ds(tf_dataset)
