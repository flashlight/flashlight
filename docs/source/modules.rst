.. _modules:

Modules
=======

Containers
----------

Module
^^^^^^
.. doxygenclass:: fl::Module
   :members:
   :protected-members:
   :undoc-members:

.. doxygenclass:: fl::UnaryModule
   :members:
   :undoc-members:

.. doxygenclass:: fl::BinaryModule
   :members:
   :undoc-members:

Container
^^^^^^^^^
.. doxygenclass:: fl::Container
   :members:
   :undoc-members:

Sequential
^^^^^^^^^^
.. doxygenclass:: fl::Sequential
   :members:
   :undoc-members:

Layers
------

Activations
^^^^^^^^^^^
.. doxygenclass:: fl::Sigmoid
  :members:

.. doxygenclass:: fl::Tanh
  :members:

.. doxygenclass:: fl::HardTanh
  :members:

.. doxygenclass:: fl::ReLU
  :members:

.. doxygenclass:: fl::LeakyReLU
  :members:

.. doxygenclass:: fl::PReLU
  :members:

.. doxygenclass:: fl::ELU
  :members:

.. doxygenclass:: fl::ThresholdReLU
  :members:

.. doxygenclass:: fl::GatedLinearUnit
  :members:

.. doxygenclass:: fl::LogSoftmax
  :members:

.. doxygenclass:: fl::Log
  :members:

BatchNorm
^^^^^^^^^
.. doxygenclass:: fl::BatchNorm
   :members:

Conv2D
^^^^^^
.. doxygenclass:: fl::Conv2D
   :members:

Dropout
^^^^^^^
.. doxygenclass:: fl::Dropout
   :members:
   :undoc-members:

Embedding
^^^^^^^^^
.. doxygenclass:: fl::Embedding
   :members:

LayerNorm
^^^^^^^^^
.. doxygenclass:: fl::LayerNorm
   :members:

Linear
^^^^^^
.. doxygenclass:: fl::Linear
   :members:

Padding
^^^^^^^
.. doxygenclass:: fl::Padding
   :members:

Pool2D
^^^^^^
.. doxygenclass:: fl::Pool2D
   :members:

Reorder
^^^^^^^
.. doxygenclass:: fl::Reorder
   :members:

RNN
^^^
.. doxygenclass:: fl::RNN
   :members:

Transform
^^^^^^^^^
.. doxygenclass:: fl::Transform
    :members:

View
^^^^
.. doxygenclass:: fl::View
   :members:
   :undoc-members:

WeightNorm
^^^^^^^^^^
.. doxygenclass:: fl::WeightNorm
   :members:

Losses
------

AdaptiveSoftMaxLoss
^^^^^^^^^^^^^^^^^^^
.. doxygenclass:: fl::AdaptiveSoftMaxLoss
   :members:

BinaryCrossEntropy
^^^^^^^^^^^^^^^^^^
.. doxygenclass:: fl::BinaryCrossEntropy
   :members:

CategoricalCrossEntropy
^^^^^^^^^^^^^^^^^^^^^^^
.. doxygenclass:: fl::CategoricalCrossEntropy
   :members:

MeanAbsoluteError
^^^^^^^^^^^^^^^^^
.. doxygenclass:: fl::MeanAbsoluteError
   :members:

MeanSquaredError
^^^^^^^^^^^^^^^^
.. doxygenclass:: fl::MeanSquaredError
   :members:

Initialization
--------------

.. doxygengroup:: nn_init_utils

Utils
--------------
.. doxygengroup:: nn_utils
    :content-only:

DistributedUtils
----------------
.. doxygengroup:: nn_distributed_utils
    :content-only:
