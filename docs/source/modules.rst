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
.. doxygenfile:: Activations.h

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

Loss
^^^^
.. doxygenclass:: fl::Loss
   :members:

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
.. doxygenfile:: nn/Init.h

Serialization
-------------
.. doxygenclass:: fl::Serializer
    :members:

Utils
--------------
.. doxygenfile:: modules/Utils.h

DistributedUtils
----------------
.. doxygenfile:: modules/DistributedUtils.h
