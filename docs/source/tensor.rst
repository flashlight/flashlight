Tensor
======

Flashlight's Tensor abstraction is itself an API. See the ``README`` in ``flashlight/fl/tensor`` for more on implementing a Tensor backend. The below documentation documents Tensor functions and usage of the ``Tensor`` class.

A ``Tensor`` in FLashlight is a multidimensional container for data that has a shape. Memory locations and operations on Tensors are defined in per a backend's implementation.

The Tensor class
----------------

.. doxygenclass:: fl::Tensor
   :members:

Indexing
--------

Flashlight Tensors can be indexed by index literal values, ranges, spans (``fl::span``) or other Tensors (advanced indexing).

.. doxygenclass:: fl::range
   :members:

.. doxygenstruct:: fl::Index
   :members:

Functions on Tensors
--------------------

.. doxygengroup:: tensor_functions
    :content-only:
