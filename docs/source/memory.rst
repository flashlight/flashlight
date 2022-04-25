.. _memory_management:

Memory Management
=================

.. note::
  This section is only applicable if using the ArrayFire backend.

Flashlight's ArrayFire backend also contains a framework to implement and use custom memory managers for devices running computation.

By default, for better performance, flashlight uses a custom memory manager (``fl::CachingMemoryManager``) implemented with this framework in place of the default ArrayFire memory manager. When flashlight is linked to, ``MemoryManagerInstaller::installDefaultMemoryManager()`` is invoked, which sets the default ArrayFire memory manager to be an instance of the ``CachingMemoryManager``. This behavior can be changed by modifying the function accordingly.

A custom memory manager can be set after flashlight initializes; see the documentation for ``fl::MemoryManagerInstaller`` below for setting custom memory managers.

.. warning::
  **The default ArrayFire memory manager is no longer supported in flashlight**; using it explicitly may result in significantly degraded performance.


Defining a Custom Memory Manager
--------------------------------

.. doxygenclass:: fl::MemoryManagerAdapter
   :members:

Activating a Memory Manager in ArrayFire
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: fl::MemoryManagerInstaller
   :members:

Native Memory Management Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: fl::MemoryManagerDeviceInterface
   :members:
