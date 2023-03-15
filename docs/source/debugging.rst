Debugging flashlight
====================

Common issues and pitfalls with the ArrayFire Backend
-----------------------------------------------------
- When spawning new threads it's required to call ``fl::setDevice()`` before
  using any ArrayFire operations.
- Although ArrayFire's JIT generally works well, one may encounter bugs. One
  common issue is that due to JIT's laziness, the input arrays accumulate
  in memory, leading to OOMs. One can add ``fl::eval()`` in the right places
  to sidestep this, but using it too liberally may degrade performance gains
  from JIT.
- ArrayFire's arrays are copy-on-write. This provides value semantics without
  unnecssary copies. However, due to this, modifying an array may incur a
  potentially expensive copy operation. On the other hand, ``fl::Variable`` has
  shared pointer semantics; copying a Variable just creates another pointer
  to the same array and autograd node.

Printing Info
-------------
- ``fl::getMemMgrInfo`` will write memory manager information to an output
  stream, if implemented by the active backend.
- Flashlight ``Tensor``s can be written directly to an arbitrary output
  stream via their ``operator<<``(e.g.
  ``std::cout << "myTensor" << myTensor << std::endl;``) or printed to
  standard out by using ``fl::print()``.

By including ArrayFire headers ``<arrayfire/arrayfire.h>`` directly, one has
access to useful debugging functions:
- ``af::info()`` and ``af::deviceInfo()`` can provide info about ArrayFire version, GPU devices, compute capabilities, etc.
- ``af::printMemInfo()`` can be useful in debugging OOMs.
- ``af::print()`` can be used to inspect arrays. Note: the 1st argument should be a message/name, and the 2nd argument shoudld be an array, not Variable:

::

  auto x = fl::constant({1}, 1.0);
  auto y = x * x;
  y.backward();
  std::cout << "x" << x.tensor() << std::endl;
  std::cout << "x_grad" << x.grad().tensor() << std::endl;

- Or, one can first convert them to vectors using ``fl::Tensor::host()`` and do
  more customized inspection. Do not use ``scalar()`` to print each element;
  that would be very slow.

Crashes
-------
flashlight uses standard C++ exceptions to indicate failure states. Unhandled
exceptions will abort the program, producing a core dump which can be analyzed
with ``gdb`` and similar tools. One can also use libraries like
`glog <https://github.com/google/glog>`_ and
`Boost.Stacktrace <https://github.com/boostorg/stacktrace>`_ to print out
a stack trace upon crashing.
For instance, with glog, one simply needs to add the line

::

  google::InstallFailureSignalHandler();

to the ``main`` function.
