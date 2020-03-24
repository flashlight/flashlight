Debugging flashlight
====================

Common issues and pitfalls
--------------------------
- When spawning new threads it's required to call ``af::setDevice()`` before
  using any ArrayFire operations.
- Although ArrayFire's JIT generally works well, one may encounter bugs. One
  common issue is that due to JIT's laziness, the input arrays accumulate
  in memory, leading to OOMs. One can add ``af::eval()`` in the right places
  to sidestep this, but using it too liberally may degrade performance gains
  from JIT.
- ArrayFire's arrays are copy-on-write. This provides value semantics without
  unnecssary copies. However, due to this, modifying an array may incur a
  potentially expensive copy operation. On the other hand, ``fl::Variable`` has
  shared pointer semantics; copying a Variable just creates another pointer
  to the same array and autograd node.

Printing info
-------------
- ``af::info()`` and ``af::deviceInfo()`` can provide info about ArrayFire
  version, GPU devices, compute capabilities, etc.
- ``af::printMemInfo()`` can be useful in debugging OOMs.
- ``af::print()`` can be used to inspect arrays. Note: the 1st argument should
  be a message/name, and the 2nd argument shoudld be an array, not Variable:

::

  auto x = fl::constant(1.0, 1);
  auto y = x * x;
  y.backward();
  af::print("x", x.array());
  af::print("x_grad", x.grad().array());

- Or, one can first convert them to vectors using ``af::array::host()`` and do
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
