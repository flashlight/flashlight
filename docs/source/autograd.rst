Introduction to Autograd
========================

This tutorial gives a brief overview of Variable and Functions which are the core
components flashlight's automatic differentiation library. We recommend reading the
`Getting Started <http://arrayfire.org/docs/gettingstarted.htm>`_ section from Arrayfire
before starting this tutorial so that you are familiar with its syntax.

Variable
--------

A Variable is a wrapper around an Arrayfire array which is central to array
operations in flashlight.

::

  auto var = fl::Variable(af::range(1.0, af::dim4(2, 3)), true /* isCalcGrad */);
  af::print("A", A.array()); // get the underlying array from a Variable
  // A
  // [2 3 1 1]
  //    0.0000     0.0000     0.0000
  //    1.0000     1.0000     1.0000
  std::cout << A.dims() // dimension of the Variable
  // 2 3 1 1
  std::cout << A.elements() // number of elements in the Variable
  // 6
  std::cout << A.type() // type of the Variable
  // 0 <- corresponds to float32 enum

The second parameter to Variable constructor `isCalcGrad` specifies whether we
want to compute the gradient for this Variable.

.. figure:: images/variable.png
   :width: 340px
   :align: center
   :height: 180px
   :alt: variable design


The above figure shows the high-level design of a Variable class. SharedData and
SharedGrad are stored as shared pointers inside the Variable. SharedData stores
the array object while SharedGrad stores the gradient, the inputs etc. The additional
members stored in Variable class help in facilitating easy differentiation,
which we'll go through in the following sections.

We note that copying a Variable makes a shallow copy and both objects will
refer to the same underlying data like array, grad etc...

The complete documentation of the Variable API can be found :ref:`here<variable>`.

::

  auto a = fl::Variable(af::randu(10, 10), false /* isCalcGrad */);
  auto b = a; // shallow copy!
  a.array() = af::constant(2.0, 3, 2);
  af::print("b", b.array()); // The array wrapped the variable 'b' is also modified
  b
  [3 2 1 1]
      2.0000     2.0000
      2.0000     2.0000
      2.0000     2.0000


Functions
---------

Similar to Arrayfire arrays, Variables can be used to perform array operations.

::

  auto expVar = exp(var);
  af::print("expVar", expVar.array());
  // expVar
  // [2 3 1 1]
  //    1.0000     1.0000     1.0000
  //    2.7183     2.7183     2.7183

  auto A = fl::Variable(af::constant(1.0, af::dim4(2, 3)), false /* isCalcGrad */);
  auto B = fl::Variable(af::constant(2.0, af::dim4(2, 3)), true /* isCalcGrad */);
  auto AB = A * B;
  af::print("AB", AB.array());
  // AB
  // [2 3 1 1]
  //    2.0000     2.0000     2.0000
  //    2.0000     2.0000     2.0000


A complete list of functions can be found :ref:`here<functions>`.

Autograd
--------

If at least one of the inputs require gradient, each output Variable stores its
input Variables to keep track of computation graph and gradFunc, a lambda function
to calculate the gradient of its inputs given the gradient with respect to the output
Variable.

To compute the gradient of a Variable with respect to all its inputs recursively
in the computation graph, we call the `backward()` function on the Variable. This
will sort the Variables in the computation graph in a topological order and compute
the gradient by chain rule. For example, calling `expVar.backward()` will calculate
the gradient of `expVar` with respect to its input Variable `var`.

::

  expVar.backward();
  af::print("expVarGrad", expVar.grad().array()); // expVar.grad() is a Variable
  af::print("varGrad", var.grad().array());
  // expVarGrad
  // [2 3 1 1]
  //     1.0000     1.0000     1.0000
  //     1.0000     1.0000     1.0000

  // varGrad
  // [2 3 1 1]
  //     1.0000     1.0000     1.0000
  //     2.7183     2.7183     2.7183

  AB.backward();
  af::print("ABGrad", AB.grad().array());
  af::print("AGrad", A.grad().array());
  // ABGrad
  // [2 3 1 1]
  //     1.0000     1.0000     1.0000
  //     1.0000     1.0000     1.0000

  // AGrad
  // [2 3 1 1]
  //     2.0000     2.0000     2.0000
  //     2.0000     2.0000     2.0000

.. warning::
  Calling `B.grad()` will throw an exception here since `isCalcGrad` is set to `false`

TODO: Add step-by-step execution details on an example computation graph

Various Optimizations
---------------------

JIT Compiler
############

Arrayfire (array backend for flashlight) uses a JIT compiler to combine many light
weight functions into a single kernel launch. Here is a very simple example which
illustrates this.

::

  auto A = fl::Variable(
      af::randu(1000, 1000), true); // 'A' is allocated, Total Memory: 4 MB
  auto B = 2.0 * A; // 'B' is not allocated, Total Memory : 4 MB
  auto C = 1.0 + B; // 'C' is not allocated, Total Memory :  4 MB
  auto D = log(C); // 'D' is not allocated (yet), Total Memory :  4 MB
  D.eval(); // only 'D' is allocated, Total Memory : 8 MB


This helps in improving performance and reducing memory usage. Visit the ArrayFire
docs for more details about the `ArrayFire JIT <https://arrayfire.com/performance-improvements-to-jit-in-arrayfire-v3-4/>`_.

In-place Operations and More
############################

Since the flashlight uses `shared_ptr` semantics for storing its internal array, any
array is automatically deleted when the Variable goes out of scope.

::

  auto A = fl::Variable(af::randu(1000, 1000), false); // Total Memory: 4 MB
  auto B = fl::Variable(af::randu(1000, 1000), false); // Total Memory: 8 MB
  auto C =  fl::transpose(A); // Total Memory: 12 MB
  C = fl::matmul(C, fl::transpose(B)); // Total Memory: 12 MB. Previous 'C' goes out of scope

We have carefully optimized memory needed during the forward and backward passes of
the computation graph. Some functions in autograd do not need to keep their input
data in order to compute their gradient. E.g. `sum (+)`, `transpose` etc.
For these functions, the output Variable doesn't store the 'SharedData' member
of the input Variables that are not needed so that the underlying array can be
freed if it is not referenced in other places.

::

  // Note calcGrad is set to true here. Total Memory: 4 MB
  auto A = fl::Variable(af::randu(1000, 1000), true);

  // Intermediate arrays are not stored. Total Memory: 8 MB
  auto C =  fl::transpose(fl::transpose(A));

Retain Graph
############

The `backward()` function takes an additional boolean parameter `retainGraph`
which is `false` by default. Keeping this `false` will make sure we clear the Variables
as soon as they are not required while performing the backward pass. This helps in
reducing peak memory usage while computing backward pass. It is not recommended to
set `retainGraph` to `true` unless you need to inspect intermediate values in the
backward graph or need to retain the full backward graph for other purposes.

::

  auto A = fl::Variable(af::randu(1000, 1000), true);
  auto B = fl::Variable(af::randu(1000, 1000), true);
  auto C = fl::matmul(A, B);
  C = fl::transpose(C);
  C = 1.0 + C;
  C.backward(false); // Note `retainGraph` is false by default

.. graphviz::

  digraph G {

   graph[rankdir=LR]
    node [fontname=Arial];

    C  [label="C", shape = "Box"]
    F  [label="F (1.0)"]
    E  [label="E"]
    D  [label="D"]
    B  [label="B", shape = "Box"]
    A  [label="A", shape = "Box"]

    E, F -> C  [label="+", color="steelblue"]
    D -> E [label=" transpose"]
    A, B -> D [label=" matmul", color="firebrick"]

    label = "Computation Graph"

  }



For example in the above graph, the intermediate Variable E can be deleted as
soon as the gradients of D are computed.
