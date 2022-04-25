Serialization API
=================

flashlight uses the `cereal <http://uscilab.github.io/cereal/>`_ library for serialization. It provides :ref:`utility macros and functions<serial>` to easily define serialization properties for arbitrary classes and save or load modules, variables, standard library containers, and many other common C++ types.

Serializing Objects
^^^^^^^^^^^^^^^^^^^

To serialize individual instances on the fly, use `save` function:

::

  // objects to be saved
  std::shared_ptr<fl::Module> module = .... ;
  std::unordered_map<int, std::string> config = .... ;

  fl::save("<filepath>", module, config);

To deserialize, use `load` function:

::

  // create the objects to be loaded in advance
  std::shared_ptr<fl::Module> module;
  std::unordered_map<int, std::string> config;

  fl::load("<filepath>", module, config);


Serializing Classes
^^^^^^^^^^^^^^^^^^^

If we define a new class that needs to support serialization, flashlight provides easy-to-use macros to avoid writing boilerplate code as required by ``cereal``. For example:

::

  // Animal.h
  class Animal {
   private:
    std::string name_;

    FL_SAVE_LOAD(name_) // Defines save, load functions

   public:
    Animal(const std::string& name) : name_(name) {}

    virtual std::string whoAmI() {
      return "Reporting from Base class - Animal.";
    }

    virtual ~Animal() = default;
  };

  CEREAL_REGISTER_TYPE(Animal)

  // Cat.h
  class Cat : public Animal {
   private:
    bool isHungry_;

    FL_SAVE_LOAD_WITH_BASE(Animal, isHungry_) // Defines save, load functions

   public:
    Cat(const std::string& name, bool isHungry)
        : Animal(name), isHungry_(isHungry) {}

    virtual std::string whoAmI() override {
      return "Hello! I'm a cat.";
    }
  };

  CEREAL_REGISTER_TYPE(Cat)


For additional guidlines, check out `flashlight/common/Serialization.h`.
To summarize, save() shouldn't mutate, load() may assume output is default-
constructed, and one shouldn't save `long`, `size_t`, `dim_t` etc. due to
differing sizes on 32-bit vs 64-bit platforms.

.. note::
  In the example above, if instead using ``FL_SAVE_LOAD(name_, isHungry_)`` for `Cat`,
  `CEREAL_REGISTER_POLYMORPHIC_RELATION(Cat, Animal) must be `declared in Cat.h <http://uscilab.github.io/cereal/polymorphism.html>`_.


Custom Serialization
^^^^^^^^^^^^^^^^^^^^

flashlight also supports defining custom serializers. For example:

::

  // Dog.h
  class Dog : public Animal {
   private:
    int capacity_;
    std::function<int(int)> eatFunc_; // `cereal` doesn't support function objects

    FL_SAVE_LOAD_DECLARE() // Declares save, load functions
   public:
    Dog(int capacity)
        : Animal("dog"),
          capacity_(capacity),
          eatFunc_([capacity](int b) { return b - capacity; }) {}
  };

  template <class Archive>
  void Dog::save(Archive& ar, const uint32_t /* version */) const {
    ar(cereal::base_class<Animal>(this), capacity_);
  }

  template <class Archive>
  void Dog::load(Archive& ar, const uint32_t /* version */) {
    ar(cereal::base_class<Animal>(this), capacity_);
    auto capacity = capacity_;
    eatFunc_ = [capacity](int b) { return b - capacity; };
  }

  CEREAL_REGISTER_TYPE(Dog)


.. warning::

  When `serializing smart pointers <https://uscilab.github.io/cereal/pointers.html>`_, ``cereal`` will only save data from the underlying object once.

  ::

    template <class Archive>
    void MyClass::save(Archive& ar, const uint32_t /* version */) const {
      Variable v(Tensor({50, 50}), true);
      ar(v); // `v` is saved
      v = v + 1; // Creates a new variable
      ar(v); // `v` is saved
      v.tensor() += 1; // `SharedData` pointer in `v` storing the Tensor is still the same
      ar(v); // `v` is NOT saved
    }


Versioning
^^^^^^^^^^

flashlight supports versioning for saving and loading to make maintaining backward compatibility easier:

::

  // Panda.h
  class Panda : public Animal {
   private:
    std::string color_;
    bool eating_;

    FL_SAVE_LOAD_WITH_BASE(Animal, color_, fl::versioned(eating_, 1))

    // fl::versioned(eating_, 1) will make sure the object is saved/loaded only
    // for versions >= 1. While using custom serialization, `version` number is passed
    // an argument to save/load functions and can be used to serialize appropriately.

   public:
    Panda(const std::string& col) : Animal("panda"), color_(col), eating_(true) {}
  };

  CEREAL_REGISTER_TYPE(Panda)
  CEREAL_CLASS_VERSION(Panda, 2) // associate class with a version number
