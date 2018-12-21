Serialization API
=================

flashlight uses `cereal <http://uscilab.github.io/cereal/>`_ library for serialization.
It provides :ref:`utility macros and functions<serial>` to easily define serialization
of classes and save/load modules, variables, STL containers and many other common C++ types.

Serializing Objects
^^^^^^^^^^^^^^^^^^^

To serialize, use `save` function

::

  // objects to be saved
  std::shared_ptr<fl::Module> module = .... ;
  std::unordered_map<int, std::string> config = .... ;

  fl::save("<filepath>", module, config);

To deserialize, use `load` function

::

  // create the objects to be loaded in advance
  std::shared_ptr<fl::Module> module;
  std::unordered_map<int, std::string> config;

  fl::load("<filepath>", module, config);

Serializing Classes
^^^^^^^^^^^^^^^^^^^

If you define a new class the needs to support serialization, flashlight provides easy-to-use macros
to avoid writing boilerplate code required by `cereal` library. Here is an example -

::

  // Animal.h
  class Animal {
   private:
    std::string name_;

    FL_SAVE_LOAD(name_) // Defines save, load functions

   public:
    Animal(const std::string& name) : name_(name) {}

    inline virtual std::string whoAmI() {
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

    inline virtual std::string whoAmI() override {
      return "Hello! I'm a cat.";
    }
  };

  CEREAL_REGISTER_TYPE(Cat)

.. note::
  In the above example, if you instead use `FL_SAVE_LOAD(name_, isHungry_)` for `Cat` class,
  `CEREAL_REGISTER_POLYMORPHIC_RELATION(Cat, Animal)` should be `declared in Cat.h <http://uscilab.github.io/cereal/polymorphism.html>`_.

Custom Serialization of Classes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we provide an example to show how to define custom save/load function for class serialization

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

  While `serializing smart pointers <https://uscilab.github.io/cereal/pointers.html>`_, `cereal` library makes sure the
  underlying data object is saved only once.
  ::

    template <class Archive>
    void MyClass::save(Archive& ar, const uint32_t /* version */) const {
      Variable v(af::array(50, 50), true);
      ar(v); // `v` is saved
      v = v + 1; // Creates a new variable
      ar(v); // `v` is saved
      v.array() += 1; // `SharedData` pointer in `v` storing the array is still the same
      ar(v); // `v` is NOT saved
    }

Versioning
^^^^^^^^^^

flashlight supports versioning for save/load functions so that it is easier to maintain
backward compatibility.

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
