.. _serial:

Serialization Library
=====================

.. doxygenfile:: common/Serialization.h

.. cpp:function:: template <typename... Args> void save(const std::string& filepath, const Args&... args)

.. cpp:function:: template <typename... Args> void save(std::ostream& ostr, const Args&... args)

.. cpp:function:: template <typename... Args> void load(const std::string& filepath, Args&... args)

.. cpp:function:: template <typename... Args> void load(std::istream& istr, Args&... args)

.. cpp:function::  template <typename T> detail::Versioned<T> versioned(T&& t, uint32_t minVersion, uint32_t maxVersion = UINT32_MAX);
