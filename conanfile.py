import os

from conans import CMake, ConanFile, tools
from conans.errors import ConanInvalidConfiguration


class FlashlightConan(ConanFile):
    name = "flashlight"
    version = "0.1"
    license = "BSD"
    author = "jacobkahn jacobkahn1@gmail.com"
    url = "https://github.com/facebookresearch/flashlight"
    requires = [
        "arrayfire/3.7.1@conan/stable",
        "cereal/1.3.0",
        "gtest/1.10.0",
        "openmpi/3.0.0@bincrafters/stable",
    ]
    description = "flashlight is a fast, flexible machine learning library written entirely in C++."
    topics = ("machine learning", "deep learning", "autograd")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "backend": ["CPU", "CUDA", "OPENCL"]}
    default_options = {"shared": True, "backend": "CUDA"}
    generators = "cmake"

    def source(self):
        self.run("git clone https://github.com/facebookresearch/flashlight.git")

    def configure(self):
        if self.settings.os == "Windows":
            raise ConanInvalidConfiguration("flashlight is not compatible with Windows.")
        if self.options.backend != "CUDA":
            raise ConanInvalidConfiguration(
                "flashlight with Conan only supports the CUDA backend."
            )
        # ArrayFire package options
        self.options["arrayfire"].graphics = False
        self.options["arrayfire"].unified_backend = False
        self.options["arrayfire"].cpu_backend = self.options.backend == "CPU"
        self.options["arrayfire"].cuda_backend = self.options.backend == "CUDA"
        self.options["arrayfire"].opencl_backend = self.options.backend == "OPENCL"

    def build(self):
        self.output.info("Running flashlight CMake build...")
        cmake = CMake(self)
        cmake.definitions["FLASHLIGHT_BACKEND"] = self.options.backend
        cmake.definitions["FL_BUILD_TESTS"] = False
        cmake.definitions["FL_BUILD_EXAMPLES"] = False
        cmake.configure(source_folder="flashlight")
        cmake.build()

    def package(self):
        self.copy("*.h", dst="include", src="flashlight")
        self.copy("*flashlight.lib", dst="lib", keep_path=False)
        self.copy("*.so", dst="lib", keep_path=False)
        self.copy("*.a", dst="lib", keep_path=False)

    def package_info(self):
        self.cpp_info.libs = ["flashlight"]
