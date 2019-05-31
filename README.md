# flashlight
[![CircleCI](https://circleci.com/gh/facebookresearch/flashlight.svg?style=svg)](https://circleci.com/gh/facebookresearch/flashlight)

flashlight is a fast, flexible machine learning library written entirely in C++
from the Facebook AI Research Speech team and the creators of Torch and
Deep Speech. It uses the [ArrayFire](https://github.com/arrayfire/arrayfire)
tensor library and features just-in-time compilation with modern C++.
flashlight supports both CPU (still in active development) and GPU backends for
maximum portability, and has an emphasis on efficiency and scale.

All documentation (including build/install instructions) can be found
[here](https://fl.readthedocs.io/en/latest/)

Experimental and in-progress project components are located in `flashlight/contrib`. Breaking changes may be made to APIs therein.

Contact: vineelkpratap@fb.com, awni@fb.com, jacobkahn@fb.com, qiantong@fb.com,
jcai@fb.com,  gab@fb.com, vitaliy888@fb.com, locronan@fb.com

flashlight is being very actively developed. See
[CONTRIBUTING](CONTRIBUTING.md) for more on how to help out.

## Acknowledgments
Some of flashlight's code is derived from
[arrayfire-ml](https://github.com/arrayfire/arrayfire-ml/).

## License
flashlight is under a BSD license. See [LICENSE](LICENSE).
