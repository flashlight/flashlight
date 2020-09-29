<img align="left" width="45" height="70" src="flashlight_logo.png" alt="flashlight_logo"/>

# flashlight

[![CircleCI](https://circleci.com/gh/facebookresearch/flashlight.svg?style=shield)](https://circleci.com/gh/facebookresearch/flashlight)
[![Documentation Status](https://img.shields.io/readthedocs/fl.svg)](https://fl.readthedocs.io/en/latest/)
[![Docker Image Build Status](https://github.com/facebookresearch/flashlight/workflows/Publish%20Docker%20images/badge.svg)](https://hub.docker.com/r/flml/flashlight/tags)
[![Join the chat at https://gitter.im/flashlight-ml/community](https://img.shields.io/gitter/room/flashlight-ml/community)](https://gitter.im/flashlight-ml/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

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

See the documentation for more on how to [use flashlight with your own project](https://fl.readthedocs.io/en/latest/installation.html#building-your-project-with-flashlight).

## Acknowledgments
Some of flashlight's code is derived from
[arrayfire-ml](https://github.com/arrayfire/arrayfire-ml/).

## License
flashlight is under a BSD license. See [LICENSE](LICENSE).
