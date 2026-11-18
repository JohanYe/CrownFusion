# Third-party notices

- **SiT** (Ma et al., "Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant
  Transformers", https://github.com/willisma/SiT): model definitions, `train.py`, `transport/`, and utilities in
  `sit_utils/` are derived from SiT / DiT (Meta Platforms, MIT License).
  See `sit_utils/LICENSE.txt`.
- **FLUX.1-schnell VAE** (`black-forest-labs/FLUX.1-schnell`, Apache-2.0): downloaded at run time from
  Hugging Face and used to initialize the autoencoder. Weights are not redistributed here.
- **3DShape2VecSet** (Zhang et al., https://github.com/1zb/3DShape2VecSet, MIT License): the point-cloud
  encoder in `pc_encoder.py` is adapted from this code.
- **Omages** (https://github.com/3dlg-hcvc/omages, MIT License): the omage utilities in `omages_utils.py`
  are adapted from this code.
- **nanoGPT** (Karpathy, https://github.com/karpathy/nanoGPT, MIT License): `CrossAttention` in
  `sit_utils/models.py` is modified from its attention module.

## MIT license notices

3DShape2VecSet:

> Copyright 2023 Biao Zhang

Omages:

> Copyright (c) 2024 3dlg-hcvc

nanoGPT:

> Copyright (c) 2022 Andrej Karpathy

All are used under the following terms:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
> associated documentation files (the "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
> following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial
> portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
> LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
> EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
> IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
> THE USE OR OTHER DEALINGS IN THE SOFTWARE.
