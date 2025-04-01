---
title: FLUX.1的一些理解
published: 2025-04-02
description: '读完Flux的一些理解'
image: ''
tags: [Flux, Stable Diffusion]
category: ''
draft: false 
lang: ''
---

主要对flux的pipeline进行一个理解，包含了flux的具体步骤，对diffusion的再理解，以及flux和stable diffusion3有什么不同之处。

### diffusion model

diffusion包含前向和后向的两个过程，前向过程是向原始图像中不断加入随机的高斯噪声，最终形成一个近似的高斯噪声，而逆向过程呢是将高斯噪声恢复成原始图像的生成过程。也就是所谓的加噪去噪。

### flow matching

flow matching和扩散不同，它是直面生成过程的，通过将已知的分布转换成真实数据分布来生成数据，使用flow来描述生成过程中的每一步的概率密度的变化。

Flow是数据在时间上的变换映射，数据变化导致对应时刻的密度变化，记作 $\phi_t$。在生成过程中，这些变化的概率密度构成的集合被称为概率密度路径，写作 $p_t$. 那么现在假设概率密度路径的长度为 $T$，初始的数据就是 $x_0 \sim p_0(x_0)$，目标数据就是 $x_T \sim p_T(x_t)$，那么从0到t就是：

$$
x_T = \phi(x_0) = \phi_T\circ...\circ\phi_{t+1}\circ\phi_{t}\circ...\circ\phi_1(x_0)
$$

并且对任意步 $x_t$:

$$
x_t = \phi_t(x_{t-1}) \\
x_{t-1} = \phi_t^{-1}(x_t)
$$

在这里呢可以根据概率密度的定义：任意时刻的概率密度积分为1，那就可以写成下面这个式子：

$$
\int p_t(x_t)dx_t = \int p_{t-1}(x_{t-1})dx_{t-1} = 1
$$

再往下推就可以写成：

$$
p_t(x_t) = p_{t-1}(x_{t-1})\det\left|\frac{\partial x_{t-1}}{\partial x_{t}}\right|=p_{t-1}(\phi_t^{-1}(x_t))\det\left|\frac{\partial \phi_t^{-1}}{\partial x_{t}}(x_t)\right|\\

$$

等式中的行列式代表的就是流 $\phi_t$的雅可比行列式，继续往下推，可以这么写：

$$
p_T(x_T)=p_0(x_0)\prod_{t=1}^T \det\left|\frac{\partial \phi_t^{-1}}{\partial x_{t}}(x_t)\right|
$$

行列式的本质是空间缩放的度量，相当于每次变换的时候都对概率密度进行归一化，这就是Normalizing Flow： $p_t=[\phi_t]p_0$， $p_0$已知，只要求解 $\phi_t$即可。那么直接用Neural Ordinary Differential Equations (NODE) 来对雅可比行列式的ODE建模来求 $\phi_t$。为了实现这个，需要把离散的时间步映射到连续的 $t \in [0,1]$, 0表示起始时间，1表示目标时间，这样就把 $p_t$重新定义为连续时间和数据点的笛卡尔积： $p : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}_{>0}$,且 $\int p_t(x)dx=1$; $\phi_t$定义为 $\phi : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$,那么这个就是所谓的Continuous Normalizing Flow(CNF)

Flow本质上是数据点的映射，且可微双射，数据点在时间上的变换可以用flow的梯度表示，这样所有数据点的梯度就可以构成一个向量场 $v : [0,1] \times \mathbb{R}^d \rightarrow \mathbb{R}^d$

$$
\frac{d}{dt}\phi_t(x)=v_t(\phi_t(x))\\
\phi_0(x)=x
$$

Flow Matching 现在通过对向量场建模，训练过后，生成时根据模型推理出来的 $v_t$，就可以用ODE数据求解器（flux中用的是欧拉法）来生成数据，由此得出目标函数：

$$
\mathcal{L}*{FM}(\theta) = \mathbb{E}*{t, p_{t}(x)} \left\| v_{t}(x) - u_{t}(x) \right\|^2
$$

其中 $u_t$ 是目标数据概率密度对应的目标向量场。

再往下走就是条件流，一般的选择都是基于条件高斯的

$$
p_{t}\left(x \mid x_{1}\right) = \mathcal{N}\left(x \mid \mu_{t}\left(x_{1}\right), \sigma_{t}\left(x_{1}\right)^2 I\right)
$$

这样条件 $p_t$对应的flow $\psi_t$就成为一个仿射变换，这个仿射变换将数据映射到均值为 $\mu_t(x_1)$标准差为 $\sigma_t(x_1)$的正态分布上，那么最终可以写出来的条件向量场形式为：

$$
u_{t}\left(x \mid x_{1}\right) = \frac{\sigma_{t}^{\prime}\left(x_{1}\right)}{\sigma_{t}\left(x_{1}\right)} \left(x - \mu_{t}\left(x_{1}\right)\right) + \mu_{t}^{\prime}\left(x_{1}\right)
$$

这里还是重新学了一下flow matching的理论，算是对之前没理解的一个补全吧

### pipeline

```python
class FluxPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
```

构造函数部分可以看到所有组件的类

之前阅读代码发现了flux区别于其他stable diffusion的一个点就是图块化patchify操作

```python
@staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents
```

这里把2 x 2个像素在通道维度上拼接到了一起，而SD3这个操作是写在去噪网络中的，因此SD3去噪网络的in_channel是16，而fllux就是64。

现在进入_call_来看

```python
def __call__(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 7.0,
)
```

对比sd3少了一组prompt且没用negative prompt。少一组提示词是因为少用了一个文本编码器。而没有负面提示词是因为该模型是指引蒸馏过的，在文本指引上没那么灵活。之后的内容和所有的扩散模型的pipeline一样：1. 检查输入是否合法 2. 给输入文本编码 3. 随机生成初始化噪声，主要对timesteps，num_inference_steps这些一知半解的地方进行进一步理解，主要代码区域如下：

```python
# 5. Prepare timesteps
  sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
  image_seq_len = latents.shape[1]
  mu = calculate_shift(
      image_seq_len,
      self.scheduler.config.base_image_seq_len,
      self.scheduler.config.max_image_seq_len,
      self.scheduler.config.base_shift,
      self.scheduler.config.max_shift,
  )
  timesteps, num_inference_steps = retrieve_timesteps(
      self.scheduler,
      num_inference_steps,
      device,
      timesteps,
      sigmas,
      mu=mu,
  )
  num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
  self._num_timesteps = len(timesteps)

  # handle guidance
  if self.transformer.config.guidance_embeds:
      guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
      guidance = guidance.expand(latents.shape[0])
  else:
      guidance = None
```

首先定义了一个sigma，根据num_inference_steps(默认28)的值生成1到1/num_inference_steps的等间距的一个序列（大小由num_inference_steps)来决定，这个sigma表示的是每个时间步所对应的噪声强度，下面定义了一个mu，是个新的变量（和sd对比）,通过calculate_shift来计算获得，可以推测这个变量是用来调整时间步的偏移量，mu计算完之后会传入retrieve_timesteps函数

```python
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu
```

retrieve_timesteps函数返回timesteps和num_inference_steps，在函数内部是使用调度器来设置timesteps，flux中的scheduler是scheduling_flow_match_euler_discrete

```python
def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
      """
      Sets the discrete timesteps used for the diffusion chain (to be run before inference).

      Args:
          num_inference_steps (`int`):
              The number of diffusion steps used when generating samples with a pre-trained model.
          device (`str` or `torch.device`, *optional*):
              The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
      """

      if self.config.use_dynamic_shifting and mu is None:
          raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

      if sigmas is None:
          self.num_inference_steps = num_inference_steps
          timesteps = np.linspace(
              self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
          )

          sigmas = timesteps / self.config.num_train_timesteps

      if self.config.use_dynamic_shifting:
          sigmas = self.time_shift(mu, 1.0, sigmas)
      else:
          sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

      sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
      timesteps = sigmas * self.config.num_train_timesteps

      self.timesteps = timesteps.to(device=device)
      self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

      self._step_index = None
      self._begin_index = None
```

timesteps和sigmas息息相关，下面梳理一下这个所谓timestep到底时如何变换的：

1. 首先是根据num_train_timesteps用np.linspace生成timesteps，sigmas由timesteps归一化获得，这里的timesteps是调度器里面自己初始化的，所以len就是训练步数
2. 接下来，传入了num_inference_steps,这时候还是通过np.linspace，根据num_inference_steps的值来生成timesteps,这里用到了_sigma_to_t，功能是通过sigma获得时间步t(因为前面有过归一化)那么这时候的len(timesteps) = num_inference_steps, 然后再计算sigmas，和前面一样，归一化

```python
def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

self.num_inference_steps = num_inference_steps
timesteps = np.linspace(
    self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
)

sigmas = timesteps / self.config.num_train_timesteps
```

1. 最重要的一步，就是把这个timestep又放到训练步长的尺度上

感觉非常绕，那就用例子来说明，假设以下情景

- num_train_timesteps=2000
- num_inference_timesteps=200

初始化，timesteps生成[1,2000]的线性序列，长度2000，那可以就是[1, 2, … , 2000]，计算sigma，也就是归一化[1/2000,…,1], 这个代表噪声强度; 接下来进入set_timesteps函数，首先需要传入sigma,这里的sigma和第一步的不同，它是传入进来的，就在上面有过定义：是根据num_inference_steps来计算的，也就是在[0,200]内线性取200个值之后进行归一化的: [1/200,1]；最后一步就是算timesteps，那么这里就直接乘上2000，也就是做到了把这个噪声强度放大了，这样才能和训练的时候对齐。

再宏观想一想：训练2000步，推理200步，推理步数明显少于训练步数，那去噪的时候的噪声强度就需要放大来和训练的时候保持一致，这样一想确实更加合理。

ok,下面继续接着pipeline的部分，接着时去噪循环部分，flux没用做CFG,而是直接把指引强度作为了一个约束信息传入transformer中

```python
with self.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]

        # compute the previous noisy sample x_t -> x_t-1
        latents_dtype = latents.dtype
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
```

进入transformer之后进行噪声预测，预测出来的噪声用于去噪，通过调度器得到latents，最后pipeline会将latents解码，不过在解码之前会先做一次反图块化操作

```python
if output_type == "latent":
  image = latents

else:
  latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
  latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
  image = self.vae.decode(latents, return_dict=False)[0]
  image = self.image_processor.postprocess(image, output_type=output_type)
```

### transformer

再深入去噪网络看看它的构造。

```
self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

text_time_guidance_cls = (
    CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
)
self.time_text_embed = text_time_guidance_cls(
    embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
)

self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)
```

flux中的位置编码时EmbedND,是一种旋转式位置编码 (RoPE)，文本嵌入类有两种选择，如果设置了guidance_embeds的话就是CombinedTinestepGuidanceTextProjEmbeddings，不然就是CombinedTimestepTextProjEmbeddings。后面两个线性层，第一个是处理文本嵌入的，x_embedder这个线性层用来处理输入的通道数，而在sd3中，input_image会在pos_embed过一个下采样两倍的卷积层，进行图块化和修改通道数的操作，而在flux里面图块化操作在去噪网络外面，所以这里的x_embedder只需要进行修改通道数的操作。

直接进入forward看，先让输入过x_embedder，后续和sd3一样，求时刻编码，修改约束文本嵌入，不过后续又多了些操作：txt_ids和img_ids 进行了concat，得到了ids，作为RoPE的

```python
hidden_states = self.x_embedder(hidden_states)

timestep = timestep.to(hidden_states.dtype) * 1000
if guidance is not None:
    guidance = guidance.to(hidden_states.dtype) * 1000
else:
    guidance = None
temb = (
    self.time_text_embed(timestep, pooled_projections)
    if guidance is None
    else self.time_text_embed(timestep, guidance, pooled_projections)
)
encoder_hidden_states = self.context_embedder(encoder_hidden_states)

ids = torch.cat((txt_ids, img_ids), dim=0)
image_rotary_emb = self.pos_embed(ids)
```

此后图像信息和文本信息会反复输入进第一类transformer中

```python
encoder_hidden_states, hidden_states = block(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    temb=temb,
    image_rotary_emb=image_rotary_emb,
)
```

随后，过完第一类transformer之后，将图像信息和文本信息concat起来再输入进第二类transformer

```python
hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
for:
    hidden_states = block(
        hidden_states=hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
    )

hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
```

### pipeline_flux_inpaint

diffusers官方还推出了inpaint版本，这里主要对比一下，inpaint和普通的pipeline有什么区别,首先看的就是这个如何调用吧，最明显的区别在于inpaint比普通的pipeline多出的输入：image, mask_image

```python
# flux inpaint
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxInpaintPipeline
        >>> from diffusers.utils import load_image

        >>> pipe = FluxInpaintPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
        >>> img_url = "<https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png>"
        >>> mask_url = "<https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png>"
        >>> source = load_image(img_url)
        >>> mask = load_image(mask_url)
        >>> image = pipe(prompt=prompt, image=source, mask_image=mask).images[0]
        >>> image.save("flux_inpainting.png")
        ```
"""

# flux
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import FluxPipeline

        >>> pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> # Depending on the variant being used, the pipeline call will slightly vary.
        >>> # Refer to the pipeline documentation for more details.
        >>> image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        >>> image.save("flux.png")
        ```
"""
```

直接进入构造函数，大部分都是一模一样的，不过inapint里多了mask_processor,和处理image一样，使用的都是VaeImageProcessor

```python
self.mask_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            vae_latent_channels=self.vae.config.latent_channels,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )
```

再往下找，多出了_encode_vae_image和get_timesteps两个函数，都是copied from sd3 inpaint，下面详细说一下两个函数的作用

```
# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

    image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    return image_latents

# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
def get_timesteps(self, num_inference_steps, strength, device):
    # get the original timestep using init_timestep
    init_timestep = min(num_inference_steps * strength, num_inference_steps)

    t_start = int(max(num_inference_steps - init_timestep, 0))
    timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
    if hasattr(self.scheduler, "set_begin_index"):
        self.scheduler.set_begin_index(t_start * self.scheduler.order)

    return timesteps, num_inference_steps - t_start
```

**encode**vae_image没什么好说的，就是让图片过vae获得latent, get__timesteps里涉及到了strength这个参数，strength的作用在文档中这样描述：

```python
表示对参考图像进行的变换程度。必须在0到1之间。`image` 用作起点，`strength` 越高，添加的噪声
越多。噪声初始添加量决定噪声消除步骤的数量。当`strength` 为1时，添加的噪声达到最大值，噪声消
除过程将运行完整的`num_inference_steps` 步骤数。`strength` 值为1时，实际上忽略了`image`
```

先把这个放一下，在这个函数被调用的时候再做详解。

再往下也都是一样的部分，只是inpaint部分多了一个prepare_mask_latents，这个函数作用和prepare_latents大差不差，不过这个masked_image_latents也是经过了图块化操作的，不同于sd3inpaint中的mask的1个channel, 这里的mask是64个channel.

下面进入__call__函数，前面检查输入，准备各种变量，到了prepare timesteps这里多了下面这一行

```python
timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
```

这时候需要回头看一下前文提到的get_timesteps方法，按平常来讲timestep, num_inference_steps在上一步都已经得到了，那这一步的意义是？

还是一样直接用例子说明，假设num_inference_steps是20，strength=0.8，那么首先计算得到init_timestep=16, t_start=20-16=4,那么timesteps就直接选择从t_start开始了，也就是直接从4开始，保留后续的16个时间步，这时候返回的num_inference_steps就是20-4=16了

这是什么原理呢，其实上面已经说到了，这就是strength的用法，在DDIM中这个参数命名为denoise，它的作用其实就是选择一个合适的生成起点吧。这里可以从sd里面举一反三：

因为SD训练时采用DDPM其加噪的总步数为1000步，而重建流程本来也是采用Diffusion那种也是用DDPM即1000步，但是这样做太耗时了，故SD在重建过程采用的**DDIM算法**，假设其设定的重建步数为20，即**在1到1000步中，均匀采样20步**，即:

**[1,51,101,151,201,251,301,351,401,451,501,551,601,651,701,751,801,851,901,951]**

如设定denoise=0.6，则 **T = 0.6*20=12, 即index=12**，那么得到的对应DDPM的加噪步数为 t=**551**。

所以生成的起点就是t=551的时候。
