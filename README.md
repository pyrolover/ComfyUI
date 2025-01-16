# ComfyUI
ComfyUI - Ksampler - 'UNetModel' object has no attribute 'x_embedder'

How can I fix this issue? 

# ComfyUI Error Report
## Error Details
- **Node ID:** 22
- **Node Type:** KSampler
- **Exception Type:** AttributeError
- **Exception Message:** 'UNetModel' object has no attribute 'x_embedder'
## Stack Trace
```
  File "/data/app/execution.py", line 323, in execute
    output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)

  File "/data/app/execution.py", line 198, in get_output_data
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)

  File "/data/app/execution.py", line 169, in _map_node_over_list
    process_inputs(input_dict, i)

  File "/data/app/execution.py", line 158, in process_inputs
    results.append(getattr(obj, func)(**inputs))

  File "/data/app/nodes.py", line 1465, in sample
    return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)

  File "/data/app/nodes.py", line 1432, in common_ksampler
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,

  File "/data/app/comfy/sample.py", line 43, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)

  File "/data/app/comfy/samplers.py", line 1020, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

  File "/data/app/comfy/samplers.py", line 918, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

  File "/data/app/comfy/samplers.py", line 904, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

  File "/data/app/comfy/patcher_extension.py", line 110, in execute
    return self.original(*args, **kwargs)

  File "/data/app/comfy/samplers.py", line 873, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)

  File "/data/app/comfy/samplers.py", line 848, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

  File "/data/app/comfy/samplers.py", line 773, in process_conds
    pre_run_control(model, conds[k])

  File "/data/app/comfy/samplers.py", line 615, in pre_run_control
    x['control'].pre_run(model, percent_to_timestep_function)

  File "/data/app/comfy/controlnet.py", line 475, in pre_run
    missing, unexpected = self.control_model.x_embedder.load_state_dict(model.diffusion_model.x_embedder.state_dict(), strict=False)

  File "/data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

```
## System Information
- **ComfyUI Version:** v0.3.6-4-g57e8bf6
- **Arguments:** /data/app/main.py --listen --port 30000
- **OS:** posix
- **Python Version:** 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0]
- **Embedded Python:** false
- **PyTorch Version:** 2.4.0+cu121
## Devices

- **Name:** cuda:0 NVIDIA L40S : cudaMallocAsync
  - **Type:** cuda
  - **VRAM Total:** 47810936832
  - **VRAM Free:** 36162673584
  - **Torch VRAM Total:** 11106516992
  - **Torch VRAM Free:** 19045296

## Logs
```
2025-01-16T20:49:46.825398 - [START] Security scan2025-01-16T20:49:46.825478 - 
2025-01-16T20:49:51.725764 - [DONE] Security scan2025-01-16T20:49:51.725795 - 
2025-01-16T20:49:52.176464 - ## ComfyUI-Manager: installing dependencies done.2025-01-16T20:49:52.176503 - 
2025-01-16T20:49:52.176522 - ** ComfyUI startup time:2025-01-16T20:49:52.176536 -  2025-01-16T20:49:52.176552 - 2025-01-16 20:49:52.1765112025-01-16T20:49:52.176565 - 
2025-01-16T20:49:52.176581 - ** Platform:2025-01-16T20:49:52.176594 -  2025-01-16T20:49:52.176607 - Linux2025-01-16T20:49:52.176619 - 
2025-01-16T20:49:52.176631 - ** Python version:2025-01-16T20:49:52.176643 -  2025-01-16T20:49:52.176655 - 3.10.14 (main, Mar 21 2024, 16:24:04) [GCC 11.2.0]2025-01-16T20:49:52.176667 - 
2025-01-16T20:49:52.176680 - ** Python executable:2025-01-16T20:49:52.176691 -  2025-01-16T20:49:52.176703 - /data/miniconda3/pkgs/comfyui/bin/python2025-01-16T20:49:52.176714 - 
2025-01-16T20:49:52.176726 - ** ComfyUI Path:2025-01-16T20:49:52.176737 -  2025-01-16T20:49:52.176748 - /data/app2025-01-16T20:49:52.176760 - 
2025-01-16T20:49:52.176785 - ** Log path:2025-01-16T20:49:52.176798 -  2025-01-16T20:49:52.176810 - /data/app/comfyui.log2025-01-16T20:49:52.176822 - 
2025-01-16T20:49:52.361057 - 
Prestartup times for custom nodes:2025-01-16T20:49:52.361099 - 
2025-01-16T20:49:52.361129 -    0.1 seconds:2025-01-16T20:49:52.361148 -  2025-01-16T20:49:52.361174 - /data/app/custom_nodes/rgthree-comfy2025-01-16T20:49:52.361190 - 
2025-01-16T20:49:52.361206 -    5.6 seconds:2025-01-16T20:49:52.361219 -  2025-01-16T20:49:52.361232 - /data/app/custom_nodes/ComfyUI-Manager2025-01-16T20:49:52.361255 - 
2025-01-16T20:49:52.361276 - 
2025-01-16T20:49:57.759305 - Total VRAM 45596 MB, total RAM 31632 MB
2025-01-16T20:49:57.759490 - pytorch version: 2.4.0+cu121
2025-01-16T20:50:01.768329 - /data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:211: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_fwd")
2025-01-16T20:50:01.847962 - /data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/xformers/ops/fmha/flash.py:344: FutureWarning: `torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.
  @torch.library.impl_abstract("xformers_flash::flash_bwd")
2025-01-16T20:50:02.128958 - xformers version: 0.0.27.post2
2025-01-16T20:50:02.129102 - Set vram state to: NORMAL_VRAM
2025-01-16T20:50:02.129189 - Device: cuda:0 NVIDIA L40S : cudaMallocAsync
2025-01-16T20:50:02.881207 - Using xformers cross attention
2025-01-16T20:50:06.763340 - [Prompt Server] web root: /data/app/web
2025-01-16T20:50:08.042481 - /data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/kornia/feature/lightglue.py:44: FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated. Please use `torch.amp.custom_fwd(args..., device_type='cuda')` instead.
  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
2025-01-16T20:50:10.689525 - ### Loading: ComfyUI-Manager (V2.50.2)2025-01-16T20:50:10.689625 - 
2025-01-16T20:50:10.787623 - ### ComfyUI Revision: 2880 [57e8bf6a] | Released on '2024-12-02'2025-01-16T20:50:10.787652 - 
2025-01-16T20:50:10.883424 - [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/github-stats.json2025-01-16T20:50:10.883457 - 
2025-01-16T20:50:10.944418 - Version of aiofiles: Not found2025-01-16T20:50:10.944448 - 
2025-01-16T20:50:10.944479 - ComfyUI-N-Sidebar is loading...2025-01-16T20:50:10.944496 - 
2025-01-16T20:50:11.030931 - [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/model-list.json2025-01-16T20:50:11.030961 - 
2025-01-16T20:50:11.035274 - [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/alter-list.json2025-01-16T20:50:11.035303 - 
2025-01-16T20:50:11.057673 - [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json2025-01-16T20:50:11.057706 - 
2025-01-16T20:50:11.085435 - [ComfyUI-Manager] default cache updated: https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json2025-01-16T20:50:11.085480 - 
2025-01-16T20:50:11.156291 - Patching UNetModel.forward2025-01-16T20:50:11.156320 - 
2025-01-16T20:50:11.156359 - UNetModel.forward has been successfully patched.2025-01-16T20:50:11.156377 - 
2025-01-16T20:50:11.601094 - [36;20m[comfyui_controlnet_aux] | INFO -> Using ckpts path: /data/app/custom_nodes/comfyui_controlnet_aux/ckpts[0m
2025-01-16T20:50:11.601175 - [36;20m[comfyui_controlnet_aux] | INFO -> Using symlinks: False[0m
2025-01-16T20:50:11.601230 - [36;20m[comfyui_controlnet_aux] | INFO -> Using ort providers: ['CUDAExecutionProvider', 'DirectMLExecutionProvider', 'OpenVINOExecutionProvider', 'ROCMExecutionProvider', 'CPUExecutionProvider', 'CoreMLExecutionProvider'][0m
2025-01-16T20:50:16.214196 - DWPose: Onnxruntime with acceleration providers detected2025-01-16T20:50:16.214237 - 
2025-01-16T20:50:21.180904 - 
2025-01-16T20:50:21.180965 - [92m[rgthree-comfy] Loaded 42 exciting nodes. 🎉[00m2025-01-16T20:50:21.180985 - 
2025-01-16T20:50:21.181002 - 
2025-01-16T20:50:21.181368 - 
Import times for custom nodes:
2025-01-16T20:50:21.181448 -    0.1 seconds: /data/app/custom_nodes/FreeU_Advanced
2025-01-16T20:50:21.181499 -    0.2 seconds: /data/app/custom_nodes/ComfyUI-N-Sidebar
2025-01-16T20:50:21.181555 -    0.3 seconds: /data/app/custom_nodes/ComfyUI-Manager
2025-01-16T20:50:21.181601 -    0.7 seconds: /data/app/custom_nodes/ComfyUI-Custom-Scripts
2025-01-16T20:50:21.181665 -    3.2 seconds: /data/app/custom_nodes/rgthree-comfy
2025-01-16T20:50:21.181716 -    6.8 seconds: /data/app/custom_nodes/comfyui_controlnet_aux
2025-01-16T20:50:21.181762 - 
2025-01-16T20:50:21.193875 - Starting server

2025-01-16T20:50:21.194278 - To see the GUI go to: http://0.0.0.0:30000
2025-01-16T20:50:21.194437 - To see the GUI go to: http://[::]:30000
2025-01-16T20:50:32.268207 - FETCH DATA from: /data/app/custom_nodes/ComfyUI-Manager/extension-node-map.json2025-01-16T20:50:32.268496 - 2025-01-16T20:50:32.274898 -  [DONE]2025-01-16T20:50:32.274927 - 
2025-01-16T20:52:46.475360 - got prompt
2025-01-16T20:52:46.511967 - Failed to validate prompt for output 9:
2025-01-16T20:52:46.512036 - * CheckpointLoaderSimple 4:
2025-01-16T20:52:46.512076 -   - Value not in list: ckpt_name: 'v1-5-pruned-emaonly.ckpt' not in (list of length 61)
2025-01-16T20:52:46.512110 - Output will be ignored
2025-01-16T20:52:47.194918 - model_path is /data/app/custom_nodes/comfyui_controlnet_aux/ckpts/LiheYoung/Depth-Anything/checkpoints/depth_anything_vitl14.pth2025-01-16T20:52:47.194974 - 
2025-01-16T20:52:48.151067 - using MLP layer as FFN
2025-01-16T20:52:51.705238 - /data/app/custom_nodes/comfyui_controlnet_aux/src/custom_controlnet_aux/depth_anything/__init__.py:42: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(model_path, map_location="cpu"))
2025-01-16T20:53:46.354007 - model weight dtype torch.float16, manual cast: None
2025-01-16T20:53:46.374943 - model_type EPS
2025-01-16T20:53:50.123947 - Using xformers attention in VAE
2025-01-16T20:53:50.127047 - Using xformers attention in VAE
2025-01-16T20:53:50.829404 - /data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
2025-01-16T20:53:50.879406 - Requested to load SDXLClipModel
2025-01-16T20:53:50.891473 - loaded completely 9.5367431640625e+25 1560.802734375 True
2025-01-16T20:53:51.928287 - loaded straight to GPU
2025-01-16T20:53:51.928434 - Requested to load SDXL
2025-01-16T20:53:51.958576 - loaded completely 9.5367431640625e+25 4897.0483474731445 True
2025-01-16T20:54:06.570269 - WARNING: no VAE provided to the controlnet apply node when this controlnet requires one.
2025-01-16T20:54:06.593884 - Requested to load ControlNetEmbedder
2025-01-16T20:54:07.263778 - loaded completely 9.5367431640625e+25 4107.81298828125 True
2025-01-16T20:54:07.352023 - !!! Exception during processing !!! 'UNetModel' object has no attribute 'x_embedder'
2025-01-16T20:54:07.360081 - Traceback (most recent call last):
  File "/data/app/execution.py", line 323, in execute
    output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
  File "/data/app/execution.py", line 198, in get_output_data
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
  File "/data/app/execution.py", line 169, in _map_node_over_list
    process_inputs(input_dict, i)
  File "/data/app/execution.py", line 158, in process_inputs
    results.append(getattr(obj, func)(**inputs))
  File "/data/app/nodes.py", line 1465, in sample
    return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
  File "/data/app/nodes.py", line 1432, in common_ksampler
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
  File "/data/app/comfy/sample.py", line 43, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "/data/app/comfy/samplers.py", line 1020, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "/data/app/comfy/samplers.py", line 918, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/samplers.py", line 904, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/patcher_extension.py", line 110, in execute
    return self.original(*args, **kwargs)
  File "/data/app/comfy/samplers.py", line 873, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/samplers.py", line 848, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)
  File "/data/app/comfy/samplers.py", line 773, in process_conds
    pre_run_control(model, conds[k])
  File "/data/app/comfy/samplers.py", line 615, in pre_run_control
    x['control'].pre_run(model, percent_to_timestep_function)
  File "/data/app/comfy/controlnet.py", line 475, in pre_run
    missing, unexpected = self.control_model.x_embedder.load_state_dict(model.diffusion_model.x_embedder.state_dict(), strict=False)
  File "/data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'UNetModel' object has no attribute 'x_embedder'

2025-01-16T20:54:07.360918 - Prompt executed in 80.85 seconds
2025-01-16T20:57:41.534058 - got prompt
2025-01-16T20:57:41.573418 - Failed to validate prompt for output 9:
2025-01-16T20:57:41.573490 - * CheckpointLoaderSimple 4:
2025-01-16T20:57:41.573530 -   - Value not in list: ckpt_name: 'v1-5-pruned-emaonly.ckpt' not in (list of length 61)
2025-01-16T20:57:41.573567 - Output will be ignored
2025-01-16T20:57:41.604069 - !!! Exception during processing !!! 'UNetModel' object has no attribute 'x_embedder'
2025-01-16T20:57:41.606308 - Traceback (most recent call last):
  File "/data/app/execution.py", line 323, in execute
    output_data, output_ui, has_subgraph = get_output_data(obj, input_data_all, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
  File "/data/app/execution.py", line 198, in get_output_data
    return_values = _map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True, execution_block_cb=execution_block_cb, pre_execute_cb=pre_execute_cb)
  File "/data/app/execution.py", line 169, in _map_node_over_list
    process_inputs(input_dict, i)
  File "/data/app/execution.py", line 158, in process_inputs
    results.append(getattr(obj, func)(**inputs))
  File "/data/app/nodes.py", line 1465, in sample
    return common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
  File "/data/app/nodes.py", line 1432, in common_ksampler
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
  File "/data/app/comfy/sample.py", line 43, in sample
    samples = sampler.sample(noise, positive, negative, cfg=cfg, latent_image=latent_image, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, denoise_mask=noise_mask, sigmas=sigmas, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "/data/app/comfy/samplers.py", line 1020, in sample
    return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
  File "/data/app/comfy/samplers.py", line 918, in sample
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/samplers.py", line 904, in sample
    output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/patcher_extension.py", line 110, in execute
    return self.original(*args, **kwargs)
  File "/data/app/comfy/samplers.py", line 873, in outer_sample
    output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
  File "/data/app/comfy/samplers.py", line 848, in inner_sample
    self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)
  File "/data/app/comfy/samplers.py", line 773, in process_conds
    pre_run_control(model, conds[k])
  File "/data/app/comfy/samplers.py", line 615, in pre_run_control
    x['control'].pre_run(model, percent_to_timestep_function)
  File "/data/app/comfy/controlnet.py", line 475, in pre_run
    missing, unexpected = self.control_model.x_embedder.load_state_dict(model.diffusion_model.x_embedder.state_dict(), strict=False)
  File "/data/miniconda3/pkgs/comfyui/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1729, in __getattr__
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
AttributeError: 'UNetModel' object has no attribute 'x_embedder'

2025-01-16T20:57:41.606658 - Prompt executed in 0.03 seconds

```
## Attached Workflow
Please make sure that workflow does not contain any sensitive information such as API keys or passwords.
```
{"last_node_id":22,"last_link_id":24,"nodes":[{"id":7,"type":"CLIPTextEncode","pos":[413,389],"size":[425.27801513671875,180.6060791015625],"flags":{},"order":7,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":5}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[6],"slot_index":0}],"properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":["text, watermark"]},{"id":6,"type":"CLIPTextEncode","pos":[415,186],"size":[422.84503173828125,164.31304931640625],"flags":{},"order":6,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":3}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[4],"slot_index":0}],"properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":["beautiful scenery nature glass bottle landscape, , purple galaxy bottle,"]},{"id":5,"type":"EmptyLatentImage","pos":[473,609],"size":[315,106],"flags":{},"order":0,"mode":0,"inputs":[],"outputs":[{"name":"LATENT","type":"LATENT","links":[2],"slot_index":0}],"properties":{"Node name for S&R":"EmptyLatentImage"},"widgets_values":[512,512,1]},{"id":3,"type":"KSampler","pos":[863,186],"size":[315,262],"flags":{},"order":11,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":1},{"name":"positive","type":"CONDITIONING","link":4},{"name":"negative","type":"CONDITIONING","link":6},{"name":"latent_image","type":"LATENT","link":2}],"outputs":[{"name":"LATENT","type":"LATENT","links":[7],"slot_index":0}],"properties":{"Node name for S&R":"KSampler"},"widgets_values":[299894578475706,"randomize",20,8,"euler","normal",1]},{"id":8,"type":"VAEDecode","pos":[1209,188],"size":[210,46],"flags":{},"order":14,"mode":0,"inputs":[{"name":"samples","type":"LATENT","link":7},{"name":"vae","type":"VAE","link":8}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[9],"slot_index":0}],"properties":{"Node name for S&R":"VAEDecode"},"widgets_values":[]},{"id":9,"type":"SaveImage","pos":[1451,189],"size":[210,58],"flags":{},"order":16,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":9}],"outputs":[],"properties":{},"widgets_values":["ComfyUI"]},{"id":4,"type":"CheckpointLoaderSimple","pos":[26,474],"size":[315,98],"flags":{},"order":1,"mode":0,"inputs":[],"outputs":[{"name":"MODEL","type":"MODEL","links":[1],"slot_index":0},{"name":"CLIP","type":"CLIP","links":[3,5],"slot_index":1},{"name":"VAE","type":"VAE","links":[8],"slot_index":2}],"properties":{"Node name for S&R":"CheckpointLoaderSimple"},"widgets_values":["v1-5-pruned-emaonly.ckpt"]},{"id":10,"type":"PreviewImage","pos":[9904.689453125,2649.57080078125],"size":[210,246],"flags":{},"order":18,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":10,"localized_name":"images"}],"outputs":[],"title":"PreviewImage","properties":{"Node name for S&R":"PreviewImage"}},{"id":11,"type":"SaveImage","pos":[9873.45703125,3019.2158203125],"size":[315,270],"flags":{},"order":19,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":11,"localized_name":"images"}],"outputs":[],"title":"SaveImage","properties":{},"widgets_values":["ComfyUI"]},{"id":12,"type":"VAEDecode","pos":[9575.6025390625,2818.339111328125],"size":[210,46],"flags":{},"order":17,"mode":0,"inputs":[{"name":"samples","type":"LATENT","link":12,"localized_name":"samples"},{"name":"vae","type":"VAE","link":13,"localized_name":"vae"}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[10,11],"slot_index":0,"localized_name":"IMAGE"}],"title":"VAEDecode","properties":{"Node name for S&R":"VAEDecode"}},{"id":13,"type":"CLIPTextEncode","pos":[8538.61328125,2884.630126953125],"size":[400,200],"flags":{},"order":8,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":14,"localized_name":"clip"}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[23],"slot_index":0,"localized_name":"CONDITIONING"}],"title":"CLIPTextEncode","properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":["bad quality, bad skin, bad hair"],"color":"#322","bgcolor":"#533"},{"id":14,"type":"PreviewImage","pos":[9831.1767578125,3684.84814453125],"size":[210,246],"flags":{},"order":12,"mode":0,"inputs":[{"name":"images","type":"IMAGE","link":15,"localized_name":"images"}],"outputs":[],"title":"PreviewImage","properties":{"Node name for S&R":"PreviewImage"}},{"id":15,"type":"ControlNetApply","pos":[9347.876953125,3408.17431640625],"size":[317.4000244140625,98],"flags":{},"order":13,"mode":0,"inputs":[{"name":"conditioning","type":"CONDITIONING","link":16,"localized_name":"conditioning"},{"name":"control_net","type":"CONTROL_NET","link":17,"localized_name":"control_net"},{"name":"image","type":"IMAGE","link":18,"localized_name":"image"}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[22],"slot_index":0,"localized_name":"CONDITIONING"}],"title":"ControlNetApply","properties":{"Node name for S&R":"ControlNetApply"},"widgets_values":[0.3]},{"id":16,"type":"CheckpointLoaderSimple","pos":[8037.26806640625,2755.44921875],"size":[315,98],"flags":{},"order":2,"mode":0,"inputs":[],"outputs":[{"name":"MODEL","type":"MODEL","links":[21],"localized_name":"MODEL"},{"name":"CLIP","type":"CLIP","links":[14,19],"slot_index":1,"localized_name":"CLIP"},{"name":"VAE","type":"VAE","links":[13],"slot_index":2,"localized_name":"VAE"}],"title":"CheckpointLoaderSimple","properties":{"Node name for S&R":"CheckpointLoaderSimple"},"widgets_values":["realvisxlV50_v50LightningBakedvae.safetensors"]},{"id":17,"type":"ControlNetLoader","pos":[8900.3642578125,3419.005615234375],"size":[315,58],"flags":{},"order":3,"mode":0,"inputs":[],"outputs":[{"name":"CONTROL_NET","type":"CONTROL_NET","links":[17],"slot_index":0,"localized_name":"CONTROL_NET"}],"title":"ControlNetLoader","properties":{"Node name for S&R":"ControlNetLoader"},"widgets_values":["sd3.5_large_controlnet_depth.safetensors"]},{"id":18,"type":"LoadImage","pos":[8867.7900390625,3608.40771484375],"size":[315,314],"flags":{},"order":4,"mode":0,"inputs":[],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[20],"localized_name":"IMAGE"},{"name":"MASK","type":"MASK","links":null,"localized_name":"MASK"}],"title":"LoadImage","properties":{"Node name for S&R":"LoadImage"},"widgets_values":["090ff6be.png","image"]},{"id":19,"type":"CLIPTextEncode","pos":[8544.4091796875,2609.906005859375],"size":[400,200],"flags":{},"order":9,"mode":0,"inputs":[{"name":"clip","type":"CLIP","link":19,"localized_name":"clip"}],"outputs":[{"name":"CONDITIONING","type":"CONDITIONING","links":[16],"slot_index":0,"localized_name":"CONDITIONING"}],"title":"CLIPTextEncode","properties":{"Node name for S&R":"CLIPTextEncode"},"widgets_values":[""],"color":"#232","bgcolor":"#353"},{"id":20,"type":"DepthAnythingPreprocessor","pos":[9360.2177734375,3672.16064453125],"size":[315,82],"flags":{},"order":10,"mode":0,"inputs":[{"name":"image","type":"IMAGE","link":20,"localized_name":"image"}],"outputs":[{"name":"IMAGE","type":"IMAGE","links":[15,18],"slot_index":0,"localized_name":"IMAGE"}],"title":"DepthAnythingPreprocessor","properties":{"Node name for S&R":"DepthAnythingPreprocessor"},"widgets_values":["depth_anything_vitl14.pth",1088]},{"id":21,"type":"EmptyLatentImage","pos":[8583.6513671875,3167.576904296875],"size":[315,106],"flags":{},"order":5,"mode":0,"inputs":[],"outputs":[{"name":"LATENT","type":"LATENT","links":[24],"slot_index":0,"localized_name":"LATENT"}],"title":"EmptyLatentImage","properties":{"Node name for S&R":"EmptyLatentImage"},"widgets_values":[512,1080,1]},{"id":22,"type":"KSampler","pos":[9123.3271484375,2716.892578125],"size":[315,262],"flags":{},"order":15,"mode":0,"inputs":[{"name":"model","type":"MODEL","link":21,"localized_name":"model"},{"name":"positive","type":"CONDITIONING","link":22,"localized_name":"positive"},{"name":"negative","type":"CONDITIONING","link":23,"localized_name":"negative"},{"name":"latent_image","type":"LATENT","link":24,"localized_name":"latent_image"}],"outputs":[{"name":"LATENT","type":"LATENT","links":[12],"slot_index":0,"localized_name":"LATENT"}],"properties":{"Node name for S&R":"KSampler"},"widgets_values":[884926058759516,"randomize",6,1.5,"dpmpp_sde","karras",1]}],"links":[[1,4,0,3,0,"MODEL"],[2,5,0,3,3,"LATENT"],[3,4,1,6,0,"CLIP"],[4,6,0,3,1,"CONDITIONING"],[5,4,1,7,0,"CLIP"],[6,7,0,3,2,"CONDITIONING"],[7,3,0,8,0,"LATENT"],[8,4,2,8,1,"VAE"],[9,8,0,9,0,"IMAGE"],[10,12,0,10,0,"IMAGE"],[11,12,0,11,0,"IMAGE"],[12,22,0,12,0,"LATENT"],[13,16,2,12,1,"VAE"],[14,16,1,13,0,"CLIP"],[15,20,0,14,0,"IMAGE"],[16,19,0,15,0,"CONDITIONING"],[17,17,0,15,1,"CONTROL_NET"],[18,20,0,15,2,"IMAGE"],[19,16,1,19,0,"CLIP"],[20,18,0,20,0,"IMAGE"],[21,16,0,22,0,"MODEL"],[22,15,0,22,1,"CONDITIONING"],[23,13,0,22,2,"CONDITIONING"],[24,21,0,22,3,"LATENT"]],"groups":[],"config":{},"extra":{"ds":{"scale":0.41772481694156605,"offset":[-7069.192835184832,-2332.3031306272915]}},"version":0.4}
```

## Additional Context
(Please add any additional context or steps to reproduce the error here)
