import os, json, requests, random, time, runpod

import torch
from PIL import Image
import numpy as np

import nodes
from nodes import NODE_CLASS_MAPPINGS
from comfy_extras import nodes_flux

DualCLIPLoader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
ControlNetLoader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()

FluxGuidance = nodes_flux.NODE_CLASS_MAPPINGS["FluxGuidance"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()
VAEDecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
VAEEncode = NODE_CLASS_MAPPINGS["VAEEncode"]()
ControlNetApplyAdvanced = NODE_CLASS_MAPPINGS["ControlNetApplyAdvanced"]()
ImageScaleBy = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
LoadImage = NODE_CLASS_MAPPINGS["LoadImage"]()

with torch.inference_mode():
    clip = DualCLIPLoader.load_clip("t5xxl_fp16.safetensors", "clip_l.safetensors", "flux")[0]
    unet = UNETLoader.load_unet("flux1-dev-fp8.safetensors", "fp8_e4m3fn")[0]
    vae = VAELoader.load_vae("ae.safetensors")[0]
    controlnet = ControlNetLoader.load_controlnet("controlnet.safetensors")[0]

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image=values['input_image_check']
    input_image=download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    positive_prompt = values['positive_prompt']
    seed = values['seed']
    upscale_method = values['upscale_method']
    scale_by = values['scale_by']
    strength = values['strength']
    steps = values['steps']
    cfg = values['cfg']
    guidance = values['guidance']
    sampler_name = values['sampler_name']
    scheduler = values['scheduler']

    if seed == 0:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    print(seed)

    negative_prompt = ""
    cond = nodes.CLIPTextEncode().encode(clip, positive_prompt)[0]
    cond = FluxGuidance.append(cond, guidance)[0]
    n_cond = nodes.CLIPTextEncode().encode(clip, negative_prompt)[0]
    input_image = LoadImage.load_image(input_image)[0]
    upscaled_image = ImageScaleBy.upscale(input_image, upscale_method, scale_by)[0]
    latent_upscaled_image = VAEEncode.encode(vae, upscaled_image)[0]
    positive, negative = ControlNetApplyAdvanced.apply_controlnet(positive=cond, negative=n_cond, control_net=controlnet, image=input_image, strength=strength, vae=vae, start_percent=0, end_percent=1)
    sample = nodes.common_ksampler(model=unet,
                                    seed=seed,
                                    steps=steps,
                                    cfg=cfg,
                                    sampler_name=sampler_name,
                                    scheduler=scheduler,
                                    positive=positive,
                                    negative=negative,
                                    latent=latent_upscaled_image,
                                    denoise=1.0)[0]
    decoded = VAEDecode.decode(vae, sample)[0].detach()
    Image.fromarray(np.array(decoded*255, dtype=np.uint8)[0]).save("/content/flux.1-dev-upscaler-tost.png")

    result = "/content/flux.1-dev-upscaler-tost.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})