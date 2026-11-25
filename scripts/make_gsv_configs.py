import json, os, yaml
from pathlib import Path
PROJECT_ROOT = Path(os.environ.get('VOICE_CLONE_ROOT', Path(__file__).resolve().parents[1]))
REPO = Path(os.environ.get('GPT_SOVITS_REPO', PROJECT_ROOT / 'third_party' / 'GPT-SoVITS'))
EXP = 'dongxuelian'
S2_G = f'{REPO}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth'
S2_D = f'{REPO}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2D2333k.pth'
S1 = f'{REPO}/GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt'
exp_dir = f'{REPO}/logs/{EXP}'
os.makedirs(f'{REPO}/TEMP', exist_ok=True)
os.makedirs(exp_dir, exist_ok=True)

with open(f'{REPO}/GPT_SoVITS/configs/s2.json') as f:
    s2 = json.load(f)
s2['train'].update({
    'batch_size': 8, 'epochs': 12, 'text_low_lr_rate': 0.4,
    'pretrained_s2G': S2_G, 'pretrained_s2D': S2_D,
    'if_save_latest': True, 'if_save_every_weights': True,
    'save_every_epoch': 4, 'gpu_numbers': '0', 'grad_ckpt': False, 'lora_rank': 32,
})
s2['model']['version'] = 'v2'
s2['data']['exp_dir'] = s2['s2_ckpt_dir'] = exp_dir
s2['save_weight_dir'] = f'{REPO}/SoVITS_weights_v2'
s2['name'] = EXP
s2['version'] = 'v2'
with open(f'{REPO}/TEMP/tmp_s2.json', 'w') as f:
    json.dump(s2, f)

with open(f'{REPO}/GPT_SoVITS/configs/s1longer-v2.yaml') as f:
    s1 = yaml.safe_load(f)
s1['train'].update({
    'batch_size': 8, 'epochs': 15, 'precision': '16-mixed',
    'save_every_n_epoch': 5, 'if_save_every_weights': True,
    'if_save_latest': True, 'if_dpo': False,
    'half_weights_save_dir': f'{REPO}/GPT_weights_v2', 'exp_name': EXP,
})
s1['pretrained_s1'] = S1
s1['train_semantic_path'] = f'{exp_dir}/6-name2semantic.tsv'
s1['train_phoneme_path'] = f'{exp_dir}/2-name2text.txt'
s1['output_dir'] = f'{exp_dir}/logs_s1_v2'
with open(f'{REPO}/TEMP/tmp_s1.yaml', 'w') as f:
    yaml.dump(s1, f, default_flow_style=False)
print('wrote', f'{REPO}/TEMP/tmp_s2.json', f'{REPO}/TEMP/tmp_s1.yaml')
