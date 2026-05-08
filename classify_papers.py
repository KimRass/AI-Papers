#!/usr/bin/env python3
"""
논문 자동 분류 스크립트 - 방안 2 기반
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# 분류 규칙: 키워드 기반
CLASSIFICATION_RULES = {
    'generative': {
        'diffusion': [
            'diffusion', 'ddpm', 'ddim', 'stable_diffusion', 'sdxl',
            'ldm', 'latent_diffusion', 'score', 'denoising', 'cdm',
            'consistency_model', 'edm', 'glide', 'autoregressive_modeling'
        ],
        'gan': [
            'gan', 'generative_adversarial', 'vqgan', 'stylegan',
            'vq_vae', 'vqvae', 'adversarial', 'precision_and_recall_metric',
            'perceptual_loss'
        ],
        'text_to_image': [
            'dall_e', 'dalle', 'text_to_image', 'imagen', 'emu',
            'photogenic', 'caption'
        ],
        'image_editing': [
            'controlnet', 'inpaint', 'edit', 'try_on', 'diffuse_to_choose',
            'contrastive_learning_for_unpaired', 'fix_the_noise', 'head_blending'
        ]
    },

    'vision_language': {
        'vlm': [
            'clip', 'align', 'blip', 'coca', 'flamingo', 'florence',
            'vision_language', 'visual_language', 'catlip', 'eva_clip',
            'groupvit', 'elevater', 'can_vision_language_models_count',
            'vision_language_pre_training'
        ],
        'mllm': [
            'llava', 'mllm', 'multimodal', 'qwen', 'vision_instruction',
            'glamm', 'pixel_grounding', 'set_of_mark', 'interleaved_modal',
            'visual_planning', 'multimodal_chain_of_thought'
        ],
        'captioning': [
            'caption', 'contrastive_caption'
        ]
    },

    'document_ai': {
        'layout': [
            'donut', 'vitlp', 'layoutlm', 'lilt', 'layout', 'document',
            'docvlm', 'docling', 'dolphin', 'anls', 'chart',
            'multi_stage_pipeline', 'handwritten', 'financial_forms'
        ],
        'text_spotting': [
            'text_spot', 'synthtext', 'synthtiger', 'unreal_text',
            'visd', 'east', 'craft', 'pan', 'deer', 'character_region',
            'scene_text', 'text_detection', 'text_recognition', 'cleval',
            'bridging_the_gap', 'fast_faster', 'arbitrarily_shaped',
            'synthetic_data_for_text', 'taco_textual', 'text_localisation'
        ],
        # ocr과 table 폴더는 이미 존재하므로 건드리지 않음
        # 추가 키워드는 아래 OTHER_CATEGORIES에서 처리
    },

    'image_video_enhancement': {
        'super_resolution': [
            'super_resolution', 'swinir', 'sr3', 'esrgan', 'real_esrgan',
            'liif', 'implicit', 'continuous', 'upsampling', 'upscale',
            'blind_super_resolution', 'diffbir', 'moesr', 'arbitrary_scale'
        ],
        'restoration': [
            'restoration', 'degrad', 'enhance', 'edvr', 'deformable_non_local'
        ],
        'inpainting': [
            'inpaint', 'lama', 'context_encoder', 'mask', 'mat_'
        ]
    },

    'video': {
        'generation': [
            'video_generation', 'videogpt', 'make_a_video', 'video_ldm',
            'stable_video', 'animatediff', 'easyanimate', 'cv_vae',
            'lvdm', 'fifo_diffusion'
        ],
        'processing': [
            'video_super', 'basicvsr', 'video_restoration', 'optical_flow',
            'raft', 'flownet'
        ]
    },

    'foundation': {
        'architectures': [
            'vision_transformer', 'vit', 'swin', 'transformer', 'convnet',
            'deformable_conv', 'efficientnet', 'resnet', 'rcnn', 'fast_rcnn',
            'image_transformer', 'rwkv', 'flash_attention', 'sparse_transformer',
            'dit_', 'all_are_worth', 'flashattention', 'high_performance_large_scale',
            'normalization', 'mixture_of_experts', 'train_short_test_long',
            'linear_biases'
        ],
        'self_supervised': [
            'beit', 'imagegpt', 'dino', 'self_supervised', 'masked',
            'mae', 'moco', 'simclr', 'byol', 'emerging_properties',
            'hubert', 'big_transfer', 'momentum_contrast'
        ],
        'segmentation': [
            'segment', 'sam_', 'mask', 'instance_segmentation',
            'semantic_segmentation', 'detection', 'vitdet', 'matting',
            'zim_zero_shot'
        ]
    }
}

# LLM 폴더 추가 분류 (기존 llm 폴더 내부 정리용)
LLM_SUBCATEGORIES = {
    'foundation_models': [
        'gpt', 'llama', 'bert', 'xlm', 't5', 'polyglot', 'language_model', 'language_models',
        'survey', 'comprehensive_overview', 'deberta', 'electra', 'distilbert',
        'bart', 'gpt_understands', 'few_shot_learners',
        'unsupervised_multitask', 'zero_shot_reasoners', 'neural_machine_translation',
        'opt_iml', 'phi_technical', 'scaling_instruction', 'self_consistency',
        'solar_open', 'cot_collection', 'prompt_repetition', 'unchecked_and_overlooked',
        'large_language_models', 'instruction_tuning', 'chain_of_thought', 'reasoning',
        '_llms.pdf', 'non_reasoning_llms'
    ],
    'calibration': [
        'calibration', 'confidence', 'uncertainty', 'reliable',
        'confidence_improves', 'aleatoric_to_epistemic', 'revisiting_uncertainty',
        'uncertainty_quantification'
    ],
    'hallucination': [
        'hallucination', 'truthful', 'factual'
    ]
}

# 루트 레벨에서 LLM으로 분류할 키워드
LLM_ROOT_KEYWORDS = [
    'large_language_model', 'language_model', 'gpt', 'bert', 'distilbert',
    'neural_machine_translation', 'opt_iml', 'phi_technical', 'solar_',
    'scaling_instruction', 'chain_of_thought', 'cot_collection',
    'self_consistency', 'few_shot', 'zero_shot_reasoner'
]

# 3D 및 기타
OTHER_CATEGORIES = {
    '3d_vision': [
        '3d', 'gaussian_splatting', 'nerf', 'dreamfusion', 'radiance'
    ],
    'training': [
        'training', 'optimizer', 'augment', 'batch', 'sgd', 'learning_rate',
        'autoaugment', 'mixup', 'cutmix', 'fixmatch', 'semi_supervised',
        'fine_tuning', 'lora', 'parameter_efficient', 'train_test',
        'meta_pseudo', 'image_quality_affects', 'image_representations',
        'equivariance', 'equivalence'
    ],
    'video_generation': [
        'modelscope_text_to_video', 'sora_a_review', 'accurate_generative_models_of_video'
    ],
    'table_recognition': [
        'image_based_table', 'optimized_table_tokenization', 'grits_grid',
        'sepformer'
    ],
    'ocr_extra': [
        'hunyuanocr'
    ]
}


class PaperClassifier:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.classification_map: Dict[str, Tuple[str, str]] = {}

    def classify_file(self, filename: str) -> Tuple[str, str]:
        """
        파일명을 분석하여 카테고리와 서브카테고리를 반환
        Returns: (category, subcategory) 또는 (None, None)
        """
        # 파일명을 소문자로 변환하고, 공백을 언더스코어로 변경
        filename_lower = filename.lower().replace(' ', '_').replace('-', '_')

        # LLM 폴더에 있는 파일 체크
        if '/llm/' in str(filename) or str(filename).startswith('llm/'):
            # Vision-Language 관련은 해당 카테고리로
            if 'vision' in filename_lower or 'visual' in filename_lower or 'vqa' in filename_lower:
                if 'multimodal' in filename_lower or 'mllm' in filename_lower or 'instruction_tuning' in filename_lower:
                    return ('vision_language', 'mllm')
                return ('vision_language', 'vlm')

            # LLM 서브카테고리 분류
            for subcat, keywords in LLM_SUBCATEGORIES.items():
                if any(kw in filename_lower for kw in keywords):
                    return ('llm', subcat)
            return ('llm', 'foundation_models')  # 기본값

        # OCR, TABLE, SR 폴더는 그대로 유지
        if '/ocr/' in str(filename) or str(filename).startswith('ocr/'):
            return ('document_ai', 'ocr')
        if '/table/' in str(filename) or str(filename).startswith('table/'):
            return ('document_ai', 'table')
        if '/sr/' in str(filename) or str(filename).startswith('sr/'):
            # BasicVSR은 video/processing으로
            if 'basicvsr' in filename_lower:
                return ('video', 'processing')
            return ('image_video_enhancement', 'super_resolution')

        # 루트 레벨 LLM 파일 체크 (우선순위 높음)
        if any(kw in filename_lower for kw in LLM_ROOT_KEYWORDS):
            # Calibration 체크
            if any(kw in filename_lower for kw in LLM_SUBCATEGORIES['calibration']):
                return ('llm', 'calibration')
            # Hallucination 체크
            if any(kw in filename_lower for kw in LLM_SUBCATEGORIES['hallucination']):
                return ('llm', 'hallucination')
            # 기본은 foundation_models
            return ('llm', 'foundation_models')

        # 3D 먼저 체크
        if any(kw in filename_lower for kw in OTHER_CATEGORIES['3d_vision']):
            return ('foundation', '3d_vision')

        # Video generation 특수 케이스
        if any(kw in filename_lower for kw in OTHER_CATEGORIES['video_generation']):
            return ('video', 'generation')

        # Table recognition 특수 케이스
        if any(kw in filename_lower for kw in OTHER_CATEGORIES['table_recognition']):
            return ('document_ai', 'table')

        # OCR 특수 케이스
        if any(kw in filename_lower for kw in OTHER_CATEGORIES['ocr_extra']):
            return ('document_ai', 'ocr')

        # 메인 카테고리 분류
        for category, subcategories in CLASSIFICATION_RULES.items():
            for subcat, keywords in subcategories.items():
                if any(kw in filename_lower for kw in keywords):
                    return (category, subcat)

        # Training 관련
        if any(kw in filename_lower for kw in OTHER_CATEGORIES['training']):
            return ('foundation', 'training')

        # 분류 실패
        return (None, None)

    def scan_and_classify(self):
        """모든 PDF 파일을 스캔하고 분류"""
        pdf_files = list(self.root_dir.rglob('*.pdf'))

        print(f"총 {len(pdf_files)}개 PDF 파일 발견\n")

        unclassified = []
        category_counts = {}

        for pdf_file in pdf_files:
            rel_path = pdf_file.relative_to(self.root_dir)
            category, subcategory = self.classify_file(str(rel_path))

            if category:
                self.classification_map[str(rel_path)] = (category, subcategory)
                key = f"{category}/{subcategory}"
                category_counts[key] = category_counts.get(key, 0) + 1
            else:
                unclassified.append(str(rel_path))

        # 통계 출력
        print("=" * 60)
        print("분류 결과 통계:")
        print("=" * 60)
        for key, count in sorted(category_counts.items()):
            print(f"{key:50} {count:3}개")

        print(f"\n{'미분류':50} {len(unclassified):3}개")

        if unclassified:
            print("\n" + "=" * 60)
            print("미분류 파일 목록:")
            print("=" * 60)
            for f in sorted(unclassified):
                print(f"  - {f}")

        return len(unclassified)

    def create_folder_structure(self):
        """새 폴더 구조 생성"""
        folders_to_create = []

        # generative
        folders_to_create.extend([
            'generative/diffusion',
            'generative/gan',
            'generative/text_to_image',
            'generative/image_editing'
        ])

        # vision_language
        folders_to_create.extend([
            'vision_language/vlm',
            'vision_language/mllm',
            'vision_language/captioning'
        ])

        # document_ai (ocr, table은 이미 존재)
        folders_to_create.extend([
            'document_ai/layout',
            'document_ai/text_spotting'
        ])

        # image_video_enhancement
        folders_to_create.extend([
            'image_video_enhancement/super_resolution',
            'image_video_enhancement/restoration',
            'image_video_enhancement/inpainting'
        ])

        # video
        folders_to_create.extend([
            'video/generation',
            'video/processing'
        ])

        # foundation
        folders_to_create.extend([
            'foundation/architectures',
            'foundation/self_supervised',
            'foundation/segmentation',
            'foundation/3d_vision',
            'foundation/training'
        ])

        # llm 서브카테고리
        folders_to_create.extend([
            'llm/foundation_models',
            'llm/calibration',
            'llm/hallucination'
        ])

        print("\n" + "=" * 60)
        print("폴더 구조 생성 중...")
        print("=" * 60)

        for folder in folders_to_create:
            folder_path = self.root_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {folder}")

    def move_files(self, dry_run=True):
        """파일들을 분류된 폴더로 이동"""
        print("\n" + "=" * 60)
        if dry_run:
            print("DRY RUN MODE - 실제로 파일을 이동하지 않습니다")
        else:
            print("파일 이동 중...")
        print("=" * 60)

        moved_count = 0

        for rel_path, (category, subcategory) in sorted(self.classification_map.items()):
            src_path = self.root_dir / rel_path

            # 이미 올바른 위치에 있는지 체크
            expected_prefix = f"{category}/{subcategory}/"
            if str(rel_path).startswith(expected_prefix):
                continue

            # OCR, TABLE 폴더의 파일들을 document_ai로 이동
            if str(rel_path).startswith('ocr/'):
                dest_path = self.root_dir / 'document_ai' / 'ocr' / src_path.name
            elif str(rel_path).startswith('table/'):
                dest_path = self.root_dir / 'document_ai' / 'table' / src_path.name
            elif str(rel_path).startswith('sr/'):
                # SR 폴더 처리
                if 'basicvsr' in src_path.name.lower():
                    dest_path = self.root_dir / 'video' / 'processing' / src_path.name
                else:
                    dest_path = self.root_dir / 'image_video_enhancement' / 'super_resolution' / src_path.name
            else:
                dest_path = self.root_dir / category / subcategory / src_path.name

            # 같은 이름의 파일이 이미 있는지 체크
            if dest_path.exists() and dest_path != src_path:
                print(f"⚠️  충돌: {dest_path} 이미 존재함")
                continue

            print(f"{'→ ' if dry_run else '✓ '} {rel_path}")
            print(f"   → {dest_path.relative_to(self.root_dir)}")

            if not dry_run:
                shutil.move(str(src_path), str(dest_path))

            moved_count += 1

        print(f"\n총 {moved_count}개 파일 {'이동 예정' if dry_run else '이동 완료'}")

    def cleanup_empty_folders(self):
        """빈 폴더 정리"""
        print("\n" + "=" * 60)
        print("빈 폴더 정리 중...")
        print("=" * 60)

        # ocr, table, sr 폴더 삭제
        for folder_name in ['ocr', 'table', 'sr']:
            folder_path = self.root_dir / folder_name
            if folder_path.exists() and folder_path.is_dir():
                if not list(folder_path.glob('*')):
                    folder_path.rmdir()
                    print(f"✓ 빈 폴더 삭제: {folder_name}/")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='AI 논문 자동 분류 도구')
    parser.add_argument('--root', default='/Users/eric/workspace/AI-Papers',
                        help='프로젝트 루트 디렉토리')
    parser.add_argument('--execute', action='store_true',
                        help='실제로 파일 이동 (기본값: dry-run)')

    args = parser.parse_args()

    classifier = PaperClassifier(args.root)

    # 1. 스캔 및 분류
    unclassified_count = classifier.scan_and_classify()

    # 2. 폴더 구조 생성
    classifier.create_folder_structure()

    # 3. 파일 이동 (dry-run 또는 실행)
    classifier.move_files(dry_run=not args.execute)

    if args.execute:
        # 4. 빈 폴더 정리
        classifier.cleanup_empty_folders()
        print("\n✅ 분류 완료!")
    else:
        print("\n⚠️  DRY RUN 모드입니다. 실제 이동하려면 --execute 옵션을 사용하세요:")
        print(f"   python classify_papers.py --execute")

    if unclassified_count > 0:
        print(f"\n⚠️  {unclassified_count}개 파일이 미분류되었습니다. 수동 분류가 필요합니다.")


if __name__ == '__main__':
    main()
