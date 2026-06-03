import argparse
import csv
import glob
import os
import re
import warnings

import numpy as np
import torch
import timm
from torch.utils.data import DataLoader

from classes import CAPSULES_TO_CAPSULES
from classes import MVTEC_TO_BRATS, MVTEC_TO_BTAD, MVTEC_TO_MPDD, MVTEC_TO_MVTEC
from classes import MVTEC_TO_MVTEC3D, MVTEC_TO_MVTECLOCO, MVTEC_TO_VISA
from classes import VISA_TO_MVTEC, VISA_TO_VISA
from datasets.brats import BRATS
from datasets.btad import BTAD
from datasets.capsules import CAPSULES
from datasets.mpdd import MPDD
from datasets.mvtec import MVTEC
from datasets.mvtec_3d import MVTEC3D
from datasets.mvtec_loco import MVTECLOCO
from datasets.visa import VISA
from models.fc_flow import load_flow_model
from models.modules import MultiScaleConv
from models.soft_codebook import SoftCodebookAdapterList
from models.vq import MultiScaleVQ
from raw_vqops import print_raw_vqops_config
from validate_wav import load_soft_codebook_state, validate


warnings.filterwarnings("ignore")

TOTAL_SHOT = 4
DINOV2_BACKBONES = ("dinov2_vits14", "dinov2_vitb14")
DINOV2_FEATURE_MODES = ("final_projected", "intermediate_fixed_projected")
SETTINGS = {
    "visa_to_mvtec": VISA_TO_MVTEC,
    "mvtec_to_visa": MVTEC_TO_VISA,
    "mvtec_to_btad": MVTEC_TO_BTAD,
    "mvtec_to_mvtec3d": MVTEC_TO_MVTEC3D,
    "mvtec_to_mpdd": MVTEC_TO_MPDD,
    "mvtec_to_mvtecloco": MVTEC_TO_MVTECLOCO,
    "mvtec_to_brats": MVTEC_TO_BRATS,
    "mvtec_to_mvtec": MVTEC_TO_MVTEC,
    "visa_to_visa": VISA_TO_VISA,
    "capsules_to_capsules": CAPSULES_TO_CAPSULES,
}
SCORE_TYPES = (
    ("scores1", "Logps"),
    ("scores2", "BScores"),
    ("scores", "Merged"),
)
CSV_COLUMNS = (
    "class_name",
    "score_type",
    "image_auc",
    "image_ap",
    "image_f1",
    "pixel_auc",
    "pixel_ap",
    "pixel_f1",
    "aupro",
)


def print_soft_codebook_config(args):
    print("[SoftCodebook] use_soft_codebook:", args.use_soft_codebook)
    print("[SoftCodebook] soft_cb_pos:", args.soft_cb_pos)
    print("[SoftCodebook] soft_cb_k:", args.soft_cb_k)
    print("[SoftCodebook] soft_cb_tau:", args.soft_cb_tau)
    print("[SoftCodebook] soft_cb_gamma:", args.soft_cb_gamma)
    print("[SoftCodebook] soft_cb_warmup_epochs:", args.soft_cb_warmup_epochs)
    print("[SoftCodebook] soft_cb_conf_gate:", args.soft_cb_conf_gate)
    print("[SoftCodebook] soft_cb_gate_threshold:", args.soft_cb_gate_threshold)
    print("[SoftCodebook] soft_cb_gate_temp:", args.soft_cb_gate_temp)


def resolve_checkpoint_file(args):
    if args.checkpoint_file:
        if not os.path.isfile(args.checkpoint_file):
            raise FileNotFoundError(f"checkpoint_file not found: {args.checkpoint_file}")
        return args.checkpoint_file

    if not args.checkpoint_path:
        raise ValueError("Set --checkpoint_file or --checkpoint_path.")
    checkpoint_files = sorted(glob.glob(os.path.join(args.checkpoint_path, "*.pth")))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pth files found in checkpoint_path: {args.checkpoint_path}")
    return checkpoint_files[-1]


def resolve_eval_epoch(args, checkpoint_file):
    if args.eval_epoch is not None:
        return args.eval_epoch
    match = re.search(r"epoch_(\d+)", os.path.basename(checkpoint_file))
    if match:
        return int(match.group(1))
    return None


def build_encoder(args):
    if args.backbone == "wide_resnet50_2":
        encoder = timm.create_model(
            "wide_resnet50_2",
            features_only=True,
            out_indices=(1, 2, 3),
            pretrained=True,
        ).eval()
        encoder = encoder.to(args.device)
        return encoder, encoder.feature_info.channels()

    if args.backbone == "tf_efficientnet_b6":
        encoder = timm.create_model(
            "tf_efficientnet_b6",
            features_only=True,
            out_indices=(1, 2, 3),
            pretrained=True,
        ).eval()
        encoder = encoder.to(args.device)
        return encoder, encoder.feature_info.channels()

    if args.backbone in DINOV2_BACKBONES:
        # TODO: Consider adding a DINOv2-specific normalization option and compare it with the existing w50 normalization.
        from models.dinov2_backbone import DINOv2BackboneWrapper, print_dinov2_config

        encoder = DINOv2BackboneWrapper(
            model_name=args.backbone,
            out_dims=(40, 72, 200),
            out_sizes=(56, 28, 14),
            freeze=True,
            feature_mode=args.dinov2_feature_mode,
            layers=args.dinov2_layers,
            proj_dim=args.dinov2_proj_dim,
        ).to(args.device)
        encoder.eval()
        print_dinov2_config(encoder, image_size=224)
        return encoder, encoder.feature_info.channels()

    raise ValueError(f"Unsupported backbone: {args.backbone}")


def build_modules(args, feat_dims):
    constraintor = MultiScaleConv(feat_dims).to(args.device)
    estimators = [load_flow_model(args, feat_dim).to(args.device) for feat_dim in feat_dims]

    soft_codebook = None
    if args.use_soft_codebook:
        soft_codebook = SoftCodebookAdapterList(
            feat_dims,
            num_embeddings=args.soft_cb_k,
            tau=args.soft_cb_tau,
            gamma=args.soft_cb_gamma,
            warmup_epochs=args.soft_cb_warmup_epochs,
            conf_gate=args.soft_cb_conf_gate,
            gate_threshold=args.soft_cb_gate_threshold,
            gate_temp=args.soft_cb_gate_temp,
        ).to(args.device)

    raw_vq_ops = None
    if args.use_raw_vqops:
        raw_vq_ops = MultiScaleVQ(num_embeddings=args.num_embeddings, channels=feat_dims).to(args.device)

    return constraintor, soft_codebook, raw_vq_ops, estimators


def load_checkpoint_states(args, checkpoint_file, constraintor, soft_codebook, raw_vq_ops, estimators):
    checkpoint = torch.load(checkpoint_file, map_location=args.device)

    if "constraintor" not in checkpoint:
        raise KeyError("checkpoint does not contain 'constraintor'.")
    if "estimators" not in checkpoint:
        raise KeyError("checkpoint does not contain 'estimators'.")

    constraintor.load_state_dict(checkpoint["constraintor"])
    if len(checkpoint["estimators"]) != len(estimators):
        raise ValueError(
            f"checkpoint has {len(checkpoint['estimators'])} estimators, "
            f"but args.feature_levels={len(estimators)}."
        )
    for estimator, state_dict in zip(estimators, checkpoint["estimators"]):
        estimator.load_state_dict(state_dict)

    load_soft_codebook_state(args, checkpoint, soft_codebook)

    if args.use_raw_vqops:
        if raw_vq_ops is None:
            raise ValueError("raw_vq_ops module was not initialized.")
        if "raw_vq_ops" in checkpoint:
            raw_vq_ops.load_state_dict(checkpoint["raw_vq_ops"], strict=False)
        else:
            print("[RawVQOps] checkpoint has no raw_vq_ops state; skipping load.")


def load_mc_reference_features(root_dir, class_names, device, num_shot=4):
    refs = {}
    for class_name in class_names:
        layer1_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, "layer1.npy"))).to(device)
        layer2_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, "layer2.npy"))).to(device)
        layer3_refs = torch.from_numpy(np.load(os.path.join(root_dir, class_name, "layer3.npy"))).to(device)
        k1 = (layer1_refs.shape[0] // TOTAL_SHOT) * num_shot
        k2 = (layer2_refs.shape[0] // TOTAL_SHOT) * num_shot
        k3 = (layer3_refs.shape[0] // TOTAL_SHOT) * num_shot
        refs[class_name] = (layer1_refs[:k1, :], layer2_refs[:k2, :], layer3_refs[:k3, :])
    return refs


def build_test_dataset(args, class_name):
    dataset_kwargs = dict(
        class_name=class_name,
        train=False,
        normalize="w50",
        img_size=224,
        crp_size=224,
        msk_size=224,
        msk_crp_size=224,
    )
    if args.classes == "capsules":
        return CAPSULES(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MVTEC.CLASS_NAMES:
        return MVTEC(args.test_dataset_dir, **dataset_kwargs)
    if class_name in VISA.CLASS_NAMES:
        return VISA(args.test_dataset_dir, **dataset_kwargs)
    if class_name in BTAD.CLASS_NAMES:
        return BTAD(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MVTEC3D.CLASS_NAMES:
        return MVTEC3D(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MPDD.CLASS_NAMES:
        return MPDD(args.test_dataset_dir, **dataset_kwargs)
    if class_name in MVTECLOCO.CLASS_NAMES:
        return MVTECLOCO(args.test_dataset_dir, **dataset_kwargs)
    if class_name in BRATS.CLASS_NAMES:
        return BRATS(args.test_dataset_dir, **dataset_kwargs)
    raise ValueError(f"Unrecognized class name: {class_name}")


def metric_row(class_name, score_type, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    return {
        "class_name": class_name,
        "score_type": score_type,
        "image_auc": image_auc,
        "image_ap": image_ap,
        "image_f1": image_f1,
        "pixel_auc": pixel_auc,
        "pixel_ap": pixel_ap,
        "pixel_f1": pixel_f1,
        "aupro": aupro,
    }


def print_metric_block(title, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    print(f"[{title}]")
    print(f"Image AUC | AP | F1: {image_auc:.3f} | {image_ap:.3f} | {image_f1:.3f}")
    print(f"Pixel AUC | AP | F1 | AUPRO: {pixel_auc:.3f} | {pixel_ap:.3f} | {pixel_f1:.3f} | {aupro:.3f}")


def print_average_line(label, values):
    image_auc, image_ap, image_f1, pixel_auc, pixel_ap, pixel_f1, aupro = values
    print(
        f"({label}) Average Image AUC | AP | F1: {image_auc:.3f} | {image_ap:.3f} | {image_f1:.3f}, "
        f"Average Pixel AUC | AP | F1 | AUPRO: {pixel_auc:.3f} | {pixel_ap:.3f} | "
        f"{pixel_f1:.3f} | {aupro:.3f}"
    )


def save_csv(path, rows):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] saved: {path}")


def main(args):
    if args.use_raw_vqops and args.use_soft_codebook:
        raise ValueError("Do not use raw VQOps and SoftCB together in this ablation.")
    if args.setting not in SETTINGS:
        raise ValueError(f"Dataset setting must be in {SETTINGS.keys()}, but got {args.setting}.")

    classes = SETTINGS[args.setting]
    checkpoint_file = resolve_checkpoint_file(args)
    eval_epoch = resolve_eval_epoch(args, checkpoint_file)

    print("[Eval] checkpoint_file:", checkpoint_file)
    print("[Eval] eval_epoch:", eval_epoch)
    print("[Eval] setting:", args.setting)
    print("[Eval] classes:", classes["unseen"])
    print("[Eval] backbone:", args.backbone)
    print("[Residual] residual_mode:", args.residual_mode)
    print("[Matching] match_mode:", args.match_mode)
    print("[Matching] match_topk:", args.match_topk)
    print("[Matching] match_tau:", args.match_tau)
    print("[Matching] match_chunk_size:", args.match_chunk_size)
    print_raw_vqops_config(args)
    print_soft_codebook_config(args)

    encoder, feat_dims = build_encoder(args)
    constraintor, soft_codebook, raw_vq_ops, estimators = build_modules(args, feat_dims)
    load_checkpoint_states(args, checkpoint_file, constraintor, soft_codebook, raw_vq_ops, estimators)

    encoder.eval()
    constraintor.eval()
    if soft_codebook is not None:
        soft_codebook.eval()
    if raw_vq_ops is not None:
        raw_vq_ops.eval()
    for estimator in estimators:
        estimator.eval()

    test_ref_features = load_mc_reference_features(
        args.test_ref_feature_dir,
        classes["unseen"],
        args.device,
        args.num_ref_shot,
    )

    results_by_type = {label: [] for _, label in SCORE_TYPES}
    csv_rows = []
    for class_name in classes["unseen"]:
        test_dataset = build_test_dataset(args, class_name)
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        metrics = validate(
            args,
            encoder,
            constraintor,
            soft_codebook,
            raw_vq_ops,
            estimators,
            test_loader,
            test_ref_features[class_name],
            args.device,
            class_name,
            epoch=eval_epoch,
        )

        print(f"\nClass: {class_name}")
        for key, label in SCORE_TYPES:
            values = metrics[key]
            print_metric_block(label, values)
            results_by_type[label].append(values)
            csv_rows.append(metric_row(class_name, label, values))

    print("\nAverages")
    for _, label in SCORE_TYPES:
        values = np.mean(np.asarray(results_by_type[label]), axis=0)
        print_average_line(label, values)
        csv_rows.append(metric_row("Average", label, values))

    if args.save_csv:
        save_csv(args.save_csv, csv_rows)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--classes", type=str, default="none")
    parser.add_argument("--test_dataset_dir", type=str, required=True)
    parser.add_argument("--test_ref_feature_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--backbone", type=str, default="wide_resnet50_2")
    parser.add_argument("--dinov2_feature_mode", type=str, default="final_projected", choices=DINOV2_FEATURE_MODES)
    parser.add_argument("--dinov2_layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--dinov2_proj_dim", type=int, default=256)
    parser.add_argument("--num_ref_shot", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--eval_epoch", type=int, default=None)

    parser.add_argument("--flow_arch", type=str, default="conditional_flow_model")
    parser.add_argument("--feature_levels", default=3, type=int)
    parser.add_argument("--coupling_layers", type=int, default=10)
    parser.add_argument("--clamp_alpha", type=float, default=1.9)
    parser.add_argument("--pos_embed_dim", type=int, default=256)

    parser.add_argument("--residual_mode", type=str, default="sq", choices=["sq", "abs", "signed"])
    parser.add_argument("--match_mode", type=str, default="hard", choices=["hard", "soft_topk"])
    parser.add_argument("--match_topk", type=int, default=5)
    parser.add_argument("--match_tau", type=float, default=0.05)
    parser.add_argument("--match_chunk_size", type=int, default=8192)

    parser.add_argument("--use_wav", action="store_true")
    parser.add_argument("--wav_on", type=str, default="residual", choices=["residual", "feature"])
    parser.add_argument("--wave", type=str, default="haar", choices=["haar"])
    parser.add_argument("--hf_weight", type=float, default=1.0)
    parser.add_argument("--wav_mode", type=str, default="ll_hf", choices=["ll_hf", "ll_only", "skip_ll", "skip_hf", "hf_gate"])
    parser.add_argument("--feature_wav_mode", type=str, default="ll_only", choices=["ll_only", "hf_only", "ll_hf", "skip_ll", "skip_hf"])
    parser.add_argument("--ll_skip_alpha", type=float, default=0.5)
    parser.add_argument("--hf_gate_beta", type=float, default=1.0)
    parser.add_argument("--hf_skip_alpha", type=float, default=0.75)
    parser.add_argument("--wav_hf_normalize", action="store_true")

    parser.add_argument("--use_soft_codebook", action="store_true")
    parser.add_argument("--soft_cb_pos", type=str, default="post_constraintor", choices=["post_constraintor"])
    parser.add_argument("--soft_cb_k", type=int, default=512)
    parser.add_argument("--soft_cb_tau", type=float, default=0.2)
    parser.add_argument("--soft_cb_gamma", type=float, default=0.03)
    parser.add_argument("--soft_cb_warmup_epochs", type=int, default=5)
    parser.add_argument("--soft_cb_conf_gate", action="store_true")
    parser.add_argument("--soft_cb_gate_threshold", type=float, default=0.0)
    parser.add_argument("--soft_cb_gate_temp", type=float, default=0.05)

    parser.add_argument("--use_raw_vqops", action="store_true")
    parser.add_argument("--raw_vq_pos", type=str, default="post_constraintor", choices=["pre_constraintor", "post_constraintor"])
    parser.add_argument("--raw_vq_debug", action="store_true")
    parser.add_argument("--num_embeddings", type=int, default=1536)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
