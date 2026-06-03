import torch


def print_raw_vqops_config(args):
    print("[RawVQOps] use_raw_vqops:", args.use_raw_vqops)
    print("[RawVQOps] raw_vq_pos:", args.raw_vq_pos)


def train_raw_vqops_if_enabled(args, raw_vq_ops, optimizer_vq, features, lvl_masks):
    if not getattr(args, "use_raw_vqops", False) or raw_vq_ops is None:
        return None
    vq_features = [feature.detach() for feature in features]
    loss_vq = raw_vq_ops(vq_features, lvl_masks, train=True)
    optimizer_vq.zero_grad()
    loss_vq.backward()
    optimizer_vq.step()
    return loss_vq


def apply_raw_vqops_if_enabled(args, raw_vq_ops, features, prefix="raw_vqops", loss_vq=None):
    if not getattr(args, "use_raw_vqops", False) or raw_vq_ops is None:
        return features
    input_features = [feature.detach() for feature in features]
    with torch.no_grad():
        output_features = list(raw_vq_ops(input_features, train=False))
    _debug_raw_vqops_if_enabled(args, prefix, input_features, output_features, loss_vq=loss_vq)
    return output_features


def _debug_raw_vqops_if_enabled(args, prefix, input_features, output_features, loss_vq=None):
    if not getattr(args, "raw_vq_debug", False):
        return
    printed = getattr(args, "_raw_vq_debug_keys", set())
    loss_value = None if loss_vq is None else float(loss_vq.detach().cpu())
    for level, (x_in, x_out) in enumerate(zip(input_features, output_features)):
        key = (prefix, level)
        if key in printed:
            continue
        has_nan = torch.isnan(x_out).any().item()
        has_inf = torch.isinf(x_out).any().item()
        print(f"[RawVQOps] {prefix} level {level}: input shape {tuple(x_in.shape)}")
        print(f"[RawVQOps] {prefix} level {level}: output shape {tuple(x_out.shape)}")
        print(f"[RawVQOps] {prefix} level {level}: output mean/std {x_out.mean().item():.6f}/{x_out.std().item():.6f}")
        print(f"[RawVQOps] {prefix} level {level}: has_nan={has_nan}, has_inf={has_inf}")
        if loss_value is not None:
            print(f"[RawVQOps] {prefix} level {level}: vq_loss={loss_value:.6f}")
        printed.add(key)
    setattr(args, "_raw_vq_debug_keys", printed)
