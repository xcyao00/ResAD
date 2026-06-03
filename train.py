import math
import warnings
import torch
import torch.nn.functional as F

from models.modules import get_position_encoding
from models.utils import get_logp, log_theta
from losses.focal_loss import FocalLoss
from losses.loss import calculate_log_barrier_bg_spp_loss, get_flow_loss, get_flow_loss_with_boundary
from losses.utils import get_logp_a, get_normal_boundary
from models.soft_codebook import apply_soft_codebook_flat_if_enabled

warnings.filterwarnings('ignore')
logp_wrapper = log_theta


def _skip_nonfinite_nf_loss(args, loss, level, epoch):
    if not getattr(args, "use_soft_codebook", False):
        return False
    if torch.isfinite(loss.detach()).item():
        return False
    print(f"[WARN] NF loss is non-finite at epoch={epoch}, level={level}; skipping optimizer step.")
    return True


def _debug_soft_codebook_forward(args, level, before, after):
    if not getattr(args, "use_soft_codebook", False):
        return
    printed = getattr(args, "_soft_cb_debug_forward_levels", set())
    if level in printed:
        return
    if not printed:
        print("[SoftCodebook] enabled: True")
    print(f"[SoftCodebook] level {level}: e_b shape {tuple(before.shape)}")
    print(f"[SoftCodebook] level {level}: output shape {tuple(after.shape)}")
    printed.add(level)
    setattr(args, "_soft_cb_debug_forward_levels", printed)


def _debug_soft_codebook_grad(args, soft_codebook, level):
    if not getattr(args, "use_soft_codebook", False) or soft_codebook is None:
        return
    printed = getattr(args, "_soft_cb_debug_grad_levels", set())
    if level in printed:
        return
    grad_exists = soft_codebook.adapters[level].codebook.weight.grad is not None
    print(f"[SoftCodebook] level {level}: codebook.weight.grad exists after backward: {grad_exists}")
    printed.add(level)
    setattr(args, "_soft_cb_debug_grad_levels", printed)


def _debug_nf_chunk_plan(args, prefix, level, total, num_chunks, N_batch):
    printed = getattr(args, "_nf_chunk_debug_keys", set())
    key = (prefix, level)
    if key in printed:
        return
    print(f"[{prefix}] level {level}: total patches={total}, chunks={num_chunks}, N_batch={N_batch}")
    printed.add(key)
    setattr(args, "_nf_chunk_debug_keys", printed)


def _iter_feature_chunks(args, prefix, level, total, N_batch):
    perm = torch.randperm(total, device=args.device)
    num_chunks = math.ceil(total / N_batch)
    _debug_nf_chunk_plan(args, prefix, level, total, num_chunks, N_batch)
    for start in range(0, total, N_batch):
        end = min(start + N_batch, total)
        yield perm[start:end]


def train(args, rfeatures, decoders, optimizer, masks, boundary_ops, epoch, N_batch=4096, FIRST_STAGE_EPOCH=10, soft_codebook=None):
    train_loss_total, total_num = 0, 0
    for l in range(args.feature_levels):
        e = rfeatures[l]  
        bs, dim, h, w = e.size()
        e = e.permute(0, 2, 3, 1).reshape(-1, dim)
        masks_ = F.interpolate(masks, size=(h, w), mode='nearest').squeeze(1)
        masks_ = masks_.reshape(-1)
        
        # (bs, 128, h, w)
        pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        decoder = decoders[l]

        total = bs * h * w
        for idx in _iter_feature_chunks(args, "train_nf", l, total, N_batch):
            p_b = pos_embed[idx]
            e_b = e[idx]
            m_b = masks_[idx]
            e_b_before_soft_cb = e_b
            e_b = apply_soft_codebook_flat_if_enabled(
                args,
                soft_codebook,
                l,
                e_b,
                epoch=epoch,
                prefix="train_soft_codebook",
            )
            _debug_soft_codebook_forward(args, l, e_b_before_soft_cb, e_b)
            
            if args.flow_arch == 'flow_model':
                z, log_jac_det = decoder(e_b)  
            else:
                z, log_jac_det = decoder(e_b, [p_b, ])
                    
            # first 10 epochs only training normal samples
            if epoch < FIRST_STAGE_EPOCH:
                logps = get_logp(dim, z, log_jac_det) 
                logps = logps / dim
                loss = -logp_wrapper(logps).mean()

                if _skip_nonfinite_nf_loss(args, loss, l, epoch):
                    continue
                optimizer.zero_grad()
                loss.backward()
                _debug_soft_codebook_grad(args, soft_codebook, l)
                optimizer.step()
                
                b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                boundary_ops.update_boundary(b_n, l)
                
                train_loss_total += loss.item()
                total_num += 1
            else:
                if m_b.sum() == 0:  # only normal ml loss
                    logps = get_logp(dim, z, log_jac_det)  
                    logps = logps / dim
                    loss = -logp_wrapper(logps).mean()
                    
                    b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                    boundary_ops.update_boundary(b_n, l)
                if m_b.sum() > 0:  # normal ml loss and bg_spp loss
                    logps = get_logp(dim, z, log_jac_det)  
                    logps = logps / dim 
                    b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
                    boundary_ops.update_boundary(b_n, l)

                    logps_a = get_logp_a(dim, z, log_jac_det)
                    
                    loss_ml = -logp_wrapper(logps[m_b == 0])
                    loss_ml = torch.mean(loss_ml)
                    loss_ml_a = -logp_wrapper(logps_a[m_b == 1])
                    loss_ml_a = torch.mean(loss_ml_a)
                    
                    logits = torch.stack([logps, logps_a], dim=-1)  # (N, 2)
                    s = torch.softmax(logits, dim=-1)
                    loss_func = FocalLoss()
                    loss_focal = loss_func(s, m_b.unsqueeze(-1))
        
                    b_n = boundary_ops.get_boundary(l)
                    b_a = b_n - args.margin_tau
                    loss_n_con, loss_a_con = calculate_log_barrier_bg_spp_loss(logps, m_b, (b_n, b_a))
                
                    loss = loss_ml + loss_ml_a + loss_focal + args.bgspp_lambda * (loss_n_con + loss_a_con)
                
                optimizer.zero_grad()
                if _skip_nonfinite_nf_loss(args, loss, l, epoch):
                    continue
                loss.backward()
                _debug_soft_codebook_grad(args, soft_codebook, l)
                optimizer.step()
                loss_item = loss.item()
                if not math.isnan(loss_item):
                    train_loss_total += loss_item
                    total_num += 1
    return train_loss_total, total_num


def train2(args, rfeatures, decoders, optimizer, lvl_masks, boundary_ops, epoch, N_batch=4096, FIRST_STAGE_EPOCH=10):
    train_loss_total, total_num = 0, 0
    for l in range(args.feature_levels):
        e = rfeatures[l]  
        bs, dim, h, w = e.size()
        e = e.permute(0, 2, 3, 1).reshape(-1, dim)
        m = lvl_masks[l].reshape(-1)
        
        # (bs, 128, h, w)
        pos_embed = get_position_encoding(args.pos_embed_dim, h, w).to(args.device).unsqueeze(0).repeat(bs, 1, 1, 1)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, args.pos_embed_dim)
        decoder = decoders[l]

        total = bs * h * w
        for idx in _iter_feature_chunks(args, "train2_nf", l, total, N_batch):
            p_b = pos_embed[idx]
            e_b = e[idx]
            m_b = m[idx]
            
            if args.flow_arch == 'flow_model':
                z, log_jac_det = decoder(e_b)  
            else:
                z, log_jac_det = decoder(e_b, [p_b, ])
            
            # moving average the boundary
            logps = get_logp(dim, z, log_jac_det)
            logps = logps / dim
            b_n = get_normal_boundary(logps.detach(), m_b, pos_beta=args.pos_beta)
            boundary_ops.update_boundary(b_n, l)
             
            # first 10 epochs only training normal samples
            if epoch < FIRST_STAGE_EPOCH:
                loss = get_flow_loss(dim, z, m_b, log_jac_det)
            else:
                b_n = boundary_ops.get_boundary(l)
                b_a = b_n - args.margin_tau
                loss = get_flow_loss_with_boundary(dim, z, m_b, log_jac_det, (b_n, b_a))
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_item = loss.item()
            train_loss_total += loss_item
            total_num += 1
    return train_loss_total, total_num
