import torch
import torch.nn.functional as F

def _log_softmax(logits):
    return F.log_softmax(logits, dim=-1)

def jsd_loss(student_logits, teacher_logits, alpha: float = 0.9, mask=None):
    """
    Jensen-Shannon divergence between student and teacher distributions.
    Requires both full logits (same vocab). M = alpha*T + (1-alpha)*S
    Return mean over valid tokens in mask.
    """
    s_logprob = _log_softmax(student_logits)
    t_logprob = _log_softmax(teacher_logits)

    s_prob = s_logprob.exp()
    t_prob = t_logprob.exp()
    m_prob = alpha * t_prob + (1 - alpha) * s_prob
    m_logprob = torch.log(m_prob + 1e-8)

    kl_t_m = (t_prob * (t_logprob - m_logprob)).sum(-1)
    kl_s_m = (s_prob * (s_logprob - m_logprob)).sum(-1)
    jsd = alpha * kl_t_m + (1 - alpha) * kl_s_m
    if mask is not None:
        jsd = (jsd * mask).sum() / (mask.sum() + 1e-6)
    else:
        jsd = jsd.mean()
    return jsd

def forward_kl_on_logits(student_logits, teacher_logits, mask=None):
    """KL(teacher || student): teacher-to-student forward KL as defined in the paper."""
    s_logprob = _log_softmax(student_logits)
    t_prob = _log_softmax(teacher_logits).exp()
    fkl = (t_prob * (t_prob.log() - s_logprob)).sum(-1)  # KL(T||S)
    if mask is not None:
        fkl = (fkl * mask).sum() / (mask.sum() + 1e-6)
    else:
        fkl = fkl.mean()
    return fkl

def forward_kl_on_chosen(student_logp_chosen, teacher_logp_chosen, mask=None):
    """
    Approximate KL(T||S) when only chosen-token logprobs are available from the teacher.
    This is equivalent to cross-entropy on the chosen token with teacher soft weight.
    """
    # negative log-likelihood difference
    r = -(teacher_logp_chosen - student_logp_chosen)  # encourage student to match teacher prob
    if mask is not None:
        r = (r * mask).sum() / (mask.sum() + 1e-6)
    else:
        r = r.mean()
    return r
