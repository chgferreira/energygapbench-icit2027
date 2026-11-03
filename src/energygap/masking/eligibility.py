import math


def evaluation_mask(observed_original, artificial_hidden, target_gap_mask, padding_mask):
    lengths={len(x) for x in (observed_original,artificial_hidden,target_gap_mask,padding_mask)}
    if len(lengths)!=1: raise ValueError('mask lengths differ')
    return tuple(o and h and t and not p for o,h,t,p in zip(observed_original,artificial_hidden,target_gap_mask,padding_mask))
