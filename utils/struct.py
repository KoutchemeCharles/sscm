def get_dict_subset(d, keys, reject=False):
    
    if reject:
        cond = lambda k: k not in keys
    else:
        cond = lambda k: k in keys
        
    return {k: v for k, v in d.items() if cond(k)}