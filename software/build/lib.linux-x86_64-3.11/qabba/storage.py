def compute_storage1(centers, len_strings, num_init=1, bits_for_len=32, bits_for_inc=32, bits_for_ts=32):
    """Compute storage need for ABBA representation"""
    size_centers = centers.shape[0]*bits_for_len + centers.shape[0]*bits_for_inc
    size_strings = 8 * len_strings
    return size_centers + size_strings + bits_for_ts*num_init

def compute_storage2(centers, len_strings, num_init=1, bits_for_len=8, bits_for_inc=16, bits_for_sz=32, bits_for_ts=32):
    """Compute storage need for QABBA representation"""
    size_centers = centers.shape[0]*bits_for_len + centers.shape[0]*bits_for_inc
    size_strings = 8 * len_strings
    return size_centers + size_strings + bits_for_ts*num_init + 2*bits_for_sz
