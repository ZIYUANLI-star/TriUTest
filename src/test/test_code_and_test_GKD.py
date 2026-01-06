def pysiphash(uint64):
    assert 0 <= uint64 < 1 << 64
    if uint64 > (1 << 63) - 1:
        int64 = uint64 - (1 << 64)
    else:
        int64 = uint64
    uint32 = (uint64 ^ uint64 >> 32) & 4294967295
    if uint32 > (1 << 31) - 1:
        int32 = uint32 - (1 << 32)
    else:
        int32 = uint32
    return int32, int64


