def SumTotals(dictTotal):
    totalVal = 0.0
    for keyVal in dictTotal.keys():
        for keyVal2 in dictTotal[keyVal].keys():
            totalVal += dictTotal[keyVal][keyVal2]

    return round(totalVal, 2)


def test():
    assert SumTotals({'a': {'b': 1.0}, 'c': {'d': 7.0}}) == 8.0

