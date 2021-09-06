import logger



def PrintList(lst):
    if len(lst) < 15:
        logger.Log(lst)
    else:
        fst = [ str(l) for l in lst[0 : 5] ]
        lst = [ str(l) for l in lst[-3 : -1] ]
        logger.Log('[ {}, ... , {} ]'.format(', '.join(fst), ', '.join(lst)))


def PrintListOfLists(lst):
    if len(lst) < 10:
        for l in range(len(lst)):
            PrintList(lst[l])
    else:
        for l in range(3):
            PrintList(lst[l])
        logger.Log('    .')
        logger.Log('    .')
        logger.Log('    .')
        PrintList(lst[-1])



if __name__ == '__main__':
    print('This is a module!')

