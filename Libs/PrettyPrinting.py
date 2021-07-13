import Libs.Logger



def PrintList(lst):
    if len(lst) < 15:
        Libs.Logger.Log(lst)
    else:
        fst = [ str(l) for l in lst[0 : 5] ]
        lst = [ str(l) for l in lst[-3 : -1] ]
        Libs.Logger.Log('[ {}, ... , {} ]'.format(', '.join(fst), ', '.join(lst)))


def PrintListOfLists(lst):
    if len(lst) < 10:
        for l in range(len(lst)):
            PrintList(lst[l])
    else:
        for l in range(3):
            PrintList(lst[l])
        Libs.Logger.Log('    .')
        Libs.Logger.Log('    .')
        Libs.Logger.Log('    .')
        PrintList(lst[-1])



if __name__ == '__main__':
    print('This is a module!')

