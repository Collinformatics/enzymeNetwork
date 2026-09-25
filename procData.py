import os
import json
import sys


"""
    Convert json files obtained by COMET into lists that can be used for training a
    binary classifier

    This will process two sets
    1)  Active substrates:

        This set can consist of COMET filtered substrates, simple AA filtered sequences,
        or active substrates.


    2)  Inactive, or background:

        This set should consist of inactive substrates.
        To improve the utility of this set for training n-mer sequences that are
        characteristic of the active substrates can be removed from this set.
"""


# Input: Active Substrates
inFileNamesActive = [
    'fixedMotifSubs-SARS-CoV-2_Mᵖʳᵒ-Register_Q@R4-FinalSort-MinCounts_1.json',
    'fixedMotifSubs-SARS-CoV-2_Mᵖʳᵒ-Register_Q@R5-FinalSort-MinCounts_1.json',
    'fixedMotifSubs-SARS-CoV-2_Mᵖʳᵒ-Register_Q@R6-FinalSort-MinCounts_1.json'
]
inMinCountActive = False

# Input: Inactive Substrates
inExcludeMers = ['LQA', 'LQC', 'LQG', 'LQS']
inFileNamesInactive = ['substrates_Mpro2-I_S1_L001.json']
inMinCountInactive = 5

# Input: Save Outputs
inFileNameActive = 'Mpro2-Register_Q@R4-R6'
inFileNameInactive = 'Mpro2-Init'
inPathDir = 'Data/Train/' # Path to directory


# ========================================================================================
def processSubstrates(directory, fileNames, saveTag, setClass, excludeSeq, minCounts):
    print('================================== Convert File '
          '=================================')
    if excludeSeq:
        print(f'Exclude n-mers: {', '.join(excludeSeq)}')
    print(f'Minimum counts: {minCounts}\n')

    print(f'Loading substrates:')
    substrates = []
    for fileName in fileNames:
        pathLoad = os.path.join(directory, fileName)
        print(f'* {pathLoad}')
        with open(pathLoad, 'r') as f:
            subs = json.load(f)
            if excludeSeq:
                if minCounts:
                    for s, c in subs.items():
                        if s not in excludeSeq and c >= minCounts and s not in substrates:
                            substrates.append(s)
                else:
                    for s in subs.keys():
                        if s not in excludeSeq and s not in substrates:
                            substrates.append(s)
            else:
                if minCounts:
                    for s, c in subs.items():
                        if c >= minCounts and s not in substrates:
                            substrates.append(s)
                else:
                    for s in subs.keys():
                        if s not in substrates:
                            substrates.append(s)

    pathSave = os.path.join(directory, f'substrates_{setClass}_{saveTag}.txt')
    if substrates:
        print(f'\nSaving Substrates:\n* {pathSave}\n\n')
        with open(pathSave, 'w') as f:
            f.write('\n'.join(substrates))
    else:
        print(f'\nNo substrates were found\n\n')


processSubstrates(
    directory=inPathDir, fileNames=inFileNamesActive, saveTag=inFileNameActive,
    setClass='Pos', excludeSeq=[], minCounts=inMinCountActive
)

processSubstrates(
    directory=inPathDir, fileNames=inFileNamesInactive, saveTag=inFileNameInactive,
    setClass='Neg', excludeSeq=inExcludeMers, minCounts=inMinCountInactive
)

