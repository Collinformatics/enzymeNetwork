import os


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

# Input: Inactive Substrates
inExcludeMers = ['LQA', 'LQC', 'LQG', 'LQS']
inFileNamesInactive = ['substrates_Mpro2-I_S1_L001.json']

# Input: Outputs
inFileTag = 'Mpro2-Reg_R4-R6'
inFileTagInactive = 'Mpro2-Init'



# ========================================================================================
def processSubstrates(directory, fileNames, excludeSubstrates, saveTag):
    substrates = []
    for fileName in fileNames:
        pathLoad = os.path.join(directory, fileName)
