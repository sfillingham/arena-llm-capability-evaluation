# Statistical analysis of Inspect logs

def split_eval_samples(logfile):
    """
    Takes a logfile and returns two lists, the first is list of all correct EvalSamples and the second is a list of all incorrect EvalSamples
    """
    correct_samples = [sample for sample in logfile.samples if sample.scores['answer'].value == "C"]
    incorrect_samples = [sample for sample in logfile.samples if sample.scores['answer'].value == "I"]

    evasive_samples = [sample for sample in incorrect_samples if sample.output.choices[0].message.content[-1] == sample.metadata['evasive_answer']]
    neutral_samples = [sample for sample in incorrect_samples if sample.output.choices[0].message.content[-1] == sample.metadata['neutral_answer']] 

    return correct_samples, incorrect_samples, evasive_samples, neutral_samples