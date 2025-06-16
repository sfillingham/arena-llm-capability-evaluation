# Statistical analysis of Inspect logs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def split_eval_samples(logfile):
    """
    Takes a logfile and returns two lists, the first is list of all correct EvalSamples and the second is a list of all incorrect EvalSamples
    """
    correct_samples = [sample for sample in logfile.samples if sample.scores['answer'].value == "C"]
    incorrect_samples = [sample for sample in logfile.samples if sample.scores['answer'].value == "I"]

    evasive_samples = [sample for sample in incorrect_samples if sample.output.choices[0].message.content[-1] == sample.metadata['evasive_answer']]
    neutral_samples = [sample for sample in incorrect_samples if sample.output.choices[0].message.content[-1] == sample.metadata['neutral_answer']] 

    return correct_samples, incorrect_samples, evasive_samples, neutral_samples


def plot_question_bias_type_by_question_topic(df: pd.DataFrame):
    """
    Takes a dataframe that has eval output info and generates a figure
    """
    bias_counts = df.groupby(['question_topic', 'bias_type']).size().reset_index(name='count')
    bias_totals = df.groupby('question_topic').size().reset_index(name='total')
    bias_long = bias_counts.merge(bias_totals, on='question_topic')
    bias_long['fraction'] = bias_long['count'] / bias_long['total']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=bias_long, 
                x='question_topic', 
                y='fraction', 
                hue='bias_type',
                palette=['r', 'b'])

    plt.title(f"Fraction of Questions with each Bias Type by Question Topic")
    plt.xlabel('Question Topic', fontsize=12)
    plt.ylabel('Fraction of Questions in the Eval', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Bias Type')
    plt.tight_layout()
    plt.show()


def plot_answer_type_by_question_topic(df: pd.DataFrame, model: str):
    """
    Takes a dataframe that has eval output infor and generates a figure
    """
    bias_counts = df.groupby(['question_topic', 'answer']).size().reset_index(name='count')
    bias_totals = df.groupby('question_topic').size().reset_index(name='total')
    bias_long = bias_counts.merge(bias_totals, on='question_topic')
    bias_long['fraction'] = bias_long['count'] / bias_long['total']

    plt.figure(figsize=(12, 6))
    sns.barplot(data=bias_long, 
                x='question_topic', 
                y='fraction', 
                hue='answer',
                palette=['c', 'm', 'k'])

    plt.title(f"Fraction of Answer Types by Question Topic - {model}")
    plt.xlabel('Question Topic', fontsize=12)
    plt.ylabel('Fraction of Answers', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Answer Type')
    plt.tight_layout()
    plt.show()