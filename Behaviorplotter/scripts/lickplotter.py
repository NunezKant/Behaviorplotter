from tkinter import Tk   
from tkinter.filedialog import askopenfilename
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import mat73
from datetime import datetime, timedelta

sns.set_context("paper")
plt.style.use('bmh')

def get_dataframes(data):
    """
    Get the dataframes from the data

    Parameters
    ----------
    data : dict
        Data dictionary.

    Returns
    -------
    lick_df : dataframe
        Dataframe with the lick data.
    reward_info: dataframe
        Dataframe with the reward information.
    """
    licks = data["Licks"]

    lick= pd.DataFrame(
        licks, columns=["Trial", "Position", "Alpha", "Rewarded", "ts", "start"]
    )
    lick["Datetime"] = pd.to_datetime(
        lick["ts"].apply(
            lambda x: datetime.fromordinal(int(x))
            + timedelta(days=x % 1)
            - timedelta(days=366)
        )
    )
    lick = lick.drop(["ts"], axis=1)
    total_trials = int(lick["Trial"].max())
    if len(data.keys()) > 5:
        lick["Category"] = np.nan
        categories, _ = get_trial_categories(
            data["TrialRewardStrct"], data["TrialNewTextureStrct"]
        )[:total_trials]
        for trial in lick["Trial"].unique():
            lick.loc[
                np.where(lick["Trial"] == int(trial))[0], "Category"
            ] = categories[int(trial - 1)]
    reward = pd.DataFrame(
        data["RewardInfo"][:total_trials, :], columns=["Position", "Datetime"]
    )
    rewardr = reward.copy()
    lickr = lick.copy()
    rewardr.loc[:,"Position"] *= 10
    lickr.loc[:,"Position"] *= 10
    return lickr, rewardr

def get_trial_categories(rewarded_trial_structure, new_trial_structure):
    """
    Compute the trial categories for the new trial structure

    Parameters
    ----------
    rewarded_trial_structure : array
        vector of the rewarded trials.
    new_trial_structure : array
        vector with new exemplar trials.

    Returns
    -------
    trial_categories : list
        List of the trial categories.
    trial_counts : dict
        Dictionary with the trial categories counts.

    """
    rewarded_trial_structure = np.array(rewarded_trial_structure)
    new_trial_structure = np.array(new_trial_structure)
    trial_categories = [None] * len(rewarded_trial_structure)
    rewarded_new_counter = 0
    rewarded_counter = 0
    non_rewarded_counter = 0
    non_rewarded_new_counter = 0

    for idx in range(new_trial_structure.shape[0]):
        if np.logical_and(rewarded_trial_structure[idx], new_trial_structure[idx]):
            trial_categories[idx] = "rewarded new"
            rewarded_new_counter += 1
        elif np.logical_and(
            rewarded_trial_structure[idx], np.logical_not(new_trial_structure[idx])
        ):
            trial_categories[idx] = "rewarded"
            rewarded_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]), new_trial_structure[idx]
        ):
            trial_categories[idx] = "non rewarded new"
            non_rewarded_new_counter += 1
        elif np.logical_and(
            np.logical_not(rewarded_trial_structure[idx]),
            np.logical_not(new_trial_structure[idx]),
        ):
            trial_categories[idx] = "non rewarded"
            non_rewarded_counter += 1

        trial_counts = {
            "rewarded new": rewarded_new_counter,
            "rewarded": rewarded_counter,
            "non rewarded new": non_rewarded_new_counter,
            "non rewarded": non_rewarded_counter,
        }

    return np.array(trial_categories), trial_counts

def get_trials_with_onelick(data, xlim=(45, 110)):
    """
    Create a dataframe with the trials with at least one lick

    Parameters
    ----------
    lick_df : dataframe
        Dataframe with the lick data.

    """
    lick_df, _ = get_dataframes(data)
    effective_trial = int(lick_df["Trial"].max())
    _, trial_counts = get_trial_categories(
        data["TrialRewardStrct"][:effective_trial],
        data["TrialNewTextureStrct"][:effective_trial],
    )
    lick_df_wo_startflag = lick_df[lick_df["start"] != 1]
    lick_pct = []
    lick_pct_dict = {}
    for key, item in trial_counts.items():
        if item>0:
            trials_with_licks = len(
                lick_df_wo_startflag.loc[
                    (lick_df_wo_startflag["Category"] == key)
                    & (lick_df_wo_startflag["Position"] > xlim[0])
                    & (lick_df_wo_startflag["Position"] < xlim[1])
                ]
                .groupby("Trial")
                .min()["Position"]
                .values
            )
            lick_pct.append(np.round(trials_with_licks / trial_counts[key], 2))
            lick_pct_dict[key] = lick_pct[-1]
    return lick_pct_dict

def lick_plot(
    data,
    first_lick=True,
    bin_num=60,
    fig_size=(12, 12),
):
    """
    Plot the lick data.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe containing the behavior data
    first_lick : bool
        If True, plot the first lick distribution over trials.
    fsize : tuple
        Figure size.

    Returns
    -------
    fig : figure
        Figure of lick data.
    """

    fig = plt.figure(figsize=fig_size)
    grid = plt.GridSpec(5, 6, hspace=0.2, wspace=0.3)
    main_ax = fig.add_subplot(grid[1:, :4])
    x_hist = fig.add_subplot(grid[:1, :4], sharex=main_ax)
    pct_axis = fig.add_subplot(grid[:2, 4:])

    lick, reward_df = get_dataframes(data)
    lick_pct_dict = get_trials_with_onelick(data, xlim=(45, 110))
    effective_trial = int(lick["Trial"].max())
    categories, trial_counts = get_trial_categories(
        data["TrialRewardStrct"][:effective_trial],
        data["TrialNewTextureStrct"][:effective_trial],
    )
    lick = lick[lick["start"] != 1]
    category_number = len(np.unique(categories))
    if category_number == 4:
        categories = ["rewarded", "non rewarded", "rewarded new", "non rewarded new"]
    elif category_number == 2:
        categories = ["rewarded", "non rewarded"]

    if first_lick:
        lick = lick.groupby("Trial").min().reset_index()

    for key, value in trial_counts.items():
        lick.loc[
                np.where(lick["Category"] == key)[0], "Weight"
            ] = value

    for category in categories:
        position = lick[lick["Category"] == category]["Position"]
        trial = lick[lick["Category"] == category]["Trial"]
        counts, bins = np.histogram(position, bin_num)

        main_ax.scatter(
            position,
            trial,
            marker="*",
            label=category,
            s=10,
        )

        #x_hist.hist(
        #    bins[:-1], alpha=0.5, weights=counts / trial_counts[category], bins=bins
        #)

        pct_axis.bar(category, lick_pct_dict[category]*100, alpha=0.5)
    main_ax.scatter(
        reward_df["Position"],
        reward_df.index + 1,
        s=25,
        marker="v",
        label="reward delivery",
        c="k",
        alpha=0.3,
    )

    sns.histplot(data=lick, hue='Category', x="Position", kde=True, weights=1/lick["Weight"], bins=bins, hue_order=categories, ax=x_hist, legend=False)

    pct_axis.axhline(y=80, color='k', linestyle='--', alpha = 0.5)
    

    main_ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    main_ax.set_xlabel("Position (cm)")
    main_ax.set_ylabel("Trial")
    main_ax.set_xlim(0, lick["Position"].max() + 10)
    pct_axis.set_ylabel("% trials with one lick")
    pct_axis.set_xlabel("")
    pct_axis.set_ylim(0,100)
    x_hist.set_xlabel("")
    sns.despine()

Tk().withdraw() 
filename = askopenfilename() 
Timeline = mat73.loadmat(filename,only_include='Timeline/Results')
data = Timeline['Timeline']['Results']
lick_plot(data, first_lick=True, fig_size=(15,15), bin_num=60)
plt.show()