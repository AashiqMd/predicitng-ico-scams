
import pandas as pd
import preprocess
import torch


SCAM = "Type"

TASKS = {
    "country": [
        'country_usa',
        'country_china',
        'country_canada',
        'country_russia',
        'country_singapore',
        'country_switzerland',
        'country_israel',
        'country_uk',
        'country_hk',
    ],
    "full": [
        "github_indicator_0719",
        "utility",
        "num_emps_linkedin_wm",
        "num_emps2_wm",
        "commercialization_new",
        "listed_original",
        "one_per_token",
        "listed",
        "issuer_failed_new",
        "commercialization",
        "log_num_emps2_wm",
        "white_paper_final",
        "incentive_set_aside",
        "vesting",
        "budget",
        "VC",
        "presale",
        "had_goal_to_raise",
    ],
    "extended": [
        'github_indicator_0719',
        # 'ico_failed',  # we know this is disjoint from scams
        'utility',
        # 'original_data_flag',  # dupe of data?
        'ads',
        'data',
        'enterp',
        'newblockchain',
        'pay',
        'trading',
        'asset',
        'gaming',
        'num_emps_linkedin_wm',
        'num_emps2_wm', 
        'commercialization_new',
        'listed_original',
        'one_per_token',
        'listed',
        'issuer_failed_new',
        'commercialization',
        'log_num_emps2_wm',
        'white_paper_final',
        'incentive_set_aside',
        'vesting',
        'budget',
        'VC',
        'country_usa',
        'country_china',
        'country_canada',
        'country_russia',
        'country_singapore',
        'country_switzerland',
        'country_israel',
        'country_uk',
        'country_hk',
        'country_dispersed',
        'presale',
        'had_goal_to_raise',
    ],
    "whitepapers": None  # separate data
}

def get_dataset(task, split_pct, random_seed):
    if task not in TASKS:
        raise RuntimeError(f"unrecognized task {task}, choose from {TASKS.keys()}")

    if task == "whitepapers":
        return None
    else:
        # we left merge sapkota into howell
        howell_df = preprocess.preprocess_howell(log=False)
        sapkota_df = preprocess.preprocess_sapkota(log=False)
        merged_df = preprocess.merge(howell_df, sapkota_df, how="left", log=False)

        merged_df = merged_df[TASKS[task] + [SCAM]].dropna()

        pos = merged_df[merged_df[SCAM] == preprocess.NOT_SCAM].sample(frac=1.0, random_state=random_seed)
        neg = merged_df[merged_df[SCAM] != preprocess.NOT_SCAM].sample(frac=1.0, random_state=random_seed)

        pos_split = int(round(len(pos) * split_pct, 0))
        neg_split = int(round(len(neg) * split_pct, 0))

        train_data = pd.concat([pos.iloc[:-pos_split], neg.iloc[:-neg_split]]).sample(frac=1.0, random_state=random_seed)
        valid_data = pd.concat([pos.iloc[-pos_split:], neg.iloc[-neg_split:]]).sample(frac=1.0, random_state=random_seed)

        train = HowellDataset(train_data)
        valid = HowellDataset(valid_data)
        return train, valid


class HowellDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        
        x = torch.Tensor([item[x] for x in self.data.columns if x != SCAM])
        y = torch.Tensor([item[SCAM] != preprocess.NOT_SCAM])

        return x, y

if __name__ == "__main__":
    print(get_dataset("country", 0.2, 42)[1][0])

