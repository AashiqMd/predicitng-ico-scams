
import logging
import pandas as pd
import preprocess
import random
import torch
import whitepapers

from torch.nn.utils.rnn import pad_sequence


log = logging.getLogger(__name__)


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
        'country_dispersed',
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
    'select': [
        'github_repositories_0719',
        'github_indicator_0719',
        # 'first_date_trading',
        # 'sector',
        'utility',
        'eth_blockchain',
        # 'sector_s',
        'original_data_flag',
        'available_supply',
        # 'sector_agg',
        'gaming',
        'num_emps2',
        'num_emps_linkedin_wm',
        'commercialization_new',
        'num_emps_website',
        'issuer_failed_new',
        'log_num_emps2_wm',
        'white_paper_final',
        'incentive_set_aside',
        'vesting',
        'budget',
        'founder_bckgr_crypto',
        'founder_bckgr_finance',
        'founder_bckgr_compsciIT',
        'founder_bckgr_entrep',
        # 'sector_agg_n',
        'presale',
        'had_goal_to_raise',
    ],
    "paper_sents": None  # separate data
}


def pad_collate(batch):
    try:
        (xx, yy) = zip(*batch)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yys = torch.tensor(yy).unsqueeze(1).float()
        return xx_pad, yys
    except ValueError:
        (xx, yy, ii) = zip(*batch)
        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yys = torch.tensor(yy).unsqueeze(1).float()
        iis = torch.tensor(ii).unsqueeze(1)
        return xx_pad, yys, iis


def get_dataset(task, split_pct, random_seed, tokenizer):
    log.info("setting up data...")
    if task not in TASKS:
        raise RuntimeError(f"unrecognized task {task}, choose from {TASKS.keys()}")

    if task.startswith("paper"):
        safe_base, scam_base, dictionary = whitepapers.load_sentences(tokenizer=tokenizer)

        # assign an ID to each document
        idx = 0
        safe, scam = [], []
        for doc in safe_base:
            safe.append((doc, idx))
            idx += 1
        for doc in scam_base:
            scam.append((doc, idx))
            idx += 1

        # split positive and negative documents into train set vs valid set
        safe_split = int(round(len(safe) * split_pct, 0))
        scam_split = int(round(len(scam) * split_pct, 0))

        rnd = random.Random(random_seed)
        rnd.shuffle(safe)
        rnd.shuffle(scam)

        safe_train_sents = [(torch.tensor(sent), 0, idx) for doc, idx in safe[:-safe_split] for sent in doc]
        safe_valid_sents = [(torch.tensor(sent), 0, idx) for doc, idx in safe[-safe_split:] for sent in doc]
        scam_train_sents = [(torch.tensor(sent), 1, idx) for doc, idx in scam[:-scam_split] for sent in doc]
        scam_valid_sents = [(torch.tensor(sent), 1, idx) for doc, idx in scam[-scam_split:] for sent in doc]

        train_sents = safe_train_sents + scam_train_sents
        valid_sents = safe_valid_sents + scam_valid_sents
        rnd.shuffle(train_sents)
        rnd.shuffle(valid_sents)

        train = WhitepaperDataset(train_sents)
        valid = WhitepaperDataset(valid_sents, include_ids=True)

        return train, valid, dictionary
    else:
        # we left merge sapkota into howell
        # howell_df = preprocess.preprocess_howell(log=False)
        # sapkota_df = preprocess.preprocess_sapkota(log=False)
        # merged_df = preprocess.merge(howell_df, sapkota_df, how="left", log=False)
        # merged_df = merged_df[TASKS[task] + [SCAM]].dropna()
        merged_df = preprocess.load_final_merged()[TASKS[task] + [SCAM]]    

        # split positive and negative coins out
        safe = merged_df[merged_df[SCAM] == preprocess.NOT_SCAM].sample(frac=1.0, random_state=random_seed)
        scam = merged_df[merged_df[SCAM] != preprocess.NOT_SCAM].sample(frac=1.0, random_state=random_seed)

        # put split_pct of each coin into train set vs valid set
        safe_split = int(round(len(safe) * split_pct, 0))
        neg_split = int(round(len(scam) * split_pct, 0))
        train_data = pd.concat([safe.iloc[:-safe_split], scam.iloc[:-neg_split]]).sample(frac=1.0, random_state=random_seed)
        valid_data = pd.concat([safe.iloc[-safe_split:], scam.iloc[-neg_split:]]).sample(frac=1.0, random_state=random_seed)

        train = HowellDataset(train_data)
        valid = HowellDataset(valid_data)
        return train, valid, None  # None means no dictionary (no text / embeddings)


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


class WhitepaperDataset(torch.utils.data.Dataset):
    def __init__(self, data, include_ids=False):
        self.data = data
        self.include_ids = include_ids
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        if not self.include_ids:
            return ex[0], ex[1]
        return ex


if __name__ == "__main__":
    t, v, d = get_dataset("paper_sents", 0.2, 42)
    ex = t[2]
    print(ex)
    if d is not None:
        print(d.vec2txt(ex[0]))

